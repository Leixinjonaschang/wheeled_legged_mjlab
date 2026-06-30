from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, wrap_to_pi
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from .commands import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _TerrainRoughnessStats(NamedTuple):
  jump: torch.Tensor
  curvature: torch.Tensor
  foot_roughness: torch.Tensor
  robot_roughness: torch.Tensor
  gate: torch.Tensor


def _roughness_gate_active(
  gate: torch.Tensor,
  threshold: float = 0.2,
) -> torch.Tensor:
  return (gate > threshold).float()


def _roughness_gate_inactive(
  gate: torch.Tensor,
  threshold: float = 0.2,
) -> torch.Tensor:
  return 1.0 - _roughness_gate_active(gate, threshold)


def _resolve_grid_shape(
  num_samples: int,
  grid_shape: tuple[int, int] | None,
) -> tuple[int, int]:
  if grid_shape is not None:
    rows, cols = grid_shape
    if rows * cols != num_samples:
      raise ValueError(
        f"grid_shape={grid_shape} does not match {num_samples} height samples"
      )
    return rows, cols

  side = int(num_samples**0.5)
  if side * side != num_samples:
    raise ValueError(
      f"Cannot infer a square grid from {num_samples} height samples; "
      "pass grid_shape explicitly."
    )
  return side, side


def _terrain_roughness_from_height_samples(
  height_samples: torch.Tensor,
  *,
  wheel_radius: float,
  gate_min: float,
  gate_max: float,
  grid_shape: tuple[int, int] | None,
) -> _TerrainRoughnessStats:
  """Compute roughness from per-wheel local clearance samples."""
  if height_samples.ndim != 3:
    raise ValueError(
      "terrain roughness requires unreduced height samples with shape [B, F, N], "
      f"got {tuple(height_samples.shape)}"
    )
  if wheel_radius <= 0.0:
    raise ValueError(f"wheel_radius ({wheel_radius}) must be positive")
  if gate_max <= gate_min:
    raise ValueError(
      f"gate_max ({gate_max}) must be greater than gate_min ({gate_min})"
    )

  height_samples = torch.nan_to_num(height_samples, nan=0.0, posinf=0.0, neginf=0.0)
  rows, cols = _resolve_grid_shape(height_samples.shape[-1], grid_shape)
  grid = height_samples.view(height_samples.shape[0], height_samples.shape[1], rows, cols)
  zeros = torch.zeros_like(height_samples[..., 0])

  if cols > 1:
    jump_x = torch.abs(grid[..., 1:] - grid[..., :-1]).amax(dim=(-1, -2))
  else:
    jump_x = zeros
  if rows > 1:
    jump_y = torch.abs(grid[..., 1:, :] - grid[..., :-1, :]).amax(dim=(-1, -2))
  else:
    jump_y = zeros
  jump = torch.maximum(jump_x, jump_y)

  if cols > 2:
    curvature_x = torch.abs(
      grid[..., :, 2:] - 2.0 * grid[..., :, 1:-1] + grid[..., :, :-2]
    ).amax(dim=(-1, -2))
  else:
    curvature_x = zeros
  if rows > 2:
    curvature_y = torch.abs(
      grid[..., 2:, :] - 2.0 * grid[..., 1:-1, :] + grid[..., :-2, :]
    ).amax(dim=(-1, -2))
  else:
    curvature_y = zeros
  curvature = torch.maximum(curvature_x, curvature_y)

  foot_roughness = torch.maximum(jump / wheel_radius, curvature / wheel_radius)
  robot_roughness = foot_roughness.max(dim=1).values
  u = torch.clamp((robot_roughness - gate_min) / (gate_max - gate_min), 0.0, 1.0)
  gate = u * u * (3.0 - 2.0 * u)
  return _TerrainRoughnessStats(
    jump=jump,
    curvature=curvature,
    foot_roughness=foot_roughness,
    robot_roughness=robot_roughness,
    gate=gate,
  )


def _terrain_clearance_samples(
  sensor: TerrainHeightSensor | RayCastSensor,
) -> torch.Tensor:
  if isinstance(sensor, TerrainHeightSensor):
    return sensor.data.heights

  if isinstance(sensor, RayCastSensor):
    data = sensor.data
    f_count = sensor.num_frames
    n_count = sensor.num_rays_per_frame
    batch_size = data.distances.shape[0]
    frame_z = data.frame_pos_w[:, :, 2:3]
    hit_z = data.hit_pos_w[..., 2].view(batch_size, f_count, n_count)
    heights = frame_z - hit_z
    miss_mask = data.distances.view(batch_size, f_count, n_count) < 0
    return torch.where(
      miss_mask,
      torch.full_like(heights, sensor.cfg.max_distance),
      heights,
    )

  raise TypeError(
    "terrain roughness requires a TerrainHeightSensor or RayCastSensor, "
    f"got {type(sensor).__name__}"
  )


def _terrain_roughness_from_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  grid_shape: tuple[int, int] | None = None,
  log: bool = True,
) -> _TerrainRoughnessStats:
  sensor = env.scene[sensor_name]
  stats = _terrain_roughness_from_height_samples(
    _terrain_clearance_samples(sensor),
    wheel_radius=wheel_radius,
    gate_min=gate_min,
    gate_max=gate_max,
    grid_shape=grid_shape,
  )
  if log:
    log_data = env.extras.setdefault("log", {})
    log_data["Metrics/roughness_mean"] = stats.foot_roughness.mean()
    if stats.foot_roughness.shape[1] >= 2:
      log_data["Metrics/roughness_left_mean"] = stats.foot_roughness[:, 0].mean()
      log_data["Metrics/roughness_right_mean"] = stats.foot_roughness[:, 1].mean()
    log_data["Metrics/roughness_max_mean"] = stats.robot_roughness.mean()
    log_data["Metrics/roughness_lambda_mean"] = stats.gate.mean()
    log_data["Metrics/roughness_jump_over_R_mean"] = (
      stats.jump / wheel_radius
    ).mean()
    log_data["Metrics/roughness_curvature_over_R_mean"] = (
      stats.curvature / wheel_radius
    ).mean()
  return stats


def _command_active(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None, f"Command '{command_name}' not found."
  command = (
    command_term.command_w
    if isinstance(command_term, UniformVelocityCommand)
    else env.command_manager.get_command(command_name)
  )
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  return (linear_norm + angular_norm > command_threshold).float()


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None, f"Command '{command_name}' not found."
  if isinstance(command_term, UniformVelocityCommand):
    command_xy = command_term.command_w[:, :2]
    actual = asset.data.root_link_lin_vel_w
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_xy = command[:, :2]
    actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command_xy - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of the commanded yaw rate."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  return torch.exp(-z_error / std**2)


def base_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base roll and pitch angular velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def track_heading(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  command_norm_threshold: float = 0.2,
) -> torch.Tensor:
  """Reward the base heading for aligning with the commanded world heading."""
  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, UniformVelocityCommand)

  command_speed = torch.norm(command_term.command_w[:, :2], dim=1)
  active = command_speed > command_norm_threshold
  heading_error = wrap_to_pi(
    command_term.heading_target - command_term.robot.data.heading_w
  )
  reward = torch.exp(-torch.square(heading_error) / std**2)
  return reward * active.float()


class heading_progress:
  """Reward reductions in absolute heading error between consecutive steps."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Parameters are passed to __call__ by the manager.
    self._prev_abs_heading_error = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._has_prev = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._last_command_counter = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_progress: float = 0.05,
  ) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, UniformVelocityCommand)
    assert command_term.cfg.heading_command

    heading_error = wrap_to_pi(
      command_term.heading_target - command_term.robot.data.heading_w
    )
    abs_heading_error = torch.abs(heading_error)

    command_changed = command_term.command_counter != self._last_command_counter
    active = command_term.is_heading_env & ~command_term.is_standing_env
    valid = self._has_prev & ~command_changed & active

    progress_scale = max(max_progress, 1.0e-6)
    progress = self._prev_abs_heading_error - abs_heading_error
    progress_reward = torch.clamp(progress, min=0.0, max=progress_scale)
    progress_reward = progress_reward / progress_scale
    progress_reward = progress_reward * valid.float()

    log_data = env.extras.setdefault("log", {})
    log_data["Metrics/heading_progress_mean"] = progress_reward.mean()

    self._prev_abs_heading_error = abs_heading_error.detach()
    self._has_prev[:] = True
    self._last_command_counter = command_term.command_counter.clone()
    return progress_reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self._prev_abs_heading_error[env_ids] = 0.0
    self._has_prev[env_ids] = False
    self._last_command_counter[env_ids] = -1


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  lin_threshold: float = 0.05,
  ang_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize drifting when the sampled velocity command is near zero."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  lin_command_norm = torch.norm(command[:, :2], dim=1)
  ang_command_norm = torch.abs(command[:, 2])
  still_command = (lin_command_norm < lin_threshold) & (
    ang_command_norm < ang_threshold
  )

  lin_drift = torch.sum(torch.abs(asset.data.root_link_lin_vel_w[:, :2]), dim=1)
  yaw_drift = torch.abs(asset.data.root_link_ang_vel_w[:, 2])
  return (lin_drift + yaw_drift) * still_command.float()


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_history = torch.nan_to_num(
      data.force_history, nan=0.0, posinf=0.0, neginf=0.0
    )
    force_mag = torch.norm(force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = None,
  terrain_sample: str = "mean",
  deadband: float = 0.0,
) -> torch.Tensor:
  """Penalize base height error outside a symmetric deadband using an L2 kernel.

  When a terrain raycast sensor is provided, height is measured relative to the
  local terrain under the scan instead of world z. Errors inside the deadband
  produce zero cost; only the excess error is penalized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  if sensor_name is None:
    base_height = base_z
  else:
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, RayCastSensor):
      raise TypeError(
        "base_height_l2 terrain-relative mode requires a RayCastSensor, "
        f"got {type(sensor).__name__}"
      )
    data = sensor.data
    hit_z = data.hit_pos_w[..., 2].view(data.distances.shape[0], -1)
    valid_hit = data.distances >= 0
    if terrain_sample == "center":
      center_id = hit_z.shape[1] // 2
      ground_z = torch.where(
        valid_hit[:, center_id],
        hit_z[:, center_id],
        torch.zeros_like(base_z),
      )
    elif terrain_sample == "mean":
      hit_z = torch.where(valid_hit, hit_z, torch.zeros_like(hit_z))
      hit_count = valid_hit.float().sum(dim=1)
      ground_z = hit_z.sum(dim=1) / torch.clamp(hit_count, min=1.0)
      ground_z = torch.where(hit_count > 0, ground_z, torch.zeros_like(ground_z))
    else:
      raise ValueError(f"Unsupported terrain_sample: {terrain_sample}")
    base_height = base_z - ground_z
  height_error = torch.clamp(
    torch.abs(base_height - target_height) - deadband,
    min=0.0,
  )
  return torch.square(height_error)


def joint_power_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize mechanical joint power with an L1 kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_power = asset.data.qfrc_actuator[:, asset_cfg.joint_ids] * asset.data.joint_vel[
    :, asset_cfg.joint_ids
  ]
  return torch.sum(torch.abs(joint_power), dim=1)


def _body_positions_in_base_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  base_pos_w = asset.data.root_link_pos_w[:, None, :]
  base_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, body_pos_w.shape[1], -1
  )
  return quat_apply_inverse(base_quat_w, body_pos_w - base_pos_w)


def wheel_lateral_symmetry(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward left/right wheels for staying laterally symmetric in the base frame."""
  wheel_pos_b = _body_positions_in_base_frame(env, asset_cfg)
  lateral_error = torch.abs(wheel_pos_b[:, 0, 1]) - torch.abs(wheel_pos_b[:, 1, 1])
  return torch.exp(-torch.square(lateral_error) / std**2)


def wheel_x_alignment(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize front-back wheel misalignment in the base frame."""
  wheel_pos_b = _body_positions_in_base_frame(env, asset_cfg)
  return torch.abs(wheel_pos_b[:, 0, 0] - wheel_pos_b[:, 1, 0])


def non_rough_wheel_lateral_symmetry(
  env: ManagerBasedRlEnv,
  roughness_sensor_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  roughness_gate_threshold: float = 0.2,
  grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Reward wheel lateral symmetry only when terrain is not rough."""
  stats = _terrain_roughness_from_sensor(
    env,
    roughness_sensor_name,
    wheel_radius=wheel_radius,
    gate_min=gate_min,
    gate_max=gate_max,
    grid_shape=grid_shape,
  )
  non_rough_active = _roughness_gate_inactive(stats.gate, roughness_gate_threshold)
  return non_rough_active * wheel_lateral_symmetry(env, std, asset_cfg)


def non_rough_wheel_x_alignment(
  env: ManagerBasedRlEnv,
  roughness_sensor_name: str,
  asset_cfg: SceneEntityCfg,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  roughness_gate_threshold: float = 0.2,
  grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Penalize front-back wheel misalignment only when terrain is not rough."""
  stats = _terrain_roughness_from_sensor(
    env,
    roughness_sensor_name,
    wheel_radius=wheel_radius,
    gate_min=gate_min,
    gate_max=gate_max,
    grid_shape=grid_shape,
  )
  non_rough_active = _roughness_gate_inactive(stats.gate, roughness_gate_threshold)
  return non_rough_active * wheel_x_alignment(env, asset_cfg)


def wheel_distance(
  env: ManagerBasedRlEnv,
  min_distance: float,
  max_distance: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize wheel distance outside an allowed range in the horizontal plane."""
  wheel_pos_b = _body_positions_in_base_frame(env, asset_cfg)
  distance = torch.norm(wheel_pos_b[:, 0, :2] - wheel_pos_b[:, 1, :2], dim=1)
  return torch.clip(min_distance - distance, min=0.0) + torch.clip(
    distance - max_distance,
    min=0.0,
  )


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


class wheel_air_time_balance:
  """Penalize episode-level left/right wheel air-time imbalance."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._cumulative_air_time = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    min_total_air_time: float = 1.0,
    balance_tolerance: float = 0.2,
  ) -> torch.Tensor:
    if min_total_air_time <= 0.0:
      raise ValueError("min_total_air_time must be positive")
    if not 0.0 <= balance_tolerance < 1.0:
      raise ValueError("balance_tolerance must be in [0, 1)")

    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    if found.shape[1] != 2:
      raise ValueError(
        "wheel_air_time_balance expects exactly two contact tracks, "
        f"got {found.shape[1]}"
      )

    in_air = (found == 0).float()
    self._cumulative_air_time += in_air * env.step_dt

    total_air_time = self._cumulative_air_time.sum(dim=1)
    air_time_diff = torch.abs(
      self._cumulative_air_time[:, 0] - self._cumulative_air_time[:, 1]
    )
    imbalance_ratio = air_time_diff / torch.clamp(total_air_time, min=1.0e-6)

    # Ignore small duty-factor differences and ramp the cost in only after enough
    # swing time has been observed.  This avoids penalizing the first normal swing
    # of an episode while preserving a persistent cost for one-legged standing.
    excess_ratio = torch.clamp(
      (imbalance_ratio - balance_tolerance) / (1.0 - balance_tolerance),
      min=0.0,
      max=1.0,
    )
    activation = torch.clamp(total_air_time / min_total_air_time, 0.0, 1.0)
    cost = activation * torch.square(excess_ratio)

    log_data = env.extras.setdefault("log", {})
    log_data["Metrics/wheel_episode_air_time_left_mean"] = (
      self._cumulative_air_time[:, 0].mean()
    )
    log_data["Metrics/wheel_episode_air_time_right_mean"] = (
      self._cumulative_air_time[:, 1].mean()
    )
    log_data["Metrics/wheel_episode_air_time_diff_mean"] = air_time_diff.mean()
    log_data["Metrics/wheel_episode_air_time_imbalance_ratio_mean"] = (
      imbalance_ratio.mean()
    )
    log_data["Metrics/wheel_air_time_balance_cost_mean"] = cost.mean()
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._cumulative_air_time[env_ids] = 0.0


def standing_forward_wheel_air_time(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str,
  roughness_sensor_name: str,
  command_name: str,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  roughness_gate_threshold: float = 0.2,
  grid_shape: tuple[int, int] | None = None,
  max_time: float = 0.5,
  standing_scale: float = 2.5,
  forward_scale: float = 1.0,
  lin_threshold: float = 0.05,
  ang_threshold: float = 0.05,
  forward_speed_threshold: float = 0.05,
  forward_lateral_threshold: float = 0.05,
  forward_ang_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize standing air time globally and forward air time only on non-rough terrain."""
  stats = _terrain_roughness_from_sensor(
    env,
    roughness_sensor_name,
    wheel_radius=wheel_radius,
    gate_min=gate_min,
    gate_max=gate_max,
    grid_shape=grid_shape,
  )
  non_rough_active = _roughness_gate_inactive(stats.gate, roughness_gate_threshold)

  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  current_air_time = contact_sensor.data.current_air_time
  assert current_air_time is not None
  air_time = torch.sum(torch.clamp(current_air_time, max=max_time), dim=1)

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None, f"Command '{command_name}' not found."
  if isinstance(command_term, UniformVelocityCommand):
    standing = command_term.is_standing_env
    forward = command_term.is_forward_env & ~standing
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    standing = (linear_norm < lin_threshold) & (angular_norm < ang_threshold)
    forward = (
      (command[:, 0] > forward_speed_threshold)
      & (torch.abs(command[:, 1]) < forward_lateral_threshold)
      & (torch.abs(command[:, 2]) < forward_ang_threshold)
      & ~standing
    )

  standing_cost = air_time * standing.float() * standing_scale
  forward_cost = air_time * forward.float() * forward_scale * non_rough_active
  cost = standing_cost + forward_cost

  log_data = env.extras.setdefault("log", {})
  log_data["Metrics/standing_forward_wheel_air_time_mean"] = cost.mean()
  log_data["Metrics/standing_wheel_air_time_mean"] = standing_cost.mean()
  log_data["Metrics/non_rough_forward_wheel_air_time_mean"] = forward_cost.mean()
  return cost


def rough_wheel_usage(
  env: ManagerBasedRlEnv,
  roughness_sensor_name: str,
  asset_cfg: SceneEntityCfg,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  roughness_gate_threshold: float = 0.2,
  grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Penalize wheel speed only when local wheel terrain is rough."""
  stats = _terrain_roughness_from_sensor(
    env,
    roughness_sensor_name,
    wheel_radius=wheel_radius,
    gate_min=gate_min,
    gate_max=gate_max,
    grid_shape=grid_shape,
  )
  asset: Entity = env.scene[asset_cfg.name]
  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  rough_active = _roughness_gate_active(stats.gate, roughness_gate_threshold)
  return rough_active * torch.sum(torch.square(wheel_vel), dim=1)


def rough_wheel_foot_clearance(
  env: ManagerBasedRlEnv,
  clearance_sensor_name: str,
  roughness_sensor_name: str,
  contact_sensor_name: str,
  command_name: str,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  roughness_gate_threshold: float = 0.2,
  grid_shape: tuple[int, int] | None = None,
  clearance_grid_shape: tuple[int, int] | None = None,
  base_target_height: float = 0.06,
  range_scale: float = 0.5,
  max_target_height: float = 0.18,
  target_std: float = 0.04,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward wheel-foot swing clearance toward a roughness-aware local target."""
  stats = _terrain_roughness_from_sensor(
    env,
    roughness_sensor_name,
    wheel_radius=wheel_radius,
    gate_min=gate_min,
    gate_max=gate_max,
    grid_shape=grid_shape,
  )

  clearance_sensor = env.scene[clearance_sensor_name]
  assert isinstance(clearance_sensor, TerrainHeightSensor), (
    "rough_wheel_foot_clearance requires a TerrainHeightSensor, "
    f"got {type(clearance_sensor).__name__}"
  )
  wheel_center_clearance = clearance_sensor.data.heights
  if wheel_center_clearance.ndim == 3:
    _resolve_grid_shape(wheel_center_clearance.shape[-1], clearance_grid_shape)
    local_height_range = (
      wheel_center_clearance.max(dim=-1).values
      - wheel_center_clearance.min(dim=-1).values
    )
    wheel_center_clearance = wheel_center_clearance.amin(dim=-1)
  elif wheel_center_clearance.ndim == 2:
    local_height_range = torch.zeros_like(wheel_center_clearance)
  else:
    raise ValueError(
      "rough_wheel_foot_clearance expects clearance samples with shape "
      f"[B, F] or [B, F, N], got {tuple(wheel_center_clearance.shape)}"
    )
  wheel_foot_clearance = torch.clamp(wheel_center_clearance - wheel_radius, min=0.0)

  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  assert contact_sensor.data.found is not None
  in_air = (contact_sensor.data.found == 0).float()
  active = _command_active(env, command_name, command_threshold)

  target = base_target_height + range_scale * local_height_range
  target = torch.clamp(target, max=max_target_height)
  error = wheel_foot_clearance - target
  reward_per_foot = torch.exp(-torch.square(error) / (target_std**2))
  rough_active = _roughness_gate_active(stats.gate, roughness_gate_threshold)
  reward = torch.sum(reward_per_foot * in_air, dim=1) * rough_active * active

  log_data = env.extras.setdefault("log", {})
  log_data["Metrics/rough_wheel_foot_clearance_mean"] = (
    wheel_foot_clearance.mean()
  )
  log_data["Metrics/rough_wheel_foot_clearance_target_mean"] = target.mean()
  return reward


def rough_foot_clearance(*args, **kwargs) -> torch.Tensor:
  """Backward-compatible alias for old configs."""
  return rough_wheel_foot_clearance(*args, **kwargs)


def rough_contact_pattern(
  env: ManagerBasedRlEnv,
  roughness_sensor_name: str,
  contact_sensor_name: str,
  command_name: str,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  roughness_gate_threshold: float = 0.2,
  grid_shape: tuple[int, int] | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward rough-terrain contact patterns with a negative value for bad modes."""
  stats = _terrain_roughness_from_sensor(
    env,
    roughness_sensor_name,
    wheel_radius=wheel_radius,
    gate_min=gate_min,
    gate_max=gate_max,
    grid_shape=grid_shape,
  )

  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = contact_sensor.data.found > 0
  contact_count = torch.sum(in_contact.float(), dim=1)
  num_contacts = in_contact.shape[1]
  double_contact = contact_count == num_contacts
  no_contact = contact_count == 0
  active = _command_active(env, command_name, command_threshold)
  rough_active = _roughness_gate_active(stats.gate, roughness_gate_threshold)
  reward = -(double_contact.float() + no_contact.float()) * rough_active * active

  log_data = env.extras.setdefault("log", {})
  log_data["Metrics/rough_double_contact_mean"] = double_contact.float().mean()
  log_data["Metrics/rough_no_contact_mean"] = no_contact.float().mean()
  log_data["Metrics/rough_single_contact_mean"] = (
    (contact_count == 1).float().mean()
  )
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = torch.nan_to_num(sensor_data.force, nan=0.0, posinf=0.0, neginf=0.0)
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
