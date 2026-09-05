from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import wrap_to_pi

from .commands import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def non_finite_physics(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate envs whose physics or sensor state has become non-finite."""
  data = env.sim.data
  bad = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
  for name in (
    "qpos",
    "qvel",
    "qacc",
    "qacc_warmstart",
    "sensordata",
    "actuator_force",
    "qfrc_actuator",
  ):
    tensor = getattr(data, name, None)
    if tensor is None:
      continue
    bad |= ~torch.isfinite(tensor).reshape(env.num_envs, -1).all(dim=1)

  for sensor in env.scene.sensors.values():
    if not isinstance(sensor, ContactSensor):
      continue
    sensor_data = sensor.data
    for name in (
      "found",
      "force",
      "torque",
      "dist",
      "pos",
      "normal",
      "tangent",
      "current_air_time",
      "last_air_time",
      "current_contact_time",
      "last_contact_time",
      "force_history",
      "torque_history",
      "dist_history",
    ):
      tensor = getattr(sensor_data, name, None)
      if tensor is None:
        continue
      bad |= ~torch.isfinite(tensor).reshape(env.num_envs, -1).all(dim=1)
  return bad


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


class world_command_tracking_failure:
  """Terminate sustained world-frame command tracking failures.

  Linear progress and velocity direction are measured in the fixed world frame
  of :class:`UniformVelocityCommand`. Heading is evaluated separately against
  its heading target. A command-change grace period and full-flight exemption
  prevent planned turns and airborne parkour phases from being cut short.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Parameters are passed to __call__ by the manager.
    self._progress_bad_steps = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._direction_bad_steps = torch.zeros_like(self._progress_bad_steps)
    self._heading_bad_steps = torch.zeros_like(self._progress_bad_steps)
    self._command_age_steps = torch.zeros_like(self._progress_bad_steps)
    self._last_command_counter = torch.full_like(self._progress_bad_steps, -1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    activation_step: int = 120_000,
    command_norm_threshold: float = 0.35,
    progress_deficit_threshold: float = 0.45,
    min_progress_ratio: float = 0.55,
    progress_duration_s: float = 0.4,
    actual_speed_threshold: float = 0.2,
    direction_angle_threshold_deg: float = 70.0,
    direction_duration_s: float = 0.3,
    heading_error_threshold_deg: float = 55.0,
    heading_duration_s: float = 0.6,
    heading_alignment_gate_deg: float = 45.0,
    command_grace_s: float = 2.5,
    contact_sensor_name: str = "wheels_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    assert isinstance(command_term, UniformVelocityCommand)
    command_counter = command_term.command_counter
    command_changed = command_counter != self._last_command_counter
    self._command_age_steps = torch.where(
      command_changed,
      torch.zeros_like(self._command_age_steps),
      self._command_age_steps + 1,
    )
    self._last_command_counter = command_counter.clone()

    asset: Entity = env.scene[asset_cfg.name]
    command_xy = command_term.command_w[:, :2]
    actual_xy = asset.data.root_link_lin_vel_w[:, :2]

    command_norm = torch.norm(command_xy, dim=1)
    actual_speed = torch.norm(actual_xy, dim=1)
    command_direction = command_xy / command_norm.unsqueeze(1).clamp_min(1.0e-6)
    progress_velocity = torch.sum(actual_xy * command_direction, dim=1)
    progress_deficit = torch.clamp(command_norm - progress_velocity, min=0.0)
    progress_ratio = progress_velocity / command_norm.clamp_min(1.0e-6)
    cos_angle = progress_velocity / (actual_speed + 1.0e-6)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    heading_error = torch.abs(
      wrap_to_pi(command_term.heading_target - command_term.robot.data.heading_w)
    )
    contact_sensor: ContactSensor = env.scene[contact_sensor_name]
    assert contact_sensor.data.found is not None
    grounded = torch.any(contact_sensor.data.found, dim=-1)

    progress_duration_steps = max(1, math.ceil(progress_duration_s / env.step_dt))
    direction_duration_steps = max(1, math.ceil(direction_duration_s / env.step_dt))
    heading_duration_steps = max(1, math.ceil(heading_duration_s / env.step_dt))
    command_grace_steps = max(0, math.ceil(command_grace_s / env.step_dt))
    command_active = (
      (env.common_step_counter >= activation_step)
      & (command_norm > command_norm_threshold)
      & (self._command_age_steps >= command_grace_steps)
    )
    heading_aligned = heading_error < math.radians(heading_alignment_gate_deg)
    linear_check_active = command_active & grounded & heading_aligned
    heading_check_active = (
      command_active
      & grounded
      & command_term.is_heading_env
      & ~command_term.is_standing_env
    )

    bad_progress = linear_check_active & (
      (progress_deficit > progress_deficit_threshold)
      | (progress_ratio < min_progress_ratio)
    )
    bad_direction = (
      linear_check_active
      & (actual_speed > actual_speed_threshold)
      & (cos_angle < math.cos(math.radians(direction_angle_threshold_deg)))
    )
    bad_heading = heading_check_active & (
      heading_error > math.radians(heading_error_threshold_deg)
    )
    self._progress_bad_steps = torch.where(
      bad_progress,
      self._progress_bad_steps + 1,
      torch.zeros_like(self._progress_bad_steps),
    )
    self._direction_bad_steps = torch.where(
      bad_direction,
      self._direction_bad_steps + 1,
      torch.zeros_like(self._direction_bad_steps),
    )
    self._heading_bad_steps = torch.where(
      bad_heading,
      self._heading_bad_steps + 1,
      torch.zeros_like(self._heading_bad_steps),
    )

    log_data = env.extras.setdefault("log", {})
    log_data["Metrics/world_command_tracking_progress_deficit_mean"] = (
      progress_deficit.mean()
    )
    log_data["Metrics/world_command_tracking_progress_bad_frac"] = (
      bad_progress.float().mean()
    )
    log_data["Metrics/world_command_tracking_direction_bad_frac"] = (
      bad_direction.float().mean()
    )
    log_data["Metrics/world_command_tracking_heading_error_deg_mean"] = (
      heading_error.mean() * (180.0 / math.pi)
    )
    log_data["Metrics/world_command_tracking_heading_bad_frac"] = (
      bad_heading.float().mean()
    )
    return (
      (self._progress_bad_steps >= progress_duration_steps)
      | (self._direction_bad_steps >= direction_duration_steps)
      | (self._heading_bad_steps >= heading_duration_steps)
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self._progress_bad_steps[env_ids] = 0
    self._direction_bad_steps[env_ids] = 0
    self._heading_bad_steps[env_ids] = 0
    self._command_age_steps[env_ids] = 0
    self._last_command_counter[env_ids] = -1


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate shortly after the robot leaves the effective generated terrain.

  ``margin`` is the allowed root displacement beyond the patch grid edge. The
  generator's flat outer border is excluded from the effective terrain footprint.
  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  # Use the generated grid shape (curriculum mode overrides cfg.num_cols with
  # len(sub_terrains)); the flat outer border is only a physical safety surface.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0])
  half_y = 0.5 * (num_cols * terrain_generator.size[1])
  limit_x = half_x + margin
  limit_y = half_y + margin

  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge
