from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.sensor.camera_sensor import CameraSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  heights = sensor.data.heights
  if heights.ndim == 3:
    return heights.amin(dim=-1)
  if heights.ndim == 2:
    return heights
  raise ValueError(
    "foot_height expects terrain clearance samples with shape [B, F] or "
    f"[B, F, N], got {tuple(heights.shape)}"
  )


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  forces_flat = torch.nan_to_num(forces_flat, nan=0.0, posinf=0.0, neginf=0.0)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def depth_image(env: ManagerBasedRlEnv, sensor_name: str = "depth_camera") -> torch.Tensor:
  """Depth image from the forward-facing camera."""
  camera: CameraSensor = env.scene[sensor_name]
  assert camera.data.depth is not None, f"Sensor '{sensor_name}' has no depth data"
  return camera.data.depth.squeeze(-1)


class DepthBuffer:
  """Depth image buffer updated at a lower policy-step rate."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg, env
    self._buffer: torch.Tensor | None = None
    self._last_update_step: int | None = None
    self._invalid_env_ids: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = "depth_camera",
    buffer_size: int = 5,
    update_period: int = 5,
  ) -> torch.Tensor:
    if buffer_size < 1:
      raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
    if update_period < 1:
      raise ValueError(f"update_period must be >= 1, got {update_period}")

    step = int(getattr(env, "common_step_counter", 0))
    needs_init = self._buffer is None or self._buffer.shape[1] != buffer_size
    needs_reset_fill = self._invalid_env_ids is not None
    needs_periodic_update = (
      self._last_update_step is None
      or step - self._last_update_step >= update_period
    )

    if not (needs_init or needs_reset_fill or needs_periodic_update):
      assert self._buffer is not None
      return self._buffer

    frame = depth_image(env, sensor_name=sensor_name)

    if needs_init:
      self._buffer = frame.unsqueeze(1).repeat(
        1, buffer_size, *(1 for _ in frame.shape[1:])
      )
      self._last_update_step = step
      self._invalid_env_ids = None
      return self._buffer

    if needs_reset_fill:
      assert self._invalid_env_ids is not None
      env_ids = self._invalid_env_ids.to(device=frame.device, dtype=torch.long)
      self._buffer[env_ids] = frame[env_ids].unsqueeze(1).expand(
        -1, buffer_size, *frame.shape[1:]
      )
      self._invalid_env_ids = None

    if needs_periodic_update:
      self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
      self._buffer[:, -1] = frame
      self._last_update_step = step

    return self._buffer

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      self._buffer = None
      self._last_update_step = None
      self._invalid_env_ids = None
      return
    if self._buffer is None or env_ids.numel() == 0:
      return
    env_ids = env_ids.to(device=self._buffer.device, dtype=torch.long)
    if self._invalid_env_ids is None:
      self._invalid_env_ids = env_ids
    else:
      self._invalid_env_ids = torch.unique(
        torch.cat((self._invalid_env_ids, env_ids))
      )


depth_buffer = DepthBuffer


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


def _terrain_clearance_samples(
  sensor: TerrainHeightSensor | RayCastSensor,
) -> torch.Tensor:
  if isinstance(sensor, TerrainHeightSensor):
    height_samples = sensor.data.heights
    if height_samples.ndim != 3:
      raise ValueError(
        "terrain_roughness_indicator requires unreduced height samples with shape "
        f"[B, F, N], got {tuple(height_samples.shape)}"
      )
    return height_samples

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
    "terrain_roughness_indicator requires a TerrainHeightSensor or RayCastSensor, "
    f"got {type(sensor).__name__}"
  )


def terrain_roughness_indicator(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float = 0.127,
  gate_min: float = 0.10,
  gate_max: float = 0.40,
  grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Roughness gate from terrain clearance samples."""
  sensor = env.scene[sensor_name]
  height_samples = _terrain_clearance_samples(sensor)
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
  return gate.unsqueeze(-1)


def _normalize_to_unit_range(
  value: torch.Tensor, lower: float, upper: float
) -> torch.Tensor:
  scaled = 2.0 * (value - lower) / (upper - lower) - 1.0
  return torch.clamp(scaled, -1.0, 1.0)


def domain_randomization_delta_quantity(
  env: ManagerBasedRlEnv,
  wheel_friction_event: str = "wheel_friction",
  encoder_bias_event: str = "encoder_bias",
  base_com_event: str = "base_com",
) -> torch.Tensor:
  """Normalized domain-randomization quantities visible to the policy."""
  wheel_friction_cfg = env.event_manager.get_term_cfg(wheel_friction_event)
  friction_asset_cfg: SceneEntityCfg = wheel_friction_cfg.params["asset_cfg"]
  wheel_friction_range = wheel_friction_cfg.params["ranges"]
  friction_asset = env.scene[friction_asset_cfg.name]
  wheel_geom_ids = friction_asset.indexing.geom_ids[friction_asset_cfg.geom_ids]
  wheel_friction = env.sim.model.geom_friction[:, wheel_geom_ids, 0]
  wheel_friction = _normalize_to_unit_range(
    wheel_friction,
    wheel_friction_range[0],
    wheel_friction_range[1],
  )

  encoder_bias_cfg = env.event_manager.get_term_cfg(encoder_bias_event)
  encoder_asset_cfg: SceneEntityCfg = encoder_bias_cfg.params["asset_cfg"]
  encoder_bias_range = encoder_bias_cfg.params["bias_range"]
  encoder_asset = env.scene[encoder_asset_cfg.name]
  encoder_bias = encoder_asset.data.encoder_bias[:, encoder_asset_cfg.joint_ids]
  encoder_bias = _normalize_to_unit_range(
    encoder_bias,
    encoder_bias_range[0],
    encoder_bias_range[1],
  )

  base_com_cfg = env.event_manager.get_term_cfg(base_com_event)
  base_com_asset_cfg: SceneEntityCfg = base_com_cfg.params["asset_cfg"]
  base_com_ranges = base_com_cfg.params["ranges"]
  base_com_asset = env.scene[base_com_asset_cfg.name]
  base_body_ids = base_com_asset.indexing.body_ids[base_com_asset_cfg.body_ids]
  current_body_ipos = env.sim.model.body_ipos[:, base_body_ids, :].reshape(
    env.num_envs, -1
  )
  default_body_ipos = env.sim.get_default_field("body_ipos")[base_body_ids, :].reshape(
    1, -1
  )
  base_com_delta = current_body_ipos - default_body_ipos
  base_com_delta = torch.stack(
    [
      _normalize_to_unit_range(base_com_delta[:, axis], *base_com_ranges[axis])
      for axis in range(3)
    ],
    dim=1,
  )

  return torch.cat((wheel_friction, encoder_bias, base_com_delta), dim=1)
