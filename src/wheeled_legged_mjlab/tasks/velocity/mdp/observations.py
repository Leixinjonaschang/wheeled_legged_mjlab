from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
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
  return sensor.data.heights


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
