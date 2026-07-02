from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

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


class velocity_direction_deviation:
  """Terminate sustained sideways/backward deviation from commanded xy velocity."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Parameters are passed to __call__ by the manager.
    self._bad_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._command_age_steps = torch.zeros_like(self._bad_steps)
    self._last_command_counter = torch.full_like(self._bad_steps, -1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    activation_step: int = 120_000,
    command_norm_threshold: float = 0.2,
    actual_speed_threshold: float = 0.15,
    angle_threshold_deg: float = 35.0,
    duration_s: float = 0.15,
    command_grace_s: float = 0.5,
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
    dot = torch.sum(command_xy * actual_xy, dim=1)
    cos_angle = dot / (command_norm * actual_speed + 1.0e-6)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    cos_angle_threshold = math.cos(math.radians(angle_threshold_deg))
    duration_steps = max(1, math.ceil(duration_s / env.step_dt))
    command_grace_steps = max(0, math.ceil(command_grace_s / env.step_dt))
    check_active = (
      (env.common_step_counter >= activation_step)
      & (command_norm > command_norm_threshold)
      & (actual_speed > actual_speed_threshold)
      & (self._command_age_steps >= command_grace_steps)
    )
    bad = check_active & (cos_angle < cos_angle_threshold)
    self._bad_steps = torch.where(
      bad,
      self._bad_steps + 1,
      torch.zeros_like(self._bad_steps),
    )

    log_data = env.extras.setdefault("log", {})
    angle = torch.acos(cos_angle) * (180.0 / math.pi)
    log_data["Metrics/velocity_direction_deviation_angle_mean"] = angle.mean()
    log_data["Metrics/velocity_direction_deviation_bad_frac"] = bad.float().mean()
    log_data["Metrics/velocity_direction_deviation_active"] = torch.tensor(
      float(env.common_step_counter >= activation_step),
      device=env.device,
    )

    return self._bad_steps >= duration_steps

  def reset(self, env_ids: torch.Tensor) -> None:
    self._bad_steps[env_ids] = 0
    self._command_age_steps[env_ids] = 0
    self._last_command_counter[env_ids] = -1


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

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
  # len(sub_terrains)), and include the flat border around the patch grid.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

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
