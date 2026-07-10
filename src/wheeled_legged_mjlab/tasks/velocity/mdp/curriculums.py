from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")
_TERRAIN_COLUMN_ALIAS_SEPARATOR = "__"


def _logical_terrain_name(name: str) -> str:
  """Map a repeated terrain-column name back to its logical terrain type."""
  return name.split(_TERRAIN_COLUMN_ALIAS_SEPARATOR, maxsplit=1)[0]


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per config entry).
  # Repeated columns use a ``logical_name__index`` suffix and share metrics.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    type_indices_by_name: dict[str, list[int]] = {}
    for i, name in enumerate(sub_terrain_names):
      logical_name = _logical_terrain_name(name)
      type_indices_by_name.setdefault(logical_name, []).append(i)

    for name, type_indices in type_indices_by_name.items():
      mask = torch.zeros_like(types, dtype=torch.bool)
      for i in type_indices:
        mask |= types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def fell_over_limit_angle(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  termination_term_name: str,
  initial_limit_angle: float,
  final_limit_angle: float,
  ramp_steps: int,
) -> dict[str, torch.Tensor]:
  """Linearly relax the fall-over angle limit over global policy steps."""
  del env_ids  # The termination threshold is shared by all environments.
  progress = min(max(float(env.common_step_counter) / max(ramp_steps, 1), 0.0), 1.0)
  limit_angle = initial_limit_angle + progress * (
    final_limit_angle - initial_limit_angle
  )

  termination_cfg = env.termination_manager.get_term_cfg(termination_term_name)
  termination_cfg.params["limit_angle"] = limit_angle
  return {"limit_angle": torch.tensor(limit_angle, device=env.device)}


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
