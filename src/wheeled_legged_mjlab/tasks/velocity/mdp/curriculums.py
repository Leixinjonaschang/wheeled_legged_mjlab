from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")
_VELOCITY_CURRICULUM_STATE_ATTR = "_velocity_curriculum_state"


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def _get_velocity_curriculum_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  state = getattr(env, _VELOCITY_CURRICULUM_STATE_ATTR, None)
  if state is None:
    state = {
      "cmd_path": torch.zeros(env.num_envs, device=env.device, dtype=torch.float32),
      "progress": torch.zeros(env.num_envs, device=env.device, dtype=torch.float32),
      "tracking_error_path": torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float32
      ),
      "active_steps": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
    }
    setattr(env, _VELOCITY_CURRICULUM_STATE_ATTR, state)
  return cast(dict[str, torch.Tensor], state)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
  return numerator / torch.clamp(denominator, min=1.0e-6)


class velocity_curriculum_progress:
  """Accumulate per-episode command path and realized command-direction progress."""

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
    del cfg
    self._state = _get_velocity_curriculum_state(env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command '{command_name}' not found."
    if isinstance(command_term, UniformVelocityCommand):
      command_xy = command_term.command_w[:, :2]
      actual_xy = asset.data.root_link_lin_vel_w[:, :2]
    else:
      command = env.command_manager.get_command(command_name)
      assert command is not None, f"Command '{command_name}' not found."
      command_xy = command[:, :2]
      actual_xy = asset.data.root_link_lin_vel_b[:, :2]

    command_speed = torch.norm(command_xy, dim=1)
    active = command_speed > command_threshold
    active_f = active.float()
    command_dir = command_xy / torch.clamp(command_speed[:, None], min=1.0e-6)
    progress_step = torch.sum(actual_xy * command_dir, dim=1) * env.step_dt
    cmd_path_step = command_speed * env.step_dt
    tracking_error_step = torch.norm(actual_xy - command_xy, dim=1) * env.step_dt

    self._state["cmd_path"] += cmd_path_step * active_f
    self._state["progress"] += progress_step * active_f
    self._state["tracking_error_path"] += tracking_error_step * active_f
    self._state["active_steps"] += active.long()

    progress_ratio = _safe_ratio(self._state["progress"], self._state["cmd_path"])
    tracking_error_ratio = _safe_ratio(
      self._state["tracking_error_path"], self._state["cmd_path"]
    )
    log_data = env.extras.setdefault("log", {})
    log_data["Metrics/velocity_curriculum_progress_ratio_mean"] = (
      progress_ratio.mean()
    )
    log_data["Metrics/velocity_curriculum_tracking_error_ratio_mean"] = (
      tracking_error_ratio.mean()
    )
    return progress_ratio

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    for value in self._state.values():
      value[env_ids] = 0


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  min_command_path_ratio: float = 0.25,
  move_up_progress_ratio: float = 0.70,
  move_down_progress_ratio: float = 0.40,
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
  command_speed = torch.norm(command[env_ids, :2], dim=1)
  state = _get_velocity_curriculum_state(env)
  cmd_path = state["cmd_path"][env_ids]
  progress = state["progress"][env_ids]
  tracking_error_path = state["tracking_error_path"][env_ids]
  active_steps = state["active_steps"][env_ids]
  progress_ratio = _safe_ratio(progress, cmd_path)
  tracking_error_ratio = _safe_ratio(tracking_error_path, cmd_path)
  progress_distance = torch.clamp(progress, min=0.0)
  min_command_path = terrain_generator.size[0] * min_command_path_ratio

  # Robots that walked far enough progress to harder terrains.
  move_up = (
    (progress_distance > terrain_generator.size[0] / 2)
    & (progress_ratio > move_up_progress_ratio)
  )

  # Robots that make poor progress along the commanded path go to simpler terrains.
  move_down = (
    (cmd_path > min_command_path)
    & (progress_ratio < move_down_progress_ratio)
  )
  move_down = move_down & ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
    "move_up_rate": torch.mean(move_up.float()),
    "move_down_rate": torch.mean(move_down.float()),
    "distance_mean": torch.mean(distance),
    "command_speed_mean": torch.mean(command_speed),
    "cmd_path_mean": torch.mean(cmd_path),
    "progress_mean": torch.mean(progress),
    "progress_ratio_mean": torch.mean(progress_ratio),
    "tracking_error_ratio_mean": torch.mean(tracking_error_ratio),
    "active_steps_mean": torch.mean(active_steps.float()),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    selected_types = types[env_ids]
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])
      selected_mask = selected_types == i
      if selected_mask.any():
        result[f"{name}_move_up_rate"] = torch.mean(move_up[selected_mask].float())
        result[f"{name}_move_down_rate"] = torch.mean(
          move_down[selected_mask].float()
        )
        result[f"{name}_distance_mean"] = torch.mean(distance[selected_mask])
        result[f"{name}_command_speed_mean"] = torch.mean(
          command_speed[selected_mask]
        )
        result[f"{name}_cmd_path_mean"] = torch.mean(cmd_path[selected_mask])
        result[f"{name}_progress_mean"] = torch.mean(progress[selected_mask])
        result[f"{name}_progress_ratio_mean"] = torch.mean(
          progress_ratio[selected_mask]
        )
        result[f"{name}_tracking_error_ratio_mean"] = torch.mean(
          tracking_error_ratio[selected_mask]
        )
        result[f"{name}_active_steps_mean"] = torch.mean(
          active_steps[selected_mask].float()
        )

  return result


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
