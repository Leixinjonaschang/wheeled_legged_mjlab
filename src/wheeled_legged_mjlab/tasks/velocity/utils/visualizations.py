from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from wheeled_legged_mjlab.tasks.velocity.mdp.observations import (
  terrain_roughness_indicator,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _roughness_indicator_params(env: ManagerBasedRlEnv) -> dict[str, Any] | None:
  observations = getattr(env.cfg, "observations", {})
  for group_name in ("actor", "critic"):
    group = observations.get(group_name)
    if group is None:
      continue

    term = group.terms.get("roughness_indicator")
    if term is None:
      continue

    params = dict(term.params or {})
    sensor_name = params.get("sensor_name")
    if isinstance(sensor_name, str):
      return params

  return None


def draw_roughness_gate_marker(
  env: ManagerBasedRlEnv,
  visualizer: DebugVisualizer,
  *,
  asset_name: str = "robot",
  gate_threshold: float = 0.0,
  marker_height: float = 0.6,
  marker_radius: float = 0.08,
) -> None:
  """Draw a sphere above the robot when the roughness gate is available."""
  params = _roughness_indicator_params(env)
  if params is None:
    return

  sensor_name = params["sensor_name"]
  if sensor_name not in env.scene.sensors or asset_name not in env.scene.entities:
    return

  env_indices = list(visualizer.get_env_indices(env.num_envs))
  if not env_indices:
    return

  gate = terrain_roughness_indicator(env, **params).squeeze(-1)
  asset = env.scene[asset_name]
  offset = torch.tensor([0.0, 0.0, marker_height], device=env.device)
  centers = asset.data.root_link_pos_w + offset

  for env_id in env_indices:
    active = gate[env_id].item() > gate_threshold
    color = (0.0, 1.0, 0.0, 0.85) if active else (0.45, 0.45, 0.45, 0.65)
    visualizer.add_sphere(
      center=centers[env_id],
      radius=marker_radius,
      color=color,
      label="roughness_gate_marker",
    )
