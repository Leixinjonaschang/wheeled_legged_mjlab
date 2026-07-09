from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
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
      center=centers[env_id].cpu().numpy(),
      radius=marker_radius,
      color=color,
      label="roughness_gate_marker",
    )


def draw_predicted_lin_vel(
  env: ManagerBasedRlEnv,
  visualizer: DebugVisualizer,
  *,
  asset_name: str = "robot",
  z_offset: float = 0.42,
  scale: float = 0.5,
) -> None:
  """Draw the representation policy's predicted base linear velocity."""
  predicted_lin_vel_b = getattr(env, "predicted_lin_vel_b", None)
  if predicted_lin_vel_b is None or asset_name not in env.scene.entities:
    return

  env_indices = list(visualizer.get_env_indices(env.num_envs))
  if not env_indices:
    return

  predicted_lin_vel_b = predicted_lin_vel_b.detach()
  if predicted_lin_vel_b.ndim != 2 or predicted_lin_vel_b.shape[1] < 2:
    return

  asset = env.scene[asset_name]
  headings = asset.data.heading_w
  cos_h = torch.cos(headings)
  sin_h = torch.sin(headings)

  pred_x_b = predicted_lin_vel_b[:, 0]
  pred_y_b = predicted_lin_vel_b[:, 1]
  pred_x_w = cos_h * pred_x_b - sin_h * pred_y_b
  pred_y_w = sin_h * pred_x_b + cos_h * pred_y_b

  base_pos_ws = asset.data.root_link_pos_w.detach().cpu().numpy()
  pred_lin_vel_ws = torch.stack((pred_x_w, pred_y_w), dim=1).cpu().numpy()

  for env_id in env_indices:
    base_pos_w = base_pos_ws[env_id]
    if np.linalg.norm(base_pos_w) < 1.0e-6:
      continue

    origin = base_pos_w + np.array([0.0, 0.0, z_offset])
    target = origin + np.array(
      [pred_lin_vel_ws[env_id, 0], pred_lin_vel_ws[env_id, 1], 0.0]
    ) * scale
    visualizer.add_arrow(
      origin,
      target,
      color=(1.0, 0.45, 0.0, 0.85),
      width=0.018,
    )
