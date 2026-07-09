from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnv

from wheeled_legged_mjlab.tasks.velocity.utils.visualizations import (
  draw_predicted_lin_vel,
  draw_roughness_gate_marker,
)

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class WheeledLeggedVelocityEnv(ManagerBasedRlEnv):
  """Velocity env with task-specific play-time debug visualizations."""

  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    super().update_visualizers(visualizer)
    draw_predicted_lin_vel(self, visualizer)
    draw_roughness_gate_marker(self, visualizer)
