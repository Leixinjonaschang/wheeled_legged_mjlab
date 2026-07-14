from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    # High-level command is sampled in world yaw frame. The third column stores
    # the yaw-rate command derived from heading_target when heading mode is on.
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    """Body-frame command exposed to policy observations."""
    return self.vel_command_b

  @property
  def command_w(self) -> torch.Tensor:
    """World-frame linear command plus yaw-rate command."""
    return self.vel_command_w

  @property
  def command_heading_w(self) -> torch.Tensor:
    """High-level world-frame command: [lin_vel_x_w, lin_vel_y_w, heading]."""
    if self.cfg.heading_command:
      return torch.cat(
        (self.vel_command_w[:, :2], self.heading_target[:, None]),
        dim=1,
      )
    return self.vel_command_w

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_w[:, :2] - self.robot.data.root_link_lin_vel_w[:, :2],
        dim=-1,
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Straight-line envs: +x (50%), +y (25%), or -y (25%) in the world frame.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      speed = (
        self.vel_command_w[fwd_ids, 0].abs().clamp(min=0.3)
      )
      direction = torch.empty(len(fwd_ids), device=self.device).uniform_(0.0, 1.0)
      positive_x = direction < 0.5
      positive_y = (direction >= 0.5) & (direction < 0.75)

      self.vel_command_w[fwd_ids, 0] = torch.where(positive_x, speed, 0.0)
      self.vel_command_w[fwd_ids, 1] = torch.where(
        positive_y, speed, torch.where(positive_x, 0.0, -speed)
      )
      self.vel_command_w[fwd_ids, 2] = 0.0
      if self.cfg.heading_command:
        self.heading_target[fwd_ids] = torch.where(
          positive_x,
          0.0,
          torch.where(positive_y, np.pi / 2, -np.pi / 2),
        )

    self._update_command()

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      root_lin_vel_w = self.robot.data.root_link_lin_vel_w[init_vel_env_ids]
      root_lin_vel_w[:, :2] = self.vel_command_w[init_vel_env_ids, :2]
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    self.vel_command_b[:, 2] = self.vel_command_w[:, 2]
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      yaw_rate_cmd = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
      self.vel_command_w[env_ids, 2] = yaw_rate_cmd
      self.vel_command_b[env_ids, 2] = yaw_rate_cmd

    # Rotate fixed world-frame linear command into robot yaw frame for policy.
    heading = self.robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    vx_w = self.vel_command_w[:, 0]
    vy_w = self.vel_command_w[:, 1]
    self.vel_command_b[:, 0] = cos_h * vx_w + sin_h * vy_w
    self.vel_command_b[:, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    yaw_label = "heading" if self.cfg.heading_command else "ang_vel_z"
    yaw_max = (
      max(abs(ranges.heading[0]), abs(ranges.heading[1]))
      if self.cfg.heading_command and ranges.heading is not None
      else ranges.ang_vel_z[1]
    )
    axes = [
      ("lin_vel_x_w", ranges.lin_vel_x[1]),
      ("lin_vel_y_w", ranges.lin_vel_y[1]),
      (yaw_label, yaw_max),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        if i < 2:
          self.vel_command_w[idx, i] = s.value
        elif self.cfg.heading_command:
          self.heading_target[idx] = s.value
          self.is_heading_env[idx] = True
        else:
          self.vel_command_w[idx, i] = s.value
      self.is_standing_env[idx] = False
      self._update_command()

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds_b = self.command.cpu().numpy()
    cmds_w = self.command_w.cpu().numpy()
    heading_targets = self.heading_target.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    lin_vel_ws = self.robot.data.root_link_lin_vel_w.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      cmd_b = cmds_b[batch]
      cmd_w = cmds_w[batch]
      heading_target = heading_targets[batch]
      lin_vel_w = lin_vel_ws[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      origin = base_pos_w + np.array([0, 0, z_offset]) * scale

      # World-frame command linear velocity arrow (blue).
      cmd_lin_from = origin
      cmd_lin_to = origin + np.array([cmd_w[0], cmd_w[1], 0]) * scale
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Desired heading arrow (yellow).
      heading_from = origin + np.array([0, 0, 0.08])
      heading_to = heading_from + np.array(
        [np.cos(heading_target), np.sin(heading_target), 0]
      ) * scale
      visualizer.add_arrow(
        heading_from, heading_to, color=(0.9, 0.75, 0.15, 0.7), width=0.012
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = origin + np.array([0, 0, cmd_b[2]]) * scale
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual world-frame linear velocity arrow (cyan).
      act_lin_from = origin
      act_lin_to = origin + np.array([lin_vel_w[0], lin_vel_w[1], 0]) * scale
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = origin + np.array([0, 0, ang_vel_b[2]]) * scale
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive axis-aligned straight-line commands:
  +world-x with 50% probability, +world-y with 25%, and -world-y with 25%."""
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
