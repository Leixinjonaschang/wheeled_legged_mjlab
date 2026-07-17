from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    COMMAND_NAME,
    wf_tron1b_flat_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.commands import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


def _make_command_term(
    *,
    heading_target: torch.Tensor,
    robot_heading: torch.Tensor,
    linear_command_h: torch.Tensor,
) -> UniformVelocityCommand:
    term = UniformVelocityCommand.__new__(UniformVelocityCommand)
    term.cfg = SimpleNamespace(
        heading_command=True,
        heading_control_stiffness=1.0,
        ranges=SimpleNamespace(ang_vel_z=(-math.pi / 2, math.pi / 2)),
    )
    term.robot = SimpleNamespace(data=SimpleNamespace(heading_w=robot_heading))
    term.vel_command_h = linear_command_h
    term.vel_command_w = torch.zeros(len(heading_target), 3)
    term.vel_command_b = torch.zeros(len(heading_target), 3)
    term.heading_target = heading_target
    term.heading_error = torch.zeros(len(heading_target))
    term.is_heading_env = torch.ones(len(heading_target), dtype=torch.bool)
    term.is_standing_env = torch.zeros(len(heading_target), dtype=torch.bool)
    return term


def test_forward_limit_is_defined_in_target_heading_frame() -> None:
    term = _make_command_term(
        heading_target=torch.tensor([math.pi / 2, -math.pi / 2]),
        robot_heading=torch.zeros(2),
        linear_command_h=torch.tensor([[2.5, 0.0], [2.5, 0.0]]),
    )

    term._update_command()

    expected_world_velocity = torch.tensor([[0.0, 2.5], [0.0, -2.5]])
    assert torch.allclose(
        term.command_w[:, :2], expected_world_velocity, atol=1.0e-6
    )


def test_world_command_stays_fixed_while_robot_turns_to_target_heading() -> None:
    robot_heading = torch.zeros(1)
    term = _make_command_term(
        heading_target=torch.tensor([math.pi / 2]),
        robot_heading=robot_heading,
        linear_command_h=torch.tensor([[2.5, 0.0]]),
    )
    term._update_command()
    world_command_before_turn = term.command_w[:, :2].clone()

    robot_heading[:] = math.pi / 2
    term._update_command()

    assert torch.allclose(
        term.command_w[:, :2], world_command_before_turn, atol=1.0e-6
    )
    assert torch.allclose(
        term.command[:, :2], torch.tensor([[2.5, 0.0]]), atol=1.0e-6
    )


def test_training_and_play_configs_use_2_5_forward_speed_limit() -> None:
    for cfg in (wf_tron1b_flat_env_cfg(), wf_tron1b_flat_env_cfg(play=True)):
        command_cfg = cfg.commands[COMMAND_NAME]
        assert isinstance(command_cfg, UniformVelocityCommandCfg)
        assert command_cfg.ranges.lin_vel_x == (-1.0, 2.5)
