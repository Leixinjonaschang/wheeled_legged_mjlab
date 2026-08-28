from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from mjlab.managers.command_manager import CommandTerm
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    COMMAND_NAME,
    WORLD_COMMAND_TRACKING_ACTIVATION_STEPS,
    wf_tron1b_flat_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.commands import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.terminations import (
    world_command_tracking_failure,
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

    term._update_command(None)

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
    term._update_command(None)
    world_command_before_turn = term.command_w[:, :2].clone()

    robot_heading[:] = math.pi / 2
    term._update_command(None)

    assert torch.allclose(
        term.command_w[:, :2], world_command_before_turn, atol=1.0e-6
    )
    assert torch.allclose(
        term.command[:, :2], torch.tensor([[2.5, 0.0]]), atol=1.0e-6
    )


def test_partial_command_update_does_not_change_other_environments() -> None:
    term = _make_command_term(
        heading_target=torch.tensor([math.pi / 2, -math.pi / 2]),
        robot_heading=torch.zeros(2),
        linear_command_h=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
    )
    term.vel_command_b[1] = torch.tensor([7.0, 8.0, 9.0])
    term.vel_command_w[1] = torch.tensor([4.0, 5.0, 6.0])

    term._update_command(torch.tensor([0]))

    assert torch.equal(term.vel_command_b[1], torch.tensor([7.0, 8.0, 9.0]))
    assert torch.equal(term.vel_command_w[1], torch.tensor([4.0, 5.0, 6.0]))


def test_init_velocity_is_written_during_reset_without_rewriting_pose(monkeypatch) -> None:
    term = UniformVelocityCommand.__new__(UniformVelocityCommand)
    term._env = SimpleNamespace(device="cpu")
    term.cfg = SimpleNamespace(init_velocity_prob=1.0)
    term.vel_command_w = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    term.vel_command_b = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]])
    written = []
    term.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_w=torch.tensor(
                [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]]
            ),
            root_link_ang_vel_w=torch.tensor(
                [[0.3, 0.4, 0.0], [0.6, 0.7, 0.0]]
            ),
        ),
        write_root_link_velocity_to_sim=lambda velocity, env_ids: written.append(
            (velocity.clone(), env_ids.clone())
        ),
    )
    monkeypatch.setattr(CommandTerm, "reset", lambda self, env_ids: {})

    term.reset(torch.tensor([1]))

    assert len(written) == 1
    velocity, env_ids = written[0]
    assert torch.equal(env_ids, torch.tensor([1]))
    assert torch.allclose(velocity, torch.tensor([[3.0, 4.0, 0.2, 0.6, 0.7, 1.5]]))


def test_training_and_play_configs_use_current_forward_speed_limits() -> None:
    training_command_cfg = wf_tron1b_flat_env_cfg().commands[COMMAND_NAME]
    play_command_cfg = wf_tron1b_flat_env_cfg(play=True).commands[COMMAND_NAME]

    assert isinstance(training_command_cfg, UniformVelocityCommandCfg)
    assert isinstance(play_command_cfg, UniformVelocityCommandCfg)
    assert training_command_cfg.ranges.lin_vel_x == (-1.0, 2.0)
    assert play_command_cfg.ranges.lin_vel_x == (-1.0, 1.0)


def _make_tracking_termination_env(
    *,
    command_xy: torch.Tensor,
    actual_xy: torch.Tensor,
    heading_target: torch.Tensor | None = None,
    heading: torch.Tensor | None = None,
    grounded: torch.Tensor | None = None,
    common_step_counter: int = 10,
) -> tuple[SimpleNamespace, UniformVelocityCommand]:
    num_envs = command_xy.shape[0]
    if heading_target is None:
        heading_target = torch.zeros(num_envs)
    if heading is None:
        heading = torch.zeros(num_envs)
    if grounded is None:
        grounded = torch.ones(num_envs, dtype=torch.bool)

    robot_data = SimpleNamespace(
        root_link_lin_vel_w=torch.cat(
            (actual_xy, torch.zeros(num_envs, 1)), dim=1
        ),
        heading_w=heading,
    )
    command_term = UniformVelocityCommand.__new__(UniformVelocityCommand)
    command_term.vel_command_w = torch.cat(
        (command_xy, torch.zeros(num_envs, 1)), dim=1
    )
    command_term.command_counter = torch.zeros(num_envs, dtype=torch.long)
    command_term.heading_target = heading_target
    command_term.robot = SimpleNamespace(data=robot_data)
    command_term.is_heading_env = torch.ones(num_envs, dtype=torch.bool)
    command_term.is_standing_env = torch.zeros(num_envs, dtype=torch.bool)

    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.1,
        common_step_counter=common_step_counter,
        scene={
            "robot": SimpleNamespace(data=robot_data),
            "wheels_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=grounded[:, None])
            ),
        },
        command_manager=SimpleNamespace(get_term=lambda _: command_term),
        extras={},
    )
    return env, command_term


def _tracking_termination_params(**overrides) -> dict:
    params = {
        "command_name": COMMAND_NAME,
        "activation_step": 10,
        "command_grace_s": 0.0,
        "progress_deficit_threshold": 0.45,
        "min_progress_ratio": 0.55,
        "progress_duration_s": 0.4,
        "actual_speed_threshold": 0.2,
        "direction_angle_threshold_deg": 70.0,
        "direction_duration_s": 10.0,
        "heading_error_threshold_deg": 55.0,
        "heading_duration_s": 10.0,
        "heading_alignment_gate_deg": 45.0,
    }
    params.update(overrides)
    return params


def test_world_command_tracking_is_disabled_before_activation() -> None:
    env, _ = _make_tracking_termination_env(
        command_xy=torch.tensor([[1.0, 0.0]]),
        actual_xy=torch.zeros(1, 2),
        common_step_counter=9,
    )
    termination = world_command_tracking_failure(None, env)

    for _ in range(6):
        assert not termination(env, **_tracking_termination_params()).item()


def test_world_command_tracking_terminates_sustained_progress_deficit() -> None:
    env, _ = _make_tracking_termination_env(
        command_xy=torch.tensor([[1.0, 0.0]]),
        actual_xy=torch.zeros(1, 2),
    )
    termination = world_command_tracking_failure(None, env)

    for _ in range(3):
        assert not termination(env, **_tracking_termination_params()).item()
    assert termination(env, **_tracking_termination_params()).item()


def test_world_command_tracking_terminates_sustained_wrong_direction() -> None:
    env, _ = _make_tracking_termination_env(
        command_xy=torch.tensor([[1.0, 0.0]]),
        actual_xy=torch.tensor([[0.0, 1.0]]),
    )
    termination = world_command_tracking_failure(None, env)

    params = _tracking_termination_params(
        progress_duration_s=10.0,
        direction_duration_s=0.3,
    )
    for _ in range(2):
        assert not termination(env, **params).item()
    assert termination(env, **params).item()


def test_world_command_tracking_terminates_sustained_heading_error() -> None:
    env, _ = _make_tracking_termination_env(
        command_xy=torch.tensor([[0.0, 1.0]]),
        actual_xy=torch.tensor([[0.0, 1.0]]),
        heading_target=torch.tensor([math.pi / 2]),
    )
    termination = world_command_tracking_failure(None, env)

    params = _tracking_termination_params(heading_duration_s=0.6)
    for _ in range(5):
        assert not termination(env, **params).item()
    assert termination(env, **params).item()


def test_world_command_tracking_clears_failures_while_both_wheels_airborne() -> None:
    env, _ = _make_tracking_termination_env(
        command_xy=torch.tensor([[1.0, 0.0]]),
        actual_xy=torch.zeros(1, 2),
        grounded=torch.zeros(1, dtype=torch.bool),
    )
    termination = world_command_tracking_failure(None, env)

    for _ in range(6):
        assert not termination(env, **_tracking_termination_params()).item()
    assert termination._progress_bad_steps.item() == 0
    assert termination._direction_bad_steps.item() == 0
    assert termination._heading_bad_steps.item() == 0


def test_world_command_tracking_restarts_grace_after_command_change() -> None:
    env, command_term = _make_tracking_termination_env(
        command_xy=torch.tensor([[1.0, 0.0]]),
        actual_xy=torch.zeros(1, 2),
    )
    termination = world_command_tracking_failure(None, env)
    params = _tracking_termination_params(progress_duration_s=0.1, command_grace_s=0.2)

    assert not termination(env, **params).item()
    assert not termination(env, **params).item()
    assert termination(env, **params).item()

    command_term.command_counter += 1
    assert not termination(env, **params).item()
    assert termination._progress_bad_steps.item() == 0


def test_tracking_failure_is_not_timeout_and_is_disabled_for_play() -> None:
    training_cfg = wf_tron1b_flat_env_cfg()
    play_cfg = wf_tron1b_flat_env_cfg(play=True)
    term_cfg = training_cfg.terminations["world_command_tracking_failure"]

    assert term_cfg.time_out is False
    assert (
        term_cfg.params["activation_step"]
        == WORLD_COMMAND_TRACKING_ACTIVATION_STEPS
        == 5_000 * 24
    )
    assert term_cfg.params["contact_sensor_name"] == "wheels_ground_contact"
    assert term_cfg.params["heading_error_threshold_deg"] == 45.0
    assert "world_command_tracking_failure" not in play_cfg.terminations
