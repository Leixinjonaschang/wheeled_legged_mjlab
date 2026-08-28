from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from wheeled_legged_mjlab.rl.vecenv_wrapper import get_applied_actions
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    wf_tron1b_flat_env_cfg,
    wf_tron1b_rough_depth_env_cfg,
    wf_tron1b_rough_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.actions import (
    DelayedJointPositionActionCfg,
    DelayedJointVelocityActionCfg,
    _delay_step_bounds,
)


class DummyEntity:
    def __init__(self, *, num_envs: int, num_joints: int = 2) -> None:
        self.joint_names = tuple(f"joint_{idx}" for idx in range(num_joints))
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros(num_envs, num_joints),
            default_joint_vel=torch.zeros(num_envs, num_joints),
            encoder_bias=torch.zeros(num_envs, num_joints),
        )
        self.position_targets: list[torch.Tensor] = []
        self.velocity_targets: list[torch.Tensor] = []

    def find_joints_by_actuator_names(
        self, actuator_names: tuple[str, ...] | list[str]
    ) -> tuple[list[int], list[str]]:
        joint_ids = [self.joint_names.index(name) for name in actuator_names]
        return joint_ids, [self.joint_names[joint_id] for joint_id in joint_ids]

    def set_joint_position_target(
        self, target: torch.Tensor, *, joint_ids: torch.Tensor
    ) -> None:
        del joint_ids
        self.position_targets.append(target.clone())

    def set_joint_velocity_target(
        self, target: torch.Tensor, *, joint_ids: torch.Tensor
    ) -> None:
        del joint_ids
        self.velocity_targets.append(target.clone())


class DummyEnv:
    def __init__(
        self, *, num_envs: int = 1, physics_dt: float = 0.005, decimation: int = 4
    ) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.physics_dt = physics_dt
        self.step_dt = physics_dt * decimation
        self.common_step_counter = 0
        self.entity = DummyEntity(num_envs=num_envs)
        self.scene = {"robot": self.entity}


def test_training_configs_use_delayed_actions_and_play_configs_do_not() -> None:
    training_cfgs = (
        wf_tron1b_flat_env_cfg(),
        wf_tron1b_rough_env_cfg(),
        wf_tron1b_rough_depth_env_cfg(),
    )

    for cfg in training_cfgs:
        assert type(cfg.actions["leg_pos"]) is DelayedJointPositionActionCfg
        assert type(cfg.actions["wheel_vel"]) is DelayedJointVelocityActionCfg
        assert (
            cfg.actions["leg_pos"].delay_group == cfg.actions["wheel_vel"].delay_group
        )

    play_cfgs = (
        wf_tron1b_flat_env_cfg(play=True),
        wf_tron1b_rough_env_cfg(play=True),
        wf_tron1b_rough_depth_env_cfg(play=True),
    )

    for cfg in play_cfgs:
        assert type(cfg.actions["leg_pos"]) is JointPositionActionCfg
        assert type(cfg.actions["wheel_vel"]) is JointVelocityActionCfg


def test_zero_delay_applies_current_processed_action_immediately() -> None:
    env = DummyEnv()
    term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_0", "joint_1"),
        scale=2.0,
        use_default_offset=False,
        delay_range_s=(0.0, 0.0),
        delay_group="test",
    ).build(env)

    term.process_actions(torch.tensor([[1.0, -2.0]]))
    term.apply_actions()

    assert torch.allclose(env.entity.velocity_targets[-1], torch.tensor([[2.0, -4.0]]))


def test_fixed_delay_waits_for_elapsed_physics_substeps() -> None:
    env = DummyEnv(physics_dt=0.005)
    term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=1.0,
        use_default_offset=False,
        delay_range_s=(0.01, 0.01),
        delay_group="test",
    ).build(env)

    term.process_actions(torch.tensor([[3.0]]))

    term.apply_actions()
    assert torch.allclose(env.entity.velocity_targets[-1], torch.zeros(1, 1))
    term.apply_actions()
    assert torch.allclose(env.entity.velocity_targets[-1], torch.zeros(1, 1))
    term.apply_actions()
    assert torch.allclose(env.entity.velocity_targets[-1], torch.tensor([[3.0]]))


def test_applied_action_averages_the_actions_executed_across_substeps() -> None:
    env = DummyEnv(physics_dt=0.005, decimation=4)
    term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=10.0,
        use_default_offset=False,
        delay_range_s=(0.01, 0.01),
        delay_group="test",
    ).build(env)

    term.process_actions(torch.tensor([[3.0]]))
    for _ in range(4):
        term.apply_actions()

    # Two substeps execute the old zero action, then two execute the new action.
    assert torch.allclose(term.applied_action, torch.tensor([[1.5]]))


def test_applied_actions_follow_action_manager_term_order() -> None:
    first = SimpleNamespace(
        applied_action=torch.tensor([[1.0, 2.0]]),
        raw_action=torch.tensor([[9.0, 9.0]]),
    )
    second = SimpleNamespace(raw_action=torch.tensor([[3.0]]))
    terms = {"first": first, "second": second}
    action_manager = SimpleNamespace(
        active_terms=["first", "second"],
        get_term=terms.__getitem__,
    )

    assert torch.equal(
        get_applied_actions(action_manager),
        torch.tensor([[1.0, 2.0, 3.0]]),
    )


def test_done_env_uses_terminal_applied_action_snapshot() -> None:
    term = SimpleNamespace(
        applied_action=torch.tensor([[0.0], [2.0]]),
        terminal_applied_action=torch.tensor([[1.0], [9.0]]),
        raw_action=torch.zeros(2, 1),
    )
    action_manager = SimpleNamespace(
        active_terms=["delayed"],
        get_term=lambda _: term,
    )

    applied_actions = get_applied_actions(
        action_manager,
        dones=torch.tensor([1, 0]),
    )

    assert torch.equal(applied_actions, torch.tensor([[1.0], [2.0]]))


def test_reset_clears_delayed_targets_and_raw_action_stays_latest() -> None:
    env = DummyEnv(physics_dt=0.005)
    term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=1.0,
        use_default_offset=False,
        delay_range_s=(0.01, 0.01),
        delay_group="test",
    ).build(env)

    term.process_actions(torch.tensor([[4.0]]))
    for _ in range(3):
        term.apply_actions()
    assert torch.allclose(env.entity.velocity_targets[-1], torch.tensor([[4.0]]))
    completed_step_action = term.applied_action.clone()

    term.reset(torch.tensor([0]))
    assert torch.allclose(term.terminal_applied_action, completed_step_action)
    assert torch.allclose(term.applied_action, torch.zeros(1, 1))
    term.process_actions(torch.tensor([[7.0]]))

    assert torch.allclose(term.raw_action, torch.tensor([[7.0]]))
    term.apply_actions()
    assert torch.allclose(env.entity.velocity_targets[-1], torch.zeros(1, 1))
    term.apply_actions()
    assert torch.allclose(env.entity.velocity_targets[-1], torch.zeros(1, 1))
    term.apply_actions()
    assert torch.allclose(env.entity.velocity_targets[-1], torch.tensor([[7.0]]))


def test_delay_group_state_is_shared_across_action_terms() -> None:
    env = DummyEnv(num_envs=3)
    leg_term = DelayedJointPositionActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=1.0,
        use_default_offset=True,
        delay_range_s=(0.0, 0.02),
        delay_group="shared",
    ).build(env)
    wheel_term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_1",),
        scale=1.0,
        use_default_offset=False,
        delay_range_s=(0.0, 0.02),
        delay_group="shared",
    ).build(env)

    assert leg_term.delay_steps.data_ptr() == wheel_term.delay_steps.data_ptr()


def test_shared_delay_group_reset_resamples_once() -> None:
    env = DummyEnv(num_envs=3)
    leg_term = DelayedJointPositionActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=1.0,
        use_default_offset=True,
        delay_range_s=(0.0, 0.02),
        delay_group="shared",
    ).build(env)
    wheel_term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_1",),
        scale=1.0,
        use_default_offset=False,
        delay_range_s=(0.0, 0.02),
        delay_group="shared",
    ).build(env)
    state = leg_term._delay_state
    original_resample = state.resample
    resample_calls = 0

    def count_resample(env_ids, *, current_policy_step: int) -> None:
        nonlocal resample_calls
        resample_calls += 1
        original_resample(env_ids, current_policy_step=current_policy_step)

    state.resample = count_resample

    reset_env_ids = torch.tensor([0, 2])
    leg_term.reset(reset_env_ids)
    wheel_term.reset(reset_env_ids)

    assert resample_calls == 1


def test_delay_step_bounds_use_representable_physics_steps() -> None:
    assert _delay_step_bounds((0.0, 0.02), 0.005) == (0, 4)
    assert _delay_step_bounds((0.007, 0.018), 0.005) == (2, 3)
    assert _delay_step_bounds((0.01, 0.01), 0.005) == (2, 2)

    with pytest.raises(ValueError, match="representable physics-step"):
        _delay_step_bounds((0.006, 0.009), 0.005)


def test_delay_resampling_samples_only_representable_steps_in_range() -> None:
    env = DummyEnv(num_envs=128, physics_dt=0.005)
    term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=1.0,
        use_default_offset=False,
        delay_range_s=(0.007, 0.018),
        delay_group="test",
    ).build(env)

    term._delay_state.resample(slice(None), current_policy_step=0)

    assert torch.all((term.delay_steps == 2) | (term.delay_steps == 3))


def test_fixed_delay_resampling_keeps_single_step_value() -> None:
    env = DummyEnv(num_envs=16, physics_dt=0.005)
    term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=1.0,
        use_default_offset=False,
        delay_range_s=(0.01, 0.01),
        delay_group="test",
    ).build(env)

    term._delay_state.resample(slice(None), current_policy_step=0)

    assert torch.all(term.delay_steps == 2)


def test_partial_reset_does_not_change_unreset_env_delay_or_fifo() -> None:
    env = DummyEnv(num_envs=3, physics_dt=0.005)
    term = DelayedJointVelocityActionCfg(
        entity_name="robot",
        actuator_names=("joint_0",),
        scale=1.0,
        use_default_offset=False,
        delay_range_s=(0.0, 0.02),
        delay_group="test",
    ).build(env)
    term._delay_state.delay_steps[:] = torch.tensor([0, 1, 2])
    term.process_actions(torch.tensor([[1.0], [2.0], [3.0]]))
    for _ in range(2):
        term.apply_actions()

    unreset_fifo = term._processed_action_fifo[1].clone()
    unreset_delay = term.delay_steps[1].clone()

    term.reset(torch.tensor([0, 2]))

    assert torch.equal(term._processed_action_fifo[1], unreset_fifo)
    assert torch.equal(term.delay_steps[1], unreset_delay)
