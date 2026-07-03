from __future__ import annotations

from types import SimpleNamespace

import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    wf_tron1b_flat_env_cfg,
    wf_tron1b_rough_depth_env_cfg,
    wf_tron1b_rough_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.actions import (
    DelayedJointPositionActionCfg,
    DelayedJointVelocityActionCfg,
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

    term.reset(torch.tensor([0]))
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
