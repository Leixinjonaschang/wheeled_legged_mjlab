from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import (
    BuiltinSensor,
    BuiltinSensorCfg,
    ContactData,
    ContactMatch,
    ContactSensor,
    ContactSensorCfg,
)

from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    COM_POS_SENSOR,
    COM_VEL_SENSOR,
    COMMAND_NAME,
    SUPPORT_CONTACT_SENSOR,
    wf_tron1b_flat_env_cfg,
    wf_tron1b_rough_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp import rewards as reward_terms
from wheeled_legged_mjlab.tasks.velocity.mdp.commands import UniformVelocityCommand


def _cached_builtin_sensor(data: torch.Tensor) -> BuiltinSensor:
    sensor = BuiltinSensor.from_existing("test")
    sensor._cached_data = data
    sensor._cache_valid = True
    return sensor


def _cached_contact_sensor(
    found: torch.Tensor,
    force: torch.Tensor,
    pos: torch.Tensor,
) -> ContactSensor:
    sensor = ContactSensor(
        ContactSensorCfg(
            name="test_contact",
            primary=ContactMatch(mode="geom", pattern=("left", "right")),
            fields=("found", "force", "pos"),
            reduce="netforce",
        )
    )
    sensor._cached_data = ContactData(found=found, force=force, pos=pos)
    sensor._cache_valid = True
    return sensor


def _call_reward(
    monkeypatch,
    *,
    com_pos: torch.Tensor,
    contact_found: torch.Tensor,
    contact_pos: torch.Tensor,
    contact_force: torch.Tensor,
    com_vel: torch.Tensor | None = None,
    command_xy: torch.Tensor | None = None,
    rough_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SimpleNamespace]:
    num_envs = com_pos.shape[0]
    if com_vel is None:
        com_vel = torch.zeros(num_envs, 3)
    if command_xy is None:
        command_xy = torch.zeros(num_envs, 2)
    if rough_gate is None:
        rough_gate = torch.ones(num_envs)

    zeros_per_wheel = torch.zeros(num_envs, 2)
    zeros_per_env = torch.zeros(num_envs)
    stats = reward_terms._TerrainRoughnessStats(
        jump=zeros_per_wheel,
        curvature=zeros_per_wheel,
        foot_roughness=zeros_per_wheel,
        robot_roughness=zeros_per_env,
        gate=rough_gate,
    )
    monkeypatch.setattr(
        reward_terms,
        "_terrain_roughness_from_sensor",
        lambda *args, **kwargs: stats,
    )

    command_term = UniformVelocityCommand.__new__(UniformVelocityCommand)
    command_term.vel_command_w = torch.cat(
        (command_xy, torch.zeros(num_envs, 1)), dim=1
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        common_step_counter=0,
        command_manager=SimpleNamespace(get_term=lambda _: command_term),
        scene={
            COM_POS_SENSOR: _cached_builtin_sensor(com_pos),
            COM_VEL_SENSOR: _cached_builtin_sensor(com_vel),
            SUPPORT_CONTACT_SENSOR: _cached_contact_sensor(
                contact_found, contact_force, contact_pos
            ),
        },
        extras={},
    )
    reward = reward_terms.rough_lipm_com_guidance(
        env,
        roughness_sensor_name="terrain_scan",
        com_pos_sensor_name=COM_POS_SENSOR,
        com_vel_sensor_name=COM_VEL_SENSOR,
        support_contact_sensor_name=SUPPORT_CONTACT_SENSOR,
        command_name=COMMAND_NAME,
        grid_shape=(1, 1),
    )
    return reward, env


def test_single_contact_uses_that_wheel_as_support(monkeypatch) -> None:
    reward, _ = _call_reward(
        monkeypatch,
        com_pos=torch.tensor([[1.0, 2.0, 0.8]]),
        contact_found=torch.tensor([[1, 0]]),
        contact_pos=torch.tensor([[[1.0, 2.0, 0.0], [50.0, 50.0, 0.0]]]),
        contact_force=torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 100.0]]]),
    )

    assert torch.allclose(reward, torch.ones(1))


def test_double_contact_support_is_net_force_weighted(monkeypatch) -> None:
    reward, _ = _call_reward(
        monkeypatch,
        com_pos=torch.tensor([[1.5, 0.0, 0.8]]),
        contact_found=torch.tensor([[1, 1]]),
        contact_pos=torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        contact_force=torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]]),
    )

    assert torch.allclose(reward, torch.ones(1))


def test_near_zero_net_forces_fall_back_to_equal_contact_weights(monkeypatch) -> None:
    reward, _ = _call_reward(
        monkeypatch,
        com_pos=torch.tensor([[1.0, 0.0, 0.8]]),
        contact_found=torch.tensor([[1, 1]]),
        contact_pos=torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        contact_force=torch.tensor([[[0.0, 0.0, 1.0e-12], [0.0, 0.0, 2.0e-12]]]),
    )

    assert torch.allclose(reward, torch.ones(1))


def test_no_contact_returns_finite_zero_reward(monkeypatch) -> None:
    reward, env = _call_reward(
        monkeypatch,
        com_pos=torch.tensor([[0.0, 0.0, 0.8]]),
        contact_found=torch.tensor([[0, 0]]),
        contact_pos=torch.full((1, 2, 3), torch.nan),
        contact_force=torch.full((1, 2, 3), torch.nan),
    )

    assert torch.equal(reward, torch.zeros(1))
    assert torch.isfinite(reward).all()
    assert all(
        bool(torch.isfinite(value))
        if isinstance(value, torch.Tensor)
        else math.isfinite(value)
        for value in env.extras["log"].values()
    )


def test_target_moves_along_positive_velocity_error(monkeypatch) -> None:
    target_offset = 0.08
    command_x = target_offset * 9.81 / (0.8 * 2.0)
    reward, _ = _call_reward(
        monkeypatch,
        com_pos=torch.tensor([[target_offset, 0.0, 0.8]]),
        com_vel=torch.zeros(1, 3),
        command_xy=torch.tensor([[command_x, 0.0]]),
        contact_found=torch.tensor([[1, 0]]),
        contact_pos=torch.zeros(1, 2, 3),
        contact_force=torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]]),
    )

    assert torch.allclose(reward, torch.ones(1), atol=1.0e-6)


def test_two_dimensional_target_offset_is_norm_clipped(monkeypatch) -> None:
    axis_offset = 0.15 / math.sqrt(2.0)
    reward, env = _call_reward(
        monkeypatch,
        com_pos=torch.tensor([[axis_offset, axis_offset, 0.8]]),
        command_xy=torch.tensor([[10.0, 10.0]]),
        contact_found=torch.tensor([[1, 0]]),
        contact_pos=torch.zeros(1, 2, 3),
        contact_force=torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]]),
    )

    assert torch.allclose(reward, torch.ones(1), atol=1.0e-6)
    assert torch.allclose(
        env.extras["log"]["Metrics/lipm_desired_offset_mean"],
        torch.tensor(0.15),
        atol=1.0e-6,
    )


def test_non_rough_gate_disables_reward(monkeypatch) -> None:
    reward, env = _call_reward(
        monkeypatch,
        com_pos=torch.zeros(1, 3),
        contact_found=torch.tensor([[1, 0]]),
        contact_pos=torch.zeros(1, 2, 3),
        contact_force=torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]]),
        rough_gate=torch.zeros(1),
    )

    assert torch.equal(reward, torch.zeros(1))
    assert env.extras["log"]["Metrics/lipm_valid_contact_ratio"] == 0.0


def test_zero_command_still_recovers_against_com_drift(monkeypatch) -> None:
    reward, _ = _call_reward(
        monkeypatch,
        com_pos=torch.tensor([[-0.15, 0.0, 0.8]]),
        com_vel=torch.tensor([[1.0, 0.0, 0.0]]),
        command_xy=torch.zeros(1, 2),
        contact_found=torch.tensor([[1, 0]]),
        contact_pos=torch.zeros(1, 2, 3),
        contact_force=torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]]),
    )

    assert torch.allclose(reward, torch.ones(1), atol=1.0e-6)


def test_rough_config_contains_lipm_sensors_and_ordered_reward() -> None:
    rough_cfg = wf_tron1b_rough_env_cfg()
    flat_cfg = wf_tron1b_flat_env_cfg()
    rough_sensors = {
        getattr(sensor, "prefixed_name", sensor.name): sensor
        for sensor in rough_cfg.scene.sensors
    }
    flat_sensor_names = {
        getattr(sensor, "prefixed_name", sensor.name)
        for sensor in flat_cfg.scene.sensors
    }

    assert isinstance(rough_sensors[COM_POS_SENSOR], BuiltinSensorCfg)
    assert rough_sensors[COM_POS_SENSOR].sensor_type == "subtreecom"
    assert isinstance(rough_sensors[COM_VEL_SENSOR], BuiltinSensorCfg)
    assert rough_sensors[COM_VEL_SENSOR].sensor_type == "subtreelinvel"
    support_sensor = rough_sensors[SUPPORT_CONTACT_SENSOR]
    assert isinstance(support_sensor, ContactSensorCfg)
    assert support_sensor.fields == ("found", "force", "pos")
    assert support_sensor.reduce == "netforce"
    assert COM_POS_SENSOR not in flat_sensor_names
    assert COM_VEL_SENSOR not in flat_sensor_names
    assert SUPPORT_CONTACT_SENSOR not in flat_sensor_names

    reward_names = list(rough_cfg.rewards)
    assert reward_names.index("rough_lipm_com_guidance") == (
        reward_names.index("rough_contact_pattern") + 1
    )
    assert reward_names.index("rough_lipm_com_guidance") < reward_names.index(
        "rough_min_wheel_distance"
    )
    lipm_cfg = rough_cfg.rewards["rough_lipm_com_guidance"]
    assert lipm_cfg.func is reward_terms.rough_lipm_com_guidance
    assert lipm_cfg.weight == 0.5
    assert "rough_lipm_com_guidance" not in flat_cfg.rewards


def test_metrics_use_only_rough_environments_with_valid_contact(monkeypatch) -> None:
    command_x_for_offset_01 = 0.1 * 9.81 / (0.8 * 2.0)
    reward, env = _call_reward(
        monkeypatch,
        com_pos=torch.tensor(
            [
                [0.10, 0.0, 0.8],
                [7.00, 0.0, 0.8],
                [9.00, 0.0, 0.8],
                [0.27, 0.0, 0.8],
            ]
        ),
        command_xy=torch.tensor(
            [
                [command_x_for_offset_01, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [10.0, 0.0],
            ]
        ),
        contact_found=torch.tensor([[1, 0], [0, 0], [1, 0], [0, 1]]),
        contact_pos=torch.zeros(4, 2, 3),
        contact_force=torch.tensor(
            [
                [[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        ),
        rough_gate=torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )
    log = env.extras["log"]

    assert torch.allclose(reward, torch.tensor([1.0, 0.0, 0.0, math.exp(-1.0)]))
    assert torch.allclose(log["Metrics/lipm_com_error_mean"], torch.tensor(0.06))
    assert torch.allclose(log["Metrics/lipm_desired_offset_mean"], torch.tensor(0.125))
    assert torch.allclose(
        log["Metrics/lipm_reward_mean"],
        torch.tensor((1.0 + math.exp(-1.0)) / 2.0),
    )
    assert torch.allclose(
        log["Metrics/lipm_valid_contact_ratio"], torch.tensor(2.0 / 3.0)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MJLab GPU integration test")
def test_rough_environment_lipm_sensor_shapes_and_reward_range() -> None:
    cfg = wf_tron1b_rough_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        env.reset()
        assert env.scene[COM_POS_SENSOR].data.shape == (2, 3)
        assert env.scene[COM_VEL_SENSOR].data.shape == (2, 3)
        contact_data = env.scene[SUPPORT_CONTACT_SENSOR].data
        assert contact_data.found is not None
        assert contact_data.force is not None
        assert contact_data.pos is not None
        assert contact_data.found.shape == (2, 2)
        assert contact_data.force.shape == (2, 2, 3)
        assert contact_data.pos.shape == (2, 2, 3)

        actions = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
        )
        lipm_cfg = cfg.rewards["rough_lipm_com_guidance"]
        for _ in range(3):
            env.step(actions)
            reward = lipm_cfg.func(env, **lipm_cfg.params)
            assert reward.shape == (2,)
            assert torch.isfinite(reward).all()
            assert ((reward >= 0.0) & (reward <= 1.0)).all()
    finally:
        env.close()
