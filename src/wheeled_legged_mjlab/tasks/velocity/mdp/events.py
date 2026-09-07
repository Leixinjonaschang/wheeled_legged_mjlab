"""Event functions for the task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.dr import Operation
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

ADD_TO_CURRENT = Operation(
    name="add_to_current",
    initialize=torch.zeros_like,
    combine=torch.add,
    uses_defaults=False,
)
"""Additive DR operation that stacks on the current model value.

The built-in ``add`` operation reads the compile-time default instead of the
current value, so chaining it after an ``abs`` event on the same field and axis
silently discards the ``abs`` result. Use this when an additive per-entity
event must compose with an earlier event rather than replace it.
"""


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_pd_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Scale XML position gains and velocity-servo gains from nominal values.

    Position actuators receive independent Kp and Kd samples. Velocity actuators
    have no Kp, so their Kv is randomized with the damping scale range.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.int64)

    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")
    for actuator in asset.actuators:
        ctrl_ids = actuator.global_ctrl_ids
        sample_shape = (len(env_ids), len(ctrl_ids))

        if actuator.command_field == "position":
            stiffness_scale = torch.empty(sample_shape, device=env.device).uniform_(
                *stiffness_scale_range
            )
            env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
                default_gainprm[ctrl_ids, 0] * stiffness_scale
            )
            env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
                default_biasprm[ctrl_ids, 1] * stiffness_scale
            )
        elif actuator.command_field != "velocity":
            raise TypeError(
                "randomize_pd_gains supports only position and velocity actuators, "
                f"got {actuator.command_field!r}"
            )

        damping_scale = torch.empty(sample_shape, device=env.device).uniform_(
            *damping_scale_range
        )
        if actuator.command_field == "position":
            env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
                default_biasprm[ctrl_ids, 2] * damping_scale
            )
        else:
            env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
                default_gainprm[ctrl_ids, 0] * damping_scale
            )
            env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
                default_biasprm[ctrl_ids, 2] * damping_scale
            )


def _replace_non_finite_(tensor: torch.Tensor, env_ids: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        return
    tensor[env_ids] = torch.nan_to_num(tensor[env_ids], nan=0.0, posinf=0.0, neginf=0.0)


def clear_non_finite_sim_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Clear stale non-finite physics/sensor buffers after resetting envs."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)

    data = env.sim.data
    for name in (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "sensordata",
        "actuator_force",
        "qfrc_actuator",
    ):
        tensor = getattr(data, name, None)
        if tensor is not None:
            _replace_non_finite_(tensor, env_ids)

    for sensor in env.scene.sensors.values():
        if not isinstance(sensor, ContactSensor):
            continue
        for name in (
            "force",
            "torque",
            "dist",
            "pos",
            "normal",
            "tangent",
            "current_air_time",
            "last_air_time",
            "current_contact_time",
            "last_contact_time",
            "force_history",
            "torque_history",
            "dist_history",
        ):
            tensor = getattr(sensor.data, name, None)
            if tensor is not None:
                _replace_non_finite_(tensor, env_ids)


def prepare_quantities(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Compute the nominal foot position in the body frame.

    This function computes the nominal foot position in the body frame. This function is only suitable for TRON robot.

    The computed nominal foot position is stored in the following attributes of env:
        - env._nominal_foot_position_b: Nominal foot positions in body frame
        - env._wheels_link_ids: Body indices of wheel links
        - env._wheels_joint_ids: Joint indices of wheel joints
        - env._foot_radius: Radius of the foot/wheel (0.127m)
    """
    asset: Entity = env.scene[asset_cfg.name]

    wheel_link_idx, _ = asset.find_bodies("wheel_[RL]_Link")
    wheel_joint_ids, _ = asset.find_joints("wheel_[RL]_Joint")
    base_idx, _ = asset.find_bodies("base_Link")

    wheels_pos_w = asset.data.body_link_pos_w[:, wheel_link_idx, :]
    base_pos_w = asset.data.body_link_pos_w[:, base_idx, :]
    base_quat = asset.data.body_link_quat_w[:, base_idx, :]

    nominal_foot_position_b = torch.zeros(len(wheel_link_idx), 3, device=env.device)

    for j in range(env.num_envs):
        if torch.any(asset.data.joint_pos[j, :] > 5e-2):
            continue
        for i in range(len(wheel_link_idx)):
            nominal_foot_position_b[i, :] = quat_apply_inverse(
                base_quat[j, 0, :], wheels_pos_w[j, i, :] - base_pos_w[j, 0, :]
            )
        break

    assert (nominal_foot_position_b != 0.0).any(), "Failed to compute nominal foot positions"

    env._nominal_foot_position_b = nominal_foot_position_b  # type: ignore
    env._wheels_link_ids = wheel_link_idx  # type: ignore
    env._wheels_joint_ids = wheel_joint_ids  # type: ignore
    env._foot_radius = 0.127  # type: ignore
