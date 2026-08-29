"""WF-TRON1B velocity tracking task configuration.

This file intentionally defines the full task instead of inheriting from
``mjlab.tasks.velocity.velocity_env_cfg``. WF-TRON1B is a wheeled-legged robot:
the policy controls leg joint positions and wheel joint velocities, and several
MDP terms are wheel-specific rather than quadruped-foot-specific.
"""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    CameraSensorCfg,
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from wheeled_legged_mjlab.assets.WF_TRON1B.wf_tron1b import WF_TRON1B_ROBOT_CFG
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.mdp.actions import (
    DelayedJointPositionActionCfg,
    DelayedJointVelocityActionCfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .terrain_cfg import PLANE_ENTITY_CFG, TERRAINS_ENTITY_CFG

ROBOT_ENTITY = "robot"
COMMAND_NAME = "twist"

BASE_BODY = "base_Link"
LEG_JOINT_NAMES = (
    "abad_[LR]_Joint",
    "hip_[LR]_Joint",
    "knee_[LR]_Joint",
)
WHEEL_JOINT_NAMES = ("wheel_[LR]_Joint",)
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES

WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
WHEEL_GEOM_NAMES = ("wheel_L_collision", "wheel_R_collision")
NON_WHEEL_COLLISION_GEOMS = (
    "base_collision",
    "abad_L_collision",
    "hip_L_collision",
    "knee_L_collision",
    "abad_R_collision",
    "hip_R_collision",
    "knee_R_collision",
)

BASE_HEIGHT_TARGET = 0.82
POSE_TARGET_JOINT_POS = {
    "abad_L_Joint": 0.0,
    "hip_L_Joint": 0.2,
    "knee_L_Joint": 0.48,
    "abad_R_Joint": 0.0,
    "hip_R_Joint": -0.2,
    "knee_R_Joint": -0.48,
}
WHEEL_DISTANCE_RANGE = (0.25, 0.55)
WHEEL_RADIUS = 0.127
WHEEL_HEIGHT_SCAN_SIZE = (0.40, 0.40)
WHEEL_HEIGHT_SCAN_RESOLUTION = 0.10
WHEEL_HEIGHT_GRID_SHAPE = (5, 5)
TERRAIN_SCAN_GRID_SHAPE = (11, 11)
DEPTH_CAMERA_NAME = "depth_camera"
DEPTH_CAMERA_ENTITY_NAME = "d435"
DEPTH_CAMERA_MUJOCO_NAME = f"{ROBOT_ENTITY}/{DEPTH_CAMERA_ENTITY_NAME}"
DEPTH_CAMERA_WIDTH = 53
DEPTH_CAMERA_HEIGHT = 30
DEPTH_LEFT_CROP = 8
DEPTH_MODEL_WIDTH = DEPTH_CAMERA_WIDTH - DEPTH_LEFT_CROP
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 2.0
DEPTH_BUFFER_SIZE = 5
DEPTH_BUFFER_UPDATE_PERIOD = 5
DEPTH_CAPTURE_FREQUENCY_HZ = 30.0
DEPTH_SYSTEM_DELAY_RANGE_S = (0.0, 0.020)
DEPTH_RANDOMIZATION_ENABLED = True
DEPTH_CALIBRATION_SCALE_RANGE = (0.98, 1.02)
DEPTH_CALIBRATION_BIAS_RANGE_M = (-0.01, 0.01)
DEPTH_DISTANCE_NOISE_ENABLED = True
DEPTH_NOISE_BASE_M = 0.001
DEPTH_NOISE_QUADRATIC_COEFF = 0.005
DEPTH_DROPOUT_PROBABILITY = 0.0
DEPTH_DROPOUT_PATCH_COUNT_RANGE = (1, 3)
DEPTH_DROPOUT_AREA_FRACTION_RANGE = (0.01, 0.05)
DEPTH_DROPOUT_ASPECT_RATIO_RANGE = (0.125, 8.0)
DEPTH_CAMERA_POSITION_DELTA_RANGE_M = (-0.010, 0.010)
DEPTH_CAMERA_PITCH_DELTA_RANGE_RAD = (-math.radians(1.0), math.radians(1.0))
DEPTH_CAMERA_FOVY_DELTA_RANGE_DEG = (-1.0, 1.0)
ROUGHNESS_GATE_THRESHOLD_INITIAL = 0.0
ROUGHNESS_GATE_THRESHOLD_FINAL = 0.65
ROUGHNESS_GATE_THRESHOLD_RAMP_STEPS = 5_000 * 24
FELL_OVER_LIMIT_ANGLE_INITIAL = math.radians(65.0)
FELL_OVER_LIMIT_ANGLE_FINAL = math.radians(85.0)
FELL_OVER_LIMIT_ANGLE_RAMP_STEPS = 5_000 * 24
# The RSL-RL runner collects 24 policy steps per learning iteration.
WORLD_COMMAND_TRACKING_ACTIVATION_STEPS = 5_000 * 24


def make_scene(*, rough: bool, depth: bool = False) -> SceneCfg:
    """Scene = terrain + robot + sensors."""
    terrain = deepcopy(TERRAINS_ENTITY_CFG if rough else PLANE_ENTITY_CFG)
    if rough and terrain.terrain_generator is not None:
        terrain.terrain_generator.curriculum = True

    return SceneCfg(
        terrain=terrain,
        entities={ROBOT_ENTITY: WF_TRON1B_ROBOT_CFG},
        sensors=make_sensors(rough=rough, depth=depth),
        num_envs=4096,
        extent=1.0,
    )


def make_sensors(*, rough: bool, depth: bool = False) -> tuple:
    """Terrain scans, optional depth camera, wheel height scans, and contact sensors."""
    sensors = []

    if depth:
        sensors.append(
            CameraSensorCfg(
                name=DEPTH_CAMERA_NAME,
                camera_name=DEPTH_CAMERA_MUJOCO_NAME,
                data_types=("depth",),
                width=DEPTH_CAMERA_WIDTH,
                height=DEPTH_CAMERA_HEIGHT,
                use_textures=False,
                use_shadows=False,
                enabled_geom_groups=(0, 1),
            )
        )

    if rough:
        sensors.append(
            RayCastSensorCfg(
                name="terrain_scan",
                frame=ObjRef(type="body", name=BASE_BODY, entity=ROBOT_ENTITY),
                ray_alignment="yaw",
                pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.1),
                max_distance=10.0,
                exclude_parent_body=True,
                include_geom_groups=(0,),
                debug_vis=True,
            )
        )
        sensors.append(
            TerrainHeightSensorCfg(
                name="wheel_height_scan",
                frame=tuple(
                    ObjRef(type="body", name=body_name, entity=ROBOT_ENTITY)
                    for body_name in WHEEL_BODY_NAMES
                ),
                ray_alignment="world",
                pattern=GridPatternCfg(
                    size=WHEEL_HEIGHT_SCAN_SIZE,
                    resolution=WHEEL_HEIGHT_SCAN_RESOLUTION,
                ),
                reduction="none",
                max_distance=1.0,
                exclude_parent_body=True,
                include_geom_groups=(0,),
                debug_vis=True,
            )
        )

    sensors.extend(
        (
            ContactSensorCfg(
                name="wheels_ground_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=WHEEL_GEOM_NAMES,
                    entity=ROBOT_ENTITY,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                track_air_time=True,
            ),
            ContactSensorCfg(
                name="illegal_ground_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=NON_WHEEL_COLLISION_GEOMS,
                    entity=ROBOT_ENTITY,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
            ContactSensorCfg(
                name="self_collision",
                primary=ContactMatch(
                    mode="subtree",
                    pattern=BASE_BODY,
                    entity=ROBOT_ENTITY,
                ),
                secondary=ContactMatch(
                    mode="subtree",
                    pattern=BASE_BODY,
                    entity=ROBOT_ENTITY,
                ),
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
        )
    )
    return tuple(sensors)


def make_observations(
    *,
    rough: bool,
    depth: bool = False,
    lin_vel_representation: bool = False,
    async_depth: bool = False,
    enable_depth_distance_noise: bool = DEPTH_DISTANCE_NOISE_ENABLED,
    depth_noise_base_m: float = DEPTH_NOISE_BASE_M,
    depth_noise_quadratic_coeff: float = DEPTH_NOISE_QUADRATIC_COEFF,
) -> dict[str, ObservationGroupCfg]:
    """Student history uses noisy proprioception; teacher and critic stay clean."""
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/gyro"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=LEG_JOINT_NAMES,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=LEG_JOINT_NAMES,
                )
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        ),
        "wheel_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=WHEEL_JOINT_NAMES,
                )
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.5,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
        ),
    }
    proprio_terms = dict(actor_terms)
    command_term = proprio_terms.pop("command")

    critic_terms = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/gyro"},
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=LEG_JOINT_NAMES,
                )
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=LEG_JOINT_NAMES,
                )
            },
            scale=0.05,
        ),
        "wheel_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=WHEEL_JOINT_NAMES,
                )
            },
            scale=0.5,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
        ),
        "wheel_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "wheels_ground_contact"},
        ),
        "wheel_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "wheels_ground_contact"},
        ),
    }

    if rough:
        critic_terms["height_scan"] = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            scale=0.1,
        )
        critic_terms["wheel_height"] = ObservationTermCfg(
            func=mdp.foot_height,
            params={"sensor_name": "wheel_height_scan"},
        )
        critic_terms["roughness_indicator"] = ObservationTermCfg(
            func=mdp.terrain_roughness_indicator,
            params={
                "sensor_name": "terrain_scan",
                "wheel_radius": WHEEL_RADIUS,
                "gate_min": 0.00,
                "gate_max": 0.50,
                "grid_shape": TERRAIN_SCAN_GRID_SHAPE,
            },
        )

    dynamics_context_terms = {
        "domain_randomization_delta_quantity": ObservationTermCfg(
            func=mdp.domain_randomization_delta_quantity,
        ),
    }

    if lin_vel_representation:
        privileged_encoder_terms = deepcopy(critic_terms)
        privileged_encoder_terms.pop("base_lin_vel", None)
        privileged_encoder_terms.pop("command", None)
        observations = {
            "proprio_history": ObservationGroupCfg(
                terms=dict(proprio_terms),
                concatenate_terms=True,
                enable_corruption=True,
                history_length=5,
                flatten_history_dim=False,
            ),
            "actor_command": ObservationGroupCfg(
                terms={"command": deepcopy(command_term)},
                concatenate_terms=True,
                enable_corruption=False,
            ),
            "lin_vel_target": ObservationGroupCfg(
                terms={"base_lin_vel": deepcopy(critic_terms["base_lin_vel"])},
                concatenate_terms=True,
                enable_corruption=False,
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                concatenate_terms=True,
                enable_corruption=False,
            ),
            "privileged_encoder": ObservationGroupCfg(
                terms=privileged_encoder_terms,
                concatenate_terms=True,
                enable_corruption=False,
            ),
        }
        if rough:
            observations["wheel_roughness"] = ObservationGroupCfg(
                terms={
                    "wheel_roughness": ObservationTermCfg(
                        func=mdp.wheel_roughness_gate,
                        params={
                            "sensor_name": "wheel_height_scan",
                            "wheel_radius": WHEEL_RADIUS,
                            "gate_min": 0.00,
                            "gate_max": 0.50,
                            "grid_shape": WHEEL_HEIGHT_GRID_SHAPE,
                        },
                    )
                },
                concatenate_terms=True,
                enable_corruption=False,
            )
    else:
        observations = {
            "actor": ObservationGroupCfg(
                terms=dict(actor_terms),
                concatenate_terms=True,
                enable_corruption=False,
            ),
            "actor_history": ObservationGroupCfg(
                terms=dict(actor_terms),
                concatenate_terms=True,
                enable_corruption=True,
                history_length=5,
                flatten_history_dim=False,
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                concatenate_terms=True,
                enable_corruption=False,
            ),
        }

    observations["dynamics_context"] = ObservationGroupCfg(
        terms=dynamics_context_terms,
        concatenate_terms=True,
        enable_corruption=False,
    )

    depth_processing_params = {
        "left_crop": DEPTH_LEFT_CROP,
        "depth_min_m": DEPTH_MIN_M,
        "depth_max_m": DEPTH_MAX_M,
        "enable_depth_randomization": DEPTH_RANDOMIZATION_ENABLED,
        "calibration_scale_range": DEPTH_CALIBRATION_SCALE_RANGE,
        "calibration_bias_range_m": DEPTH_CALIBRATION_BIAS_RANGE_M,
        "enable_depth_distance_noise": enable_depth_distance_noise,
        "noise_base_m": depth_noise_base_m,
        "noise_quadratic_coeff": depth_noise_quadratic_coeff,
        "dropout_probability": DEPTH_DROPOUT_PROBABILITY,
        "dropout_patch_count_range": DEPTH_DROPOUT_PATCH_COUNT_RANGE,
        "dropout_area_fraction_range": DEPTH_DROPOUT_AREA_FRACTION_RANGE,
        "dropout_aspect_ratio_range": DEPTH_DROPOUT_ASPECT_RATIO_RANGE,
    }
    if depth and async_depth:
        observations[DEPTH_CAMERA_NAME] = ObservationGroupCfg(
            terms={
                DEPTH_CAMERA_NAME: ObservationTermCfg(
                    func=mdp.async_depth_buffer,
                    params={
                        "sensor_name": DEPTH_CAMERA_NAME,
                        "capture_frequency_hz": DEPTH_CAPTURE_FREQUENCY_HZ,
                        "system_delay_range_s": DEPTH_SYSTEM_DELAY_RANGE_S,
                        **depth_processing_params,
                    },
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
    elif depth:
        observations[DEPTH_CAMERA_NAME] = ObservationGroupCfg(
            terms={
                DEPTH_CAMERA_NAME: ObservationTermCfg(
                    func=mdp.depth_buffer,
                    params={
                        "sensor_name": DEPTH_CAMERA_NAME,
                        "buffer_size": DEPTH_BUFFER_SIZE,
                        "update_period": DEPTH_BUFFER_UPDATE_PERIOD,
                        **depth_processing_params,
                    },
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
    return observations


def make_actions(
    *,
    action_delay: bool = True,
    delay_range_s: tuple[float, float] = (0.0, 0.02),
    resampling_time_s: float = 5.0,
) -> dict[str, ActionTermCfg]:
    """Mixed control: leg positions and wheel velocities."""
    leg_action_cfg = (
        DelayedJointPositionActionCfg if action_delay else JointPositionActionCfg
    )
    wheel_action_cfg = (
        DelayedJointVelocityActionCfg if action_delay else JointVelocityActionCfg
    )
    delay_kwargs = (
        {
            "delay_range_s": delay_range_s,
            "resampling_time_s": resampling_time_s,
            "delay_group": "base_actions",
        }
        if action_delay
        else {}
    )
    return {
        "leg_pos": leg_action_cfg(
            entity_name=ROBOT_ENTITY,
            actuator_names=LEG_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
            **delay_kwargs,
        ),
        "wheel_vel": wheel_action_cfg(
            entity_name=ROBOT_ENTITY,
            actuator_names=WHEEL_JOINT_NAMES,
            scale=10.0,
            use_default_offset=False,
            **delay_kwargs,
        ),
    }


def make_commands() -> dict[str, CommandTermCfg]:
    """Sample heading-frame linear velocities and track them in world frame."""
    return {
        COMMAND_NAME: UniformVelocityCommandCfg(
            entity_name=ROBOT_ENTITY,
            resampling_time_range=(6.0, 10.0),
            rel_standing_envs=0.1,
            rel_heading_envs=1.0,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-math.pi / 2, math.pi / 2),
                heading=(-math.pi, math.pi),
            ),
        )
    }


def make_events(*, depth: bool = False) -> dict[str, EventTermCfg]:
    """Reset logic and domain randomization used by the velocity task."""
    events = {
        "prepare_quantities": EventTermCfg(
            func=mdp.prepare_quantities,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg(ROBOT_ENTITY)},
        ),
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
                    "yaw": (-math.pi, math.pi),
                },
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
            },
        ),
        "reset_leg_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.3, 0.5),
                "velocity_range": (-0.2, 0.2),
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=LEG_JOINT_NAMES,
                ),
            },
        ),
        "reset_wheel_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (-0.1, 0.1),
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=WHEEL_JOINT_NAMES,
                ),
            },
        ),
        "clear_non_finite_sim_data": EventTermCfg(
            func=mdp.clear_non_finite_sim_data,
            mode="reset",
            params={},
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(15.0, 15.5),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.35, 0.35),
                    "pitch": (-0.35, 0.35),
                    "yaw": (-0.5, 0.5),
                },
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
            },
        ),
        "wheel_friction": EventTermCfg(
            func=mdp.dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    geom_names=WHEEL_GEOM_NAMES,
                ),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": False,
            },
        ),
        "encoder_bias": EventTermCfg(
            func=mdp.dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            func=mdp.dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY, body_names=(BASE_BODY,)),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
    }
    if depth:
        events.update(
            {
                "cam_pos": EventTermCfg(
                    func=mdp.dr.cam_pos,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            ROBOT_ENTITY,
                            camera_names=(DEPTH_CAMERA_ENTITY_NAME,),
                        ),
                        "distribution": "uniform",
                        "operation": "add",
                        "ranges": {
                            0: DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
                            1: DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
                            2: DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
                        },
                        "shared_random": False,
                    },
                ),
                "cam_pitch": EventTermCfg(
                    func=mdp.dr.cam_quat,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            ROBOT_ENTITY,
                            camera_names=(DEPTH_CAMERA_ENTITY_NAME,),
                        ),
                        "distribution": "uniform",
                        "roll_range": (0.0, 0.0),
                        "pitch_range": DEPTH_CAMERA_PITCH_DELTA_RANGE_RAD,
                        "yaw_range": (0.0, 0.0),
                    },
                ),
                "cam_fovy": EventTermCfg(
                    func=mdp.dr.cam_fovy,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            ROBOT_ENTITY,
                            camera_names=(DEPTH_CAMERA_ENTITY_NAME,),
                        ),
                        "distribution": "uniform",
                        "operation": "add",
                        "ranges": DEPTH_CAMERA_FOVY_DELTA_RANGE_DEG,
                        "shared_random": False,
                    },
                ),
            }
        )
    return events


def make_rewards(*, rough: bool) -> dict[str, RewardTermCfg]:
    """Velocity tracking rewards plus wheel-legged posture and safety terms."""
    wheel_body_cfg = SceneEntityCfg(ROBOT_ENTITY, body_names=WHEEL_BODY_NAMES)
    wheel_joint_cfg = SceneEntityCfg(ROBOT_ENTITY, joint_names=WHEEL_JOINT_NAMES)
    leg_joint_cfg = SceneEntityCfg(ROBOT_ENTITY, joint_names=LEG_JOINT_NAMES)
    all_joint_cfg = SceneEntityCfg(ROBOT_ENTITY, joint_names=ALL_JOINT_NAMES)

    rewards = {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=0.1),
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=4.0,
            params={"command_name": COMMAND_NAME, "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=1.0,
            params={"command_name": COMMAND_NAME, "std": math.sqrt(0.25)},
        ),
        "base_ang_vel_xy": RewardTermCfg(
            func=mdp.base_ang_vel_xy_l2,
            weight=-0.15,
            params={
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
                "roll_weight": 2.0,
                "pitch_weight": 1.0,
            },
        ),
        "track_heading": RewardTermCfg(
            func=mdp.track_heading,
            weight=0.5,
            params={
                "command_name": COMMAND_NAME,
                "std": math.sqrt(0.20),
                "command_norm_threshold": 0.05,
            },
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY, body_names=(BASE_BODY,)),
                "terrain_sensor_names": ("terrain_scan",) if rough else None,
            },
        ),
        "flat_orientation": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-3.0,
            params={"asset_cfg": SceneEntityCfg(ROBOT_ENTITY)},
        ),
        "base_height": RewardTermCfg(
            func=mdp.base_height_l2,
            weight=-50.0,
            params={
                "target_height": BASE_HEIGHT_TARGET,
                "deadband": 0.04,
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
                "sensor_name": "terrain_scan" if rough else None,
                "terrain_sample": "quantile" if rough else "center",
                "terrain_quantile": 0.75,
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=0.5,
            params={
                "asset_cfg": leg_joint_cfg,
                "command_name": COMMAND_NAME,
                "target_joint_pos": POSE_TARGET_JOINT_POS,
                "std_standing": {
                    r".*abad.*": 0.05,
                    r".*hip.*": 0.08,
                    r".*knee.*": 0.10,
                },
                "std_walking": {
                    r".*abad.*": 0.15,
                    r".*hip.*": 0.25,
                    r".*knee.*": 0.35,
                },
                "std_running": {
                    r".*abad.*": 0.20,
                    r".*hip.*": 0.35,
                    r".*knee.*": 0.50,
                },
                "walking_threshold": 0.05,
                "running_threshold": 0.5,
            },
        ),
        "stand_still": RewardTermCfg(
            func=mdp.stand_still,
            weight=-2.0,
            params={
                "command_name": COMMAND_NAME,
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
            },
        ),
        "wheel_distance": RewardTermCfg(
            func=mdp.wheel_distance,
            weight=-5.0,
            params={
                "asset_cfg": wheel_body_cfg,
                "min_distance": WHEEL_DISTANCE_RANGE[0],
                "max_distance": WHEEL_DISTANCE_RANGE[1],
            },
        ),
        "leg_joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-5.0,
            params={"asset_cfg": leg_joint_cfg},
        ),
        "leg_joint_vel": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.015,
            params={"asset_cfg": leg_joint_cfg},
        ),
        "leg_joint_torque": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1.0e-5,
            params={"asset_cfg": leg_joint_cfg},
        ),
        "wheel_joint_vel": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.0005,
            params={"asset_cfg": wheel_joint_cfg},
        ),
        "joint_acc": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-1.0e-7,
            params={"asset_cfg": all_joint_cfg},
        ),
        "joint_power": RewardTermCfg(
            func=mdp.joint_power_l1,
            weight=-5.0e-5,
            params={"asset_cfg": all_joint_cfg},
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.1,
            params={"sensor_name": "self_collision"},
        ),
        "illegal_ground_contact": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "illegal_ground_contact"},
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-3.0e-5,
            params={
                "sensor_name": "wheels_ground_contact",
                "command_name": COMMAND_NAME,
                "command_threshold": 0.05,
            },
        ),
        "wheel_air_time_balance": RewardTermCfg(
            func=mdp.wheel_air_time_balance,
            weight=-4.0,
            params={
                "sensor_name": "wheels_ground_contact",
                "min_total_air_time": 1.0,
                "balance_tolerance": 0.2,
            },
        ),
    }

    if rough:
        roughness_params = {
            "roughness_sensor_name": "terrain_scan",
            "wheel_radius": WHEEL_RADIUS,
            "gate_min": 0.00,
            "gate_max": 0.50,
            "roughness_gate_threshold": ROUGHNESS_GATE_THRESHOLD_INITIAL,
            "roughness_gate_threshold_final": ROUGHNESS_GATE_THRESHOLD_FINAL,
            "roughness_gate_threshold_ramp_steps": (
                ROUGHNESS_GATE_THRESHOLD_RAMP_STEPS
            ),
            "grid_shape": TERRAIN_SCAN_GRID_SHAPE,
        }
        rewards.update(
            {   # legged motion
                "rough_wheel_usage": RewardTermCfg(
                    func=mdp.rough_wheel_usage,
                    weight=-1.5e-2,
                    params={
                        **roughness_params,
                        "asset_cfg": wheel_joint_cfg,
                    },
                ),
                "rough_wheel_foot_clearance": RewardTermCfg(
                    func=mdp.rough_wheel_foot_clearance,
                    weight=2.0,
                    params={
                        **roughness_params,
                        "clearance_sensor_name": "wheel_height_scan",
                        "clearance_grid_shape": WHEEL_HEIGHT_GRID_SHAPE,
                        "contact_sensor_name": "wheels_ground_contact",
                        "command_name": COMMAND_NAME,
                        "base_target_height": 0.03,
                        "range_scale": 0.5,
                        "max_target_height": 0.18,
                        "target_std": 0.04,
                        "command_threshold": 0.05,
                    },
                ),
                "rough_contact_pattern": RewardTermCfg(
                    func=mdp.rough_contact_pattern,
                    weight=0.3,
                    params={
                        **roughness_params,
                        "contact_sensor_name": "wheels_ground_contact",
                        "command_name": COMMAND_NAME,
                        "command_threshold": 0.05,
                    },
                ),
                # wheeled motion
                "non_rough_wheel_lateral_symmetry": RewardTermCfg(
                    func=mdp.non_rough_wheel_lateral_symmetry,
                    weight=0.5,
                    params={
                        **roughness_params,
                        "asset_cfg": wheel_body_cfg,
                        "std": math.sqrt(0.5),
                    },
                ),
                "non_rough_wheel_x_alignment": RewardTermCfg(
                    func=mdp.non_rough_wheel_x_alignment,
                    weight=-50.0,
                    params={
                        **roughness_params,
                        "asset_cfg": wheel_body_cfg,
                    },
                ),
                "non_rough_flat_orientation": RewardTermCfg(
                    func=mdp.non_rough_flat_orientation,
                    weight=-10.0,
                    params={
                        **roughness_params,
                        "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
                        "roll_weight": 2.0,
                        "pitch_weight": 1.0,
                    },
                ),
                "standing_forward_wheel_air_time": RewardTermCfg(
                    func=mdp.standing_forward_wheel_air_time,
                    weight=-10.0,
                    params={
                        **roughness_params,
                        "contact_sensor_name": "wheels_ground_contact",
                        "command_name": COMMAND_NAME,
                        "max_time": 0.5,
                        "air_time_offset": 0.05,
                    },
                ),
            }
        )

    return rewards


def make_terminations(*, rough: bool) -> dict[str, TerminationTermCfg]:
    """Episode reset conditions."""
    terminations = {
        "non_finite_physics": TerminationTermCfg(func=mdp.non_finite_physics),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": FELL_OVER_LIMIT_ANGLE_INITIAL},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "illegal_ground_contact"},
        ),
        "world_command_tracking_failure": TerminationTermCfg(
            func=mdp.world_command_tracking_failure,
            time_out=False,
            params={
                "command_name": COMMAND_NAME,
                "activation_step": WORLD_COMMAND_TRACKING_ACTIVATION_STEPS,
                "command_norm_threshold": 0.35,
                "progress_deficit_threshold": 0.45,
                "min_progress_ratio": 0.55,
                "progress_duration_s": 0.4,
                "actual_speed_threshold": 0.2,
                "direction_angle_threshold_deg": 70.0,
                "direction_duration_s": 0.3,
                "heading_error_threshold_deg": 45.0,
                "heading_duration_s": 0.6,
                "heading_alignment_gate_deg": 45.0,
                "command_grace_s": 2.5,
                "contact_sensor_name": "wheels_ground_contact",
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
            },
        ),
    }
    if rough:
        terminations["out_of_terrain_bounds"] = TerminationTermCfg(
            func=mdp.out_of_terrain_bounds,
            params={"margin": 0.05},
            time_out=True,
        )
    return terminations


def make_curriculum(*, rough: bool) -> dict[str, CurriculumTermCfg]:
    """Training curricula for recovery tolerance and rough terrain."""
    curriculum = {
        "fell_over_limit_angle": CurriculumTermCfg(
            func=mdp.fell_over_limit_angle,
            params={
                "termination_term_name": "fell_over",
                "initial_limit_angle": FELL_OVER_LIMIT_ANGLE_INITIAL,
                "final_limit_angle": FELL_OVER_LIMIT_ANGLE_FINAL,
                "ramp_steps": FELL_OVER_LIMIT_ANGLE_RAMP_STEPS,
            },
        )
    }
    if rough:
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": COMMAND_NAME},
        )
    return curriculum


def make_metrics() -> dict[str, MetricsTermCfg]:
    return {
        "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    }


def make_sim(*, rough: bool) -> SimulationCfg:
    return SimulationCfg(
        nconmax=256 if rough else None, 
        njmax=512 if rough else 300,
        contact_sensor_maxmatch=256 if rough else 64,
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
            disableflags=("multiccd", "nativeccd"),
        ),
    )


def make_viewer() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name=ROBOT_ENTITY,
        body_name=BASE_BODY,
        distance=3.0,
        elevation=10.0,
        azimuth=90.0,
    )


def make_env_cfg(
    *,
    rough: bool,
    play: bool = False,
    depth: bool = False,
    lin_vel_representation: bool = False,
    async_depth: bool = False,
    enable_depth_distance_noise: bool = DEPTH_DISTANCE_NOISE_ENABLED,
    depth_noise_base_m: float = DEPTH_NOISE_BASE_M,
    depth_noise_quadratic_coeff: float = DEPTH_NOISE_QUADRATIC_COEFF,
) -> ManagerBasedRlEnvCfg:
    cfg = ManagerBasedRlEnvCfg(
        scene=make_scene(rough=rough, depth=depth),
        observations=make_observations(
            rough=rough,
            depth=depth,
            lin_vel_representation=lin_vel_representation,
            async_depth=async_depth,
            enable_depth_distance_noise=enable_depth_distance_noise,
            depth_noise_base_m=depth_noise_base_m,
            depth_noise_quadratic_coeff=depth_noise_quadratic_coeff,
        ),
        actions=make_actions(action_delay=not play),
        commands=make_commands(),
        events=make_events(depth=depth),
        rewards=make_rewards(rough=rough),
        terminations=make_terminations(rough=rough),
        curriculum=make_curriculum(rough=rough),
        metrics=make_metrics(),
        viewer=make_viewer(),
        sim=make_sim(rough=rough),
        decimation=4,
        episode_length_s=20.0,
        seed=0,
    )
    if play:
        apply_play_overrides(cfg, rough=rough)
    return cfg


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg, *, rough: bool) -> None:
    """Make rollout/play deterministic enough to inspect behavior."""
    cfg.episode_length_s = int(1e9)
    if "actor" in cfg.observations:
        cfg.observations["actor"].enable_corruption = False
    if "actor_history" in cfg.observations:
        cfg.observations["actor_history"].enable_corruption = False
    if "proprio_history" in cfg.observations:
        cfg.observations["proprio_history"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    for event_name in ("cam_pos", "cam_pitch", "cam_fovy"):
        cfg.events.pop(event_name, None)
    if DEPTH_CAMERA_NAME in cfg.observations:
        depth_term = cfg.observations[DEPTH_CAMERA_NAME].terms[DEPTH_CAMERA_NAME]
        depth_term.params["enable_depth_randomization"] = False
        depth_term.params["enable_depth_distance_noise"] = False
        if "system_delay_range_s" in depth_term.params:
            depth_term.params["system_delay_range_s"] = (0.0, 0.0)
    cfg.curriculum = {}
    cfg.terminations["fell_over"].params["limit_angle"] = (
        FELL_OVER_LIMIT_ANGLE_FINAL
    )
    cfg.terminations.pop("world_command_tracking_failure", None)

    twist_cmd = cfg.commands[COMMAND_NAME]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-math.pi / 2, math.pi / 2)

    if rough:
        for reward_term in cfg.rewards.values():
            if "roughness_gate_threshold_ramp_steps" in reward_term.params:
                reward_term.params["roughness_gate_threshold"] = (
                    ROUGHNESS_GATE_THRESHOLD_FINAL
                )
                reward_term.params["roughness_gate_threshold_ramp_steps"] = 0

        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        terrain = cfg.scene.terrain
        if terrain is not None and terrain.terrain_generator is not None:
            terrain.terrain_generator.curriculum = True
            terrain.terrain_generator.num_cols = len(terrain.terrain_generator.sub_terrains)
            terrain.terrain_generator.num_rows = 5
            terrain.terrain_generator.border_width = 5.0


def wf_tron1b_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create WF-TRON1B rough-terrain velocity tracking configuration."""
    return make_env_cfg(rough=True, play=play)


def wf_tron1b_rough_depth_env_cfg(
    play: bool = False,
    enable_depth_distance_noise: bool = DEPTH_DISTANCE_NOISE_ENABLED,
    depth_noise_base_m: float = DEPTH_NOISE_BASE_M,
    depth_noise_quadratic_coeff: float = DEPTH_NOISE_QUADRATIC_COEFF,
) -> ManagerBasedRlEnvCfg:
    """Create WF-TRON1B rough-terrain configuration with a depth camera."""
    return make_env_cfg(
        rough=True,
        play=play,
        depth=True,
        enable_depth_distance_noise=enable_depth_distance_noise,
        depth_noise_base_m=depth_noise_base_m,
        depth_noise_quadratic_coeff=depth_noise_quadratic_coeff,
    )


def wf_tron1b_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create WF-TRON1B flat-ground velocity tracking configuration."""
    return make_env_cfg(rough=False, play=play)


def wf_tron1b_rough_rep_ts_lin_vel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create WF-TRON1B rough-terrain velocity representation configuration."""
    return make_env_cfg(rough=True, play=play, lin_vel_representation=True)


def wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg(
    play: bool = False,
    enable_depth_distance_noise: bool = DEPTH_DISTANCE_NOISE_ENABLED,
    depth_noise_base_m: float = DEPTH_NOISE_BASE_M,
    depth_noise_quadratic_coeff: float = DEPTH_NOISE_QUADRATIC_COEFF,
) -> ManagerBasedRlEnvCfg:
    """Create rough-terrain velocity representation configuration with async depth."""
    return make_env_cfg(
        rough=True,
        play=play,
        depth=True,
        lin_vel_representation=True,
        async_depth=True,
        enable_depth_distance_noise=enable_depth_distance_noise,
        depth_noise_base_m=depth_noise_base_m,
        depth_noise_quadratic_coeff=depth_noise_quadratic_coeff,
    )


def wf_tron1b_flat_rep_ts_lin_vel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create WF-TRON1B flat-ground velocity representation configuration."""
    return make_env_cfg(rough=False, play=play, lin_vel_representation=True)
