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
WHEEL_DISTANCE_RANGE = (0.25, 0.55)
WHEEL_RADIUS = 0.127
WHEEL_HEIGHT_SCAN_SIZE = (0.40, 0.40)
WHEEL_HEIGHT_SCAN_RESOLUTION = 0.10
WHEEL_HEIGHT_GRID_SHAPE = (5, 5)
TERRAIN_SCAN_GRID_SHAPE = (11, 11)


def make_scene(*, rough: bool) -> SceneCfg:
    """Scene = terrain + robot + sensors."""
    terrain = deepcopy(TERRAINS_ENTITY_CFG if rough else PLANE_ENTITY_CFG)
    if rough and terrain.terrain_generator is not None:
        terrain.terrain_generator.curriculum = True

    return SceneCfg(
        terrain=terrain,
        entities={ROBOT_ENTITY: WF_TRON1B_ROBOT_CFG},
        sensors=make_sensors(rough=rough),
        num_envs=2048,
        extent=1.0,
    )


def make_sensors(*, rough: bool) -> tuple:
    """Terrain scans, wheel height scans, and contact sensors."""
    sensors = []

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


def make_observations(*, rough: bool) -> dict[str, ObservationGroupCfg]:
    """Teacher observations: actor and critic both use privileged terms."""
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
        "leg_joint_pos": ObservationTermCfg(
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
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
        ),
        "domain_randomization_delta_quantity": ObservationTermCfg(
            func=mdp.domain_randomization_delta_quantity,
        ),
    }

    critic_terms = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        **actor_terms,
        "wheel_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    ROBOT_ENTITY,
                    joint_names=WHEEL_JOINT_NAMES,
                )
            },
            scale=0.05,
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
        actor_terms["height_scan"] = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=0.1,
        )
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
                "gate_min": 0.10,
                "gate_max": 0.40,
                "grid_shape": TERRAIN_SCAN_GRID_SHAPE,
            },
        )

    return {
        "actor": ObservationGroupCfg(
            terms=dict(critic_terms),
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }


def make_actions() -> dict[str, ActionTermCfg]:
    """Mixed control: leg positions and wheel velocities."""
    return {
        "leg_pos": JointPositionActionCfg(
            entity_name=ROBOT_ENTITY,
            actuator_names=LEG_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        ),
        "wheel_vel": JointVelocityActionCfg(
            entity_name=ROBOT_ENTITY,
            actuator_names=WHEEL_JOINT_NAMES,
            scale=10.0,
            use_default_offset=False,
        ),
    }


def make_commands() -> dict[str, CommandTermCfg]:
    """Sample world-frame velocity commands with standing and heading modes."""
    return {
        COMMAND_NAME: UniformVelocityCommandCfg(
            entity_name=ROBOT_ENTITY,
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=1.0,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-math.pi / 2, math.pi / 2),
                heading=(-math.pi, math.pi),
            ),
        )
    }


def make_events() -> dict[str, EventTermCfg]:
    """Reset logic and domain randomization used by the velocity task."""
    return {
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
                    "x": (-0.2, 0.2),
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
                "position_range": (-0.2, 0.2),
                "velocity_range": (-0.1, 0.1),
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
                "velocity_range": (-0.01, 0.01),
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
            interval_range_s=(1.0, 3.0),
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
                "shared_random": True,
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
            weight=1.5,
            params={"command_name": COMMAND_NAME, "std": math.sqrt(0.25)},
        ),
        "base_ang_vel_xy": RewardTermCfg(
            func=mdp.base_ang_vel_xy_l2,
            weight=-0.25,
            params={"asset_cfg": SceneEntityCfg(ROBOT_ENTITY)},
        ),
        "track_heading": RewardTermCfg(
            func=mdp.track_heading,
            weight=0.5,
            params={
                "command_name": COMMAND_NAME,
                "std": math.sqrt(0.15),
                "command_norm_threshold": 0.05,
            },
        ),
        "heading_progress": RewardTermCfg(
            func=mdp.heading_progress,
            weight=0.0,
            params={"command_name": COMMAND_NAME, "max_progress": 0.05},
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
                "terrain_sample": "center",
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=0.5,
            params={
                "asset_cfg": leg_joint_cfg,
                "command_name": COMMAND_NAME,
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
                "running_threshold": 1.5,
            },
        ),
        "stand_still": RewardTermCfg(
            func=mdp.stand_still,
            weight=-1.0,
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
            weight=-2.0,
            params={"asset_cfg": leg_joint_cfg},
        ),
        "leg_joint_vel": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.02,
            params={"asset_cfg": leg_joint_cfg},
        ),
        "wheel_joint_vel": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.002,
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
            weight=-1.0e-5,
            params={
                "sensor_name": "wheels_ground_contact",
                "command_name": COMMAND_NAME,
                "command_threshold": 0.05,
            },
        ),
        "wheel_air_time_balance": RewardTermCfg(
            func=mdp.wheel_air_time_balance,
            weight=-2.0,
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
            "gate_min": 0.10,
            "gate_max": 0.40,
            "roughness_gate_threshold": 0.4,
            "grid_shape": TERRAIN_SCAN_GRID_SHAPE,
        }
        rewards.update(
            {   # legged motion
                "rough_wheel_usage": RewardTermCfg(
                    func=mdp.rough_wheel_usage,
                    weight=-4.0e-2,
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
                    weight=0,
                    params={
                        **roughness_params,
                        "contact_sensor_name": "wheels_ground_contact",
                        "command_name": COMMAND_NAME,
                        "command_threshold": 0.05,
                    },
                
                # wheeled motion
                ),
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
                "standing_forward_wheel_air_time": RewardTermCfg(
                    func=mdp.standing_forward_wheel_air_time,
                    weight=-4.0,
                    params={
                        **roughness_params,
                        "contact_sensor_name": "wheels_ground_contact",
                        "command_name": COMMAND_NAME,
                        "max_time": 0.5,
                        "standing_scale": 2.5,
                        "forward_scale": 1.0,
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
            params={"limit_angle": math.radians(70.0)},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "illegal_ground_contact"},
        ),
    }
    if rough:
        terminations["out_of_terrain_bounds"] = TerminationTermCfg(
            func=mdp.out_of_terrain_bounds,
            params={"margin": 1.5},
            time_out=True,
        )
    return terminations


def make_curriculum(*, rough: bool) -> dict[str, CurriculumTermCfg]:
    """Terrain curriculum for rough training."""
    curriculum = {}
    if rough:
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={
                "command_name": COMMAND_NAME,
                "move_up_distance_ratio": 0.50,
                "min_command_path_ratio": 0.25,
                "move_up_progress_ratio": 0.65,
                "move_down_progress_ratio": 0.35,
            },
        )
    return curriculum


def make_metrics() -> dict[str, MetricsTermCfg]:
    return {
        "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
        "velocity_curriculum_progress": MetricsTermCfg(
            func=mdp.velocity_curriculum_progress,
            params={
                "command_name": COMMAND_NAME,
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
            },
        ),
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


def make_env_cfg(*, rough: bool, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = ManagerBasedRlEnvCfg(
        scene=make_scene(rough=rough),
        observations=make_observations(rough=rough),
        actions=make_actions(),
        commands=make_commands(),
        events=make_events(),
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
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    twist_cmd = cfg.commands[COMMAND_NAME]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-math.pi / 2, math.pi / 2)

    if rough:
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.terminations.pop("velocity_direction_deviation", None)
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        terrain = cfg.scene.terrain
        if terrain is not None and terrain.terrain_generator is not None:
            terrain.terrain_generator.curriculum = True
            terrain.terrain_generator.num_cols = 5
            terrain.terrain_generator.num_rows = 5
            terrain.terrain_generator.border_width = 10.0


def wf_tron1b_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create WF-TRON1B rough-terrain velocity tracking configuration."""
    return make_env_cfg(rough=True, play=play)


def wf_tron1b_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create WF-TRON1B flat-ground velocity tracking configuration."""
    return make_env_cfg(rough=False, play=play)
