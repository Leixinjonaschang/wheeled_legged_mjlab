from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from wheeled_legged_mjlab.assets.WF_TRON1B.wf_tron1b import (
    WF_TRON1B_INIT_STATE,
    WF_TRON1B_XML,
)
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    BASE_HEIGHT_TARGET,
    FELL_OVER_LIMIT_ANGLE_FINAL,
    FELL_OVER_LIMIT_ANGLE_INITIAL,
    FELL_OVER_LIMIT_ANGLE_RAMP_STEPS,
    POSE_TARGET_JOINT_POS,
    WHEEL_DISTANCE_RANGE,
    WHEEL_RADIUS,
    wf_tron1b_flat_env_cfg,
    wf_tron1b_rough_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.terrain_cfg import (
    TERRAINS_CFG,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.curriculums import (
    fell_over_limit_angle,
    terrain_levels_vel,
)
from wheeled_legged_mjlab.tasks.velocity.mdp import rewards as reward_terms
from wheeled_legged_mjlab.tasks.velocity.mdp.rewards import (
    base_height_l2,
    variable_posture,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.terminations import out_of_terrain_bounds


@dataclass
class DummyTerminationCfg:
    params: dict[str, float] = field(default_factory=dict)


class DummyTerminationManager:
    def __init__(self) -> None:
        self.cfg = DummyTerminationCfg(params={"limit_angle": -1.0})

    def get_term_cfg(self, term_name: str) -> DummyTerminationCfg:
        assert term_name == "fell_over"
        return self.cfg


class DummyEnv:
    def __init__(self, common_step_counter: int) -> None:
        self.common_step_counter = common_step_counter
        self.device = "cpu"
        self.termination_manager = DummyTerminationManager()


def apply_curriculum(env: DummyEnv) -> float:
    state = fell_over_limit_angle(
        env,
        torch.tensor([0]),
        termination_term_name="fell_over",
        initial_limit_angle=FELL_OVER_LIMIT_ANGLE_INITIAL,
        final_limit_angle=FELL_OVER_LIMIT_ANGLE_FINAL,
        ramp_steps=FELL_OVER_LIMIT_ANGLE_RAMP_STEPS,
    )
    assert math.isclose(
        state["limit_angle"].item(),
        env.termination_manager.cfg.params["limit_angle"],
        rel_tol=1.0e-6,
    )
    return env.termination_manager.cfg.params["limit_angle"]


def test_fell_over_limit_angle_curriculum_interpolates_and_clamps() -> None:
    start_env = DummyEnv(common_step_counter=0)
    midpoint_env = DummyEnv(common_step_counter=FELL_OVER_LIMIT_ANGLE_RAMP_STEPS // 2)
    finished_env = DummyEnv(common_step_counter=2 * FELL_OVER_LIMIT_ANGLE_RAMP_STEPS)

    assert math.isclose(apply_curriculum(start_env), FELL_OVER_LIMIT_ANGLE_INITIAL)
    assert math.isclose(
        apply_curriculum(midpoint_env),
        (FELL_OVER_LIMIT_ANGLE_INITIAL + FELL_OVER_LIMIT_ANGLE_FINAL) / 2,
    )
    assert math.isclose(apply_curriculum(finished_env), FELL_OVER_LIMIT_ANGLE_FINAL)


def test_training_and_play_configs_use_expected_fell_over_limits() -> None:
    for cfg_factory in (wf_tron1b_flat_env_cfg, wf_tron1b_rough_env_cfg):
        training_cfg = cfg_factory()
        play_cfg = cfg_factory(play=True)

        assert math.isclose(
            training_cfg.terminations["fell_over"].params["limit_angle"],
            FELL_OVER_LIMIT_ANGLE_INITIAL,
        )
        assert "fell_over_limit_angle" in training_cfg.curriculum
        assert math.isclose(
            play_cfg.terminations["fell_over"].params["limit_angle"],
            FELL_OVER_LIMIT_ANGLE_FINAL,
        )
        assert play_cfg.curriculum == {}


EXPECTED_TERRAIN_COLUMNS = (
    "flat__0",
    "discrete_obstacles",
    "flat__1",
    "random_rough",
    "flat__2",
    "hf_pyramid_slope",
    "flat__3",
    "hf_pyramid_slope_inv",
    "flat__4",
    "pyramid_stair_inv",
    "flat__5",
    "pyramid_stair",
    "flat__6",
    "random_stairs",
    "flat__7",
    "random_spread",
    "flat__8",
    "stepping_stones",
    "flat__9",
    "tilted_grid",
)


def test_interleaved_terrain_columns_preserve_logical_proportions() -> None:
    sub_terrains = TERRAINS_CFG.sub_terrains
    assert tuple(sub_terrains) == EXPECTED_TERRAIN_COLUMNS
    assert TERRAINS_CFG.num_cols == len(EXPECTED_TERRAIN_COLUMNS)

    logical_names = [name.split("__", maxsplit=1)[0] for name in sub_terrains]
    assert all(
        {logical_names[i], logical_names[i + 1]} != {"flat"}
        and "flat" in {logical_names[i], logical_names[i + 1]}
        for i in range(len(logical_names) - 1)
    )

    assert math.isclose(
        sum(cfg.proportion for name, cfg in sub_terrains.items() if name.startswith("flat__")),
        0.28,
    )
    assert math.isclose(sum(cfg.proportion for cfg in sub_terrains.values()), 1.0)
    assert math.isclose(sub_terrains["discrete_obstacles"].proportion, 0.10)
    assert math.isclose(sub_terrains["random_rough"].proportion, 0.08)
    for name in (
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
        "pyramid_stair",
        "random_stairs",
        "tilted_grid",
    ):
        assert math.isclose(sub_terrains[name].proportion, 0.05)
    assert math.isclose(sub_terrains["pyramid_stair_inv"].proportion, 0.15)
    assert math.isclose(sub_terrains["random_spread"].proportion, 0.07)
    assert math.isclose(sub_terrains["stepping_stones"].proportion, 0.07)

    stepping_stones = sub_terrains["stepping_stones"]
    assert stepping_stones.stone_distance_range == (0.0, 0.15)
    assert stepping_stones.stone_size_range == (0.50, 0.75)
    assert stepping_stones.stone_height == 0.0
    assert math.isclose(
        (stepping_stones.stone_distance_range[1] - stepping_stones.stone_distance_range[0])
        / TERRAINS_CFG.num_rows,
        0.003,
    )
    assert math.isclose(
        (stepping_stones.stone_size_range[1] - stepping_stones.stone_size_range[0])
        / TERRAINS_CFG.num_rows,
        0.005,
    )
    assert stepping_stones.stone_height_variation / TERRAINS_CFG.num_rows <= 0.0015
    assert stepping_stones.displacement_range / TERRAINS_CFG.num_rows <= 0.0015

    random_stairs = sub_terrains["random_stairs"]
    assert random_stairs.step_width == 0.35
    assert random_stairs.step_height_range == (0.03, 0.25)

    tilted_grid = sub_terrains["tilted_grid"]
    assert tilted_grid.grid_width == 0.6
    assert tilted_grid.tilt_range_deg == 15.0
    assert tilted_grid.height_range == 0.15

    random_spread = sub_terrains["random_spread"]
    assert random_spread.num_boxes == 64
    assert random_spread.box_height_range == (0.03, 0.25)


def test_play_config_uses_all_interleaved_terrain_columns() -> None:
    terrain_generator = wf_tron1b_rough_env_cfg(play=True).scene.terrain.terrain_generator
    assert terrain_generator is not None
    assert terrain_generator.num_cols == len(EXPECTED_TERRAIN_COLUMNS)
    assert terrain_generator.num_rows == 5


def _resolved_wf_tron1b_joint_positions(
    model: mujoco.MjModel, joint_pos_expr: dict[str, float]
) -> dict[str, float]:
    patterns = [(re.compile(pattern), value) for pattern, value in joint_pos_expr.items()]
    joint_pos = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        value = 0.0
        for pattern, candidate in patterns:
            if pattern.match(name):
                value = candidate
                break
        joint_pos[name] = value
    return joint_pos


def _wf_tron1b_wheel_geometry(joint_pos: dict[str, float]):
    model = mujoco.MjModel.from_xml_path(str(WF_TRON1B_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    for joint_name, value in joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = value

    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    wheel_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_L_Link"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_R_Link"),
    ]
    base_pos = data.xpos[base_id].copy()
    wheel_pos = torch.tensor(
        np.array([data.xpos[wheel_id].copy() for wheel_id in wheel_ids])
    )
    wheel_pos_b = wheel_pos - torch.tensor(base_pos)

    base_height = float(-wheel_pos_b[:, 2].mean() + WHEEL_RADIUS)
    wheel_distance = float(
        torch.linalg.norm(wheel_pos_b[0, :2] - wheel_pos_b[1, :2])
    )
    return base_height, wheel_distance, wheel_pos_b


def test_wf_tron1b_initial_state_keeps_reset_clearance() -> None:
    model = mujoco.MjModel.from_xml_path(str(WF_TRON1B_XML))
    initial_joint_pos = _resolved_wf_tron1b_joint_positions(
        model, WF_TRON1B_INIT_STATE.joint_pos
    )

    assert WF_TRON1B_INIT_STATE.joint_pos == {".*": 0.0}
    assert math.isclose(WF_TRON1B_INIT_STATE.pos[2], 0.8 + 0.166)
    assert all(value == 0.0 for value in initial_joint_pos.values())


def test_wf_tron1b_pose_target_matches_base_height_target() -> None:
    model = mujoco.MjModel.from_xml_path(str(WF_TRON1B_XML))
    pose_target_joint_pos = _resolved_wf_tron1b_joint_positions(
        model, POSE_TARGET_JOINT_POS
    )
    pose_target_height, wheel_distance, wheel_pos_b = _wf_tron1b_wheel_geometry(
        pose_target_joint_pos
    )
    initial_height, _, _ = _wf_tron1b_wheel_geometry(
        _resolved_wf_tron1b_joint_positions(model, WF_TRON1B_INIT_STATE.joint_pos)
    )
    cfg = wf_tron1b_flat_env_cfg()

    assert cfg.rewards["pose"].params["target_joint_pos"] == POSE_TARGET_JOINT_POS
    assert math.isclose(pose_target_height, BASE_HEIGHT_TARGET, abs_tol=2.0e-3)
    assert initial_height - pose_target_height > 0.08
    assert WHEEL_DISTANCE_RANGE[0] <= wheel_distance <= WHEEL_DISTANCE_RANGE[1]
    assert math.isclose(
        float(wheel_pos_b[0, 0]), float(wheel_pos_b[1, 0]), abs_tol=1.0e-6
    )
    assert math.isclose(
        float(wheel_pos_b[0, 2]), float(wheel_pos_b[1, 2]), abs_tol=1.0e-6
    )


def test_base_height_quantile_ignores_sparse_stepping_stone_pit_samples(
    monkeypatch,
) -> None:
    class DummyRayCastSensor:
        def __init__(self) -> None:
            self.data = SimpleNamespace(
                distances=torch.ones(1, 4),
                hit_pos_w=torch.tensor(
                    [[[[0.0, 0.0, -0.8], [0.0, 0.0, -0.8],
                       [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
                ),
            )

    monkeypatch.setattr(reward_terms, "RayCastSensor", DummyRayCastSensor)
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.82]]))
            ),
            "terrain_scan": DummyRayCastSensor(),
        }
    )

    mean_cost = base_height_l2(
        env, target_height=0.82, sensor_name="terrain_scan", terrain_sample="mean"
    )
    support_cost = base_height_l2(
        env,
        target_height=0.82,
        sensor_name="terrain_scan",
        terrain_sample="quantile",
        terrain_quantile=0.75,
    )

    assert mean_cost.item() > 0.1
    assert torch.allclose(support_cost, torch.zeros(1))


class DummyPostureAsset:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros(1, 2),
            joint_pos=torch.tensor([[0.2, -0.2]]),
        )

    def find_joints(self, joint_names):
        assert joint_names == ("joint_a", "joint_b")
        return [0, 1], ["joint_a", "joint_b"]


def test_variable_posture_uses_configured_target_pose() -> None:
    asset_cfg = SimpleNamespace(
        name="robot",
        joint_names=("joint_a", "joint_b"),
        joint_ids=[0, 1],
    )
    cfg = SimpleNamespace(
        params={
            "asset_cfg": asset_cfg,
            "target_joint_pos": {"joint_a": 0.2, "joint_b": -0.2},
            "std_standing": {".*": 0.1},
            "std_walking": {".*": 0.1},
            "std_running": {".*": 0.1},
        }
    )
    env = SimpleNamespace(
        device="cpu",
        scene={"robot": DummyPostureAsset()},
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.zeros(1, 3),
        ),
    )

    reward = variable_posture(cfg, env)(
        env,
        std_standing=None,
        std_walking=None,
        std_running=None,
        asset_cfg=asset_cfg,
        command_name="twist",
        target_joint_pos=cfg.params["target_joint_pos"],
        walking_threshold=0.05,
        running_threshold=1.5,
    )

    assert torch.allclose(reward, torch.ones(1))


def test_non_rough_flat_orientation_applies_only_on_non_rough_terrain(
    monkeypatch,
) -> None:
    asset_cfg = SimpleNamespace(name="robot")
    env = SimpleNamespace(
        common_step_counter=0,
        extras={},
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    projected_gravity_b=torch.tensor([[0.1, 0.2, -0.97]])
                )
            )
        },
    )

    monkeypatch.setattr(
        reward_terms,
        "_terrain_roughness_from_sensor",
        lambda *args, **kwargs: SimpleNamespace(gate=torch.tensor([0.0])),
    )
    flat_cost = reward_terms.non_rough_flat_orientation(
        env,
        roughness_sensor_name="terrain_scan",
        asset_cfg=asset_cfg,
        roll_weight=2.0,
        pitch_weight=1.0,
    )
    assert torch.allclose(flat_cost, torch.tensor([0.09]))

    monkeypatch.setattr(
        reward_terms,
        "_terrain_roughness_from_sensor",
        lambda *args, **kwargs: SimpleNamespace(gate=torch.tensor([0.3])),
    )
    rough_cost = reward_terms.non_rough_flat_orientation(
        env,
        roughness_sensor_name="terrain_scan",
        asset_cfg=asset_cfg,
        roll_weight=2.0,
        pitch_weight=1.0,
    )
    assert torch.allclose(rough_cost, torch.zeros(1))


def test_non_rough_flat_orientation_replaces_base_ang_vel_reward() -> None:
    cfg = wf_tron1b_rough_env_cfg()

    assert "non_rough_base_ang_vel_xy" not in cfg.rewards
    term = cfg.rewards["non_rough_flat_orientation"]
    assert term.func is reward_terms.non_rough_flat_orientation
    assert math.isclose(term.weight, -10.0)
    assert term.params["roll_weight"] == 2.0
    assert term.params["pitch_weight"] == 1.0


class DummyTerrain:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            terrain_generator=SimpleNamespace(
                size=(8.0, 8.0),
                sub_terrains={"flat__0": object(), "rough": object(), "flat__1": object()},
            )
        )
        self.terrain_levels = torch.tensor([1, 5, 3])
        self.terrain_origins = torch.zeros(1, 3, 3)
        self.terrain_types = torch.tensor([0, 1, 2])
        self.updated_env_ids: list[torch.Tensor] = []

    def update_env_origins(
        self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor
    ) -> None:
        del move_up, move_down
        self.updated_env_ids.append(env_ids.clone())


class DummyTerrainScene:
    def __init__(self) -> None:
        self.terrain = DummyTerrain()
        self.env_origins = torch.zeros(3, 3)
        self._robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(3, 3)))

    def __getitem__(self, name: str) -> SimpleNamespace:
        assert name == "robot"
        return self._robot


def test_terrain_curriculum_groups_repeated_flat_columns() -> None:
    env = SimpleNamespace(
        scene=DummyTerrainScene(),
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[1.0, 0.0]]).repeat(3, 1)
        ),
        max_episode_length_s=20.0,
        common_step_counter=1,
    )

    result = terrain_levels_vel(env, torch.tensor([0, 1, 2]), command_name="base_velocity")

    assert result["flat"].item() == 2.0
    assert result["rough"].item() == 5.0
    assert "flat__0" not in result
    assert "flat__1" not in result


def test_terrain_curriculum_skips_initial_reset_and_scopes_partial_reset() -> None:
    scene = DummyTerrainScene()
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[1.0, 0.0]]).repeat(3, 1)
        ),
        max_episode_length_s=20.0,
        common_step_counter=0,
    )

    terrain_levels_vel(env, torch.tensor([0, 1, 2]), command_name="base_velocity")
    assert scene.terrain.updated_env_ids == []

    env.common_step_counter = 1
    terrain_levels_vel(env, torch.tensor([1]), command_name="base_velocity")
    assert len(scene.terrain.updated_env_ids) == 1
    assert torch.equal(scene.terrain.updated_env_ids[0], torch.tensor([1]))


class DummyTerrainBoundsScene:
    def __init__(self, root_xy: list[tuple[float, float]]) -> None:
        self.terrain = SimpleNamespace(
            cfg=SimpleNamespace(
                terrain_type="generator",
                terrain_generator=SimpleNamespace(
                    size=(8.0, 6.0),
                    border_width=5.0,
                ),
            ),
            terrain_origins=torch.zeros((2, 3, 3)),
        )
        root_pos = torch.zeros((len(root_xy), 3))
        root_pos[:, :2] = torch.tensor(root_xy)
        self._robot = SimpleNamespace(
            data=SimpleNamespace(root_link_pos_w=root_pos)
        )

    def __getitem__(self, name: str) -> SimpleNamespace:
        assert name == "robot"
        return self._robot


def test_out_of_terrain_bounds_margin_is_distance_beyond_effective_terrain() -> None:
    root_xy = [
        (8.04, 0.0),
        (8.06, 0.0),
        (-8.06, 0.0),
        (0.0, 9.04),
        (0.0, 9.06),
    ]
    env = SimpleNamespace(
        num_envs=len(root_xy),
        device="cpu",
        scene=DummyTerrainBoundsScene(root_xy),
    )

    result = out_of_terrain_bounds(env, margin=0.05)

    assert result.tolist() == [False, True, True, False, True]


def test_rough_training_uses_small_timeout_margin() -> None:
    cfg = wf_tron1b_rough_env_cfg()
    term = cfg.terminations["out_of_terrain_bounds"]

    assert term.params["margin"] == 0.05
    assert term.time_out is True
    assert cfg.scene.terrain.terrain_generator.border_width == 5.0

    play_cfg = wf_tron1b_rough_env_cfg(play=True)
    assert play_cfg.scene.terrain.terrain_generator.border_width == 5.0
