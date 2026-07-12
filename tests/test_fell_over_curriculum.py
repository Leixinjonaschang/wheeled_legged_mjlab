from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    FELL_OVER_LIMIT_ANGLE_FINAL,
    FELL_OVER_LIMIT_ANGLE_INITIAL,
    FELL_OVER_LIMIT_ANGLE_RAMP_STEPS,
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
        0.3,
    )
    assert math.isclose(sub_terrains["discrete_obstacles"].proportion, 0.2)
    assert math.isclose(sub_terrains["random_rough"].proportion, 0.2)
    assert math.isclose(sub_terrains["hf_pyramid_slope"].proportion, 0.1)
    assert math.isclose(sub_terrains["hf_pyramid_slope_inv"].proportion, 0.1)
    assert math.isclose(sub_terrains["pyramid_stair_inv"].proportion, 0.2)
    assert math.isclose(sub_terrains["pyramid_stair"].proportion, 0.1)


def test_play_config_uses_all_interleaved_terrain_columns() -> None:
    terrain_generator = wf_tron1b_rough_env_cfg(play=True).scene.terrain.terrain_generator
    assert terrain_generator is not None
    assert terrain_generator.num_cols == len(EXPECTED_TERRAIN_COLUMNS)
    assert terrain_generator.num_rows == 5


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

    def update_env_origins(
        self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor
    ) -> None:
        del env_ids, move_up, move_down


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
    )

    result = terrain_levels_vel(env, torch.tensor([0, 1, 2]), command_name="base_velocity")

    assert result["flat"].item() == 2.0
    assert result["rough"].item() == 5.0
    assert "flat__0" not in result
    assert "flat__1" not in result
