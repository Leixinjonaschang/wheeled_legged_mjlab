from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    FELL_OVER_LIMIT_ANGLE_FINAL,
    FELL_OVER_LIMIT_ANGLE_INITIAL,
    FELL_OVER_LIMIT_ANGLE_RAMP_STEPS,
    wf_tron1b_flat_env_cfg,
    wf_tron1b_rough_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.curriculums import (
    fell_over_limit_angle,
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
