from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import action_rate_l2
from mjlab.tasks.registry import load_env_cfg

import wheeled_legged_mjlab  # noqa: F401
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    make_metrics,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.rewards import (
    ActionSmoothnessPenalty,
    action_term_rate_l2,
    action_term_smoothness_l2,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.rewards import (
    action_rate_l2 as weighted_action_rate_l2,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MJLab GPU integration test")
def test_flat_environment_terms_have_strict_vector_outputs() -> None:
    cfg = load_env_cfg("Mjlab-Velocity-Flat-WF-Tron1B")
    cfg.scene.num_envs = 2
    assert "action_rate" not in cfg.rewards
    assert "action_smoothness" not in cfg.rewards
    for group, action_term_name in (("leg", "leg_pos"), ("wheel", "wheel_vel")):
        for name, func in (
            ("action_rate", action_term_rate_l2),
            ("action_smoothness", action_term_smoothness_l2),
        ):
            term = cfg.rewards[f"{group}_{name}"]
            assert term.func is func
            assert term.params == {"action_term_name": action_term_name}
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        env.reset()
        actions = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
        )
        _, reward, terminated, truncated, _ = env.step(actions)

        for value in (reward, terminated, truncated):
            assert value.shape == (env.num_envs,)
            assert torch.isfinite(value).all()

        for manager in (
            env.reward_manager,
            env.termination_manager,
            env.metrics_manager,
        ):
            for name, term_cfg in zip(
                manager.active_terms,
                manager._term_cfgs,
                strict=True,
            ):
                value = term_cfg.func(env, **term_cfg.params)
                assert value.shape == (env.num_envs,), name
                assert torch.isfinite(value).all(), name
    finally:
        env.close()


class _ActionManager:
    def __init__(
        self,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        prev_prev_action: torch.Tensor,
    ) -> None:
        self.active_terms = ["leg_pos", "wheel_vel"]
        self.action = action
        self.prev_action = prev_action
        self.prev_prev_action = prev_prev_action
        self._terms = {
            "leg_pos": SimpleNamespace(action_dim=6),
            "wheel_vel": SimpleNamespace(action_dim=2),
        }

    def get_term(self, name: str) -> SimpleNamespace:
        return self._terms[name]


def _make_action_metric_env(
    action: torch.Tensor,
    prev_action: torch.Tensor,
    prev_prev_action: torch.Tensor,
    episode_length_buf: torch.Tensor,
) -> SimpleNamespace:
    return SimpleNamespace(
        action_manager=_ActionManager(action, prev_action, prev_prev_action),
        episode_length_buf=episode_length_buf,
    )


def test_action_rate_metrics_split_leg_and_wheel_terms() -> None:
    action = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
        ]
    )
    zeros = torch.zeros_like(action)
    env = _make_action_metric_env(action, zeros, zeros, torch.full((2,), 3))

    leg_rate = action_term_rate_l2(env, "leg_pos")
    wheel_rate = action_term_rate_l2(env, "wheel_vel")

    assert torch.allclose(leg_rate, torch.tensor([91.0, 91.0]))
    assert torch.allclose(wheel_rate, torch.tensor([113.0, 113.0]))
    assert torch.allclose(leg_rate + wheel_rate, action_rate_l2(env))


def test_action_smoothness_metrics_split_terms_and_mask_warmup() -> None:
    prev_prev_action = torch.zeros((2, 8))
    prev_action = torch.ones((2, 8))
    action = torch.full((2, 8), 3.0)
    env = _make_action_metric_env(
        action,
        prev_action,
        prev_prev_action,
        torch.tensor([3, 2]),
    )

    leg_smoothness = action_term_smoothness_l2(env, "leg_pos")
    wheel_smoothness = action_term_smoothness_l2(env, "wheel_vel")

    assert torch.allclose(leg_smoothness, torch.tensor([6.0, 0.0]))
    assert torch.allclose(wheel_smoothness, torch.tensor([2.0, 0.0]))


def test_make_metrics_registers_split_action_metrics() -> None:
    metrics = make_metrics()

    assert list(metrics) == [
        "mean_action_acc",
        "leg_action_rate",
        "wheel_action_rate",
        "leg_action_smoothness",
        "wheel_action_smoothness",
    ]
    assert metrics["leg_action_rate"].params == {"action_term_name": "leg_pos"}
    assert metrics["wheel_action_rate"].params == {
        "action_term_name": "wheel_vel"
    }
    assert metrics["leg_action_smoothness"].params == {
        "action_term_name": "leg_pos"
    }
    assert metrics["wheel_action_smoothness"].params == {
        "action_term_name": "wheel_vel"
    }


@pytest.mark.parametrize("reverse_terms", [False, True])
@pytest.mark.parametrize(
    ("leg_coefficient", "wheel_coefficient"),
    [(1.0, 1.0), (0.2, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
)
def test_weighted_action_costs_and_partial_reset(
    reverse_terms: bool, leg_coefficient: float, wheel_coefficient: float
) -> None:
    # Nonuniform actions distinguish both groups and catch incorrect slicing.
    base = torch.arange(1.0, 9.0).repeat(2, 1)
    if reverse_terms:
        base = torch.cat((base[:, 6:], base[:, :6]), dim=1)
    env = _make_action_metric_env(
        torch.zeros_like(base), torch.zeros_like(base), torch.zeros_like(base),
        torch.zeros(2, dtype=torch.long),
    )
    manager = env.action_manager
    if reverse_terms:
        manager.active_terms.reverse()
    smoothness = ActionSmoothnessPenalty(None, env)
    default_smoothness = ActionSmoothnessPenalty(None, env)
    coefficients = {
        "leg_coefficient": leg_coefficient, "wheel_coefficient": wheel_coefficient
    }
    for step, multiplier in enumerate((1.0, 2.0, 4.0, 7.0, 11.0, 16.0), start=1):
        if step == 4:
            smoothness.reset(torch.tensor([0]))
            default_smoothness.reset(torch.tensor([0]))
            for history in (manager.action, manager.prev_action, manager.prev_prev_action):
                history[0] = 0.0
            env.episode_length_buf[0] = 0
        manager.prev_prev_action = manager.prev_action.clone()
        manager.prev_action = manager.action.clone()
        manager.action = multiplier * base
        env.episode_length_buf += 1

        expected_rate = (
            leg_coefficient * action_term_rate_l2(env, "leg_pos")
            + wheel_coefficient * action_term_rate_l2(env, "wheel_vel")
        )
        expected_smoothness = (
            leg_coefficient * action_term_smoothness_l2(env, "leg_pos")
            + wheel_coefficient * action_term_smoothness_l2(env, "wheel_vel")
        )
        torch.testing.assert_close(weighted_action_rate_l2(env, **coefficients), expected_rate)
        torch.testing.assert_close(smoothness(env, **coefficients), expected_smoothness)
        torch.testing.assert_close(weighted_action_rate_l2(env), action_rate_l2(env))
        expected_default_smoothness = torch.sum(
            (manager.action - 2 * manager.prev_action + manager.prev_prev_action).square(), dim=1
        )
        expected_default_smoothness[env.episode_length_buf < 3] = 0.0
        torch.testing.assert_close(default_smoothness(env), expected_default_smoothness)
        if step == 3:
            # Second difference is exactly base: leg sum=91, wheel sum=113.
            torch.testing.assert_close(
                expected_smoothness,
                torch.full((2,), leg_coefficient * 91.0 + wheel_coefficient * 113.0),
            )
