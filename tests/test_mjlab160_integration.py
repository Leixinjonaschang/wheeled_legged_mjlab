from __future__ import annotations

import pytest
import torch

import wheeled_legged_mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MJLab GPU integration test")
def test_flat_environment_terms_have_strict_vector_outputs() -> None:
    cfg = load_env_cfg("Mjlab-Velocity-Flat-WF-Tron1B")
    cfg.scene.num_envs = 2
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
