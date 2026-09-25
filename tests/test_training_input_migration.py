"""Regression coverage for the migrated depth and dynamics training inputs."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import wheeled_legged_mjlab  # noqa: F401
from wheeled_legged_mjlab.tasks.velocity import mdp


def test_depth_invalid_pixels_and_custom_range_preserve_sensor_data():
    raw = torch.tensor(
        [
            99.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            -1.0,
            0.0,
            0.1,
            0.4,
            1.2,
            2.0,
            3.0,
        ]
    ).reshape(1, 1, -1, 1)
    original = raw.clone()
    env = SimpleNamespace(
        scene={"depth_camera": SimpleNamespace(data=SimpleNamespace(depth=raw))}
    )
    result = mdp.depth_image(env, left_crop=1, depth_min_m=0.4, depth_max_m=2.0)
    torch.testing.assert_close(
        result.flatten(),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0]),
    )
    torch.testing.assert_close(raw, original, equal_nan=True)
    for lower, upper in [(-0.1, 2.0), (2.0, 2.0), (3.0, 2.0)]:
        with pytest.raises(ValueError):
            mdp.depth_image(env, depth_min_m=lower, depth_max_m=upper)


def test_randomized_model_context_and_partial_gain_updates():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-WF-Tron1B-RepTS-LinVel")
    cfg.scene.num_envs = 4
    torch.manual_seed(123)
    env = ManagerBasedRlEnv(cfg, device="cpu", render_mode=None)
    try:
        env.reset()
        context = mdp.domain_randomization_delta_quantity(env)
        assert context.shape == (4, 87)
        assert torch.isfinite(context).all()
        assert context.abs().max() <= 1.0
        robot = env.scene["robot"]
        ids = robot.indexing.body_ids
        mass = (
            env.sim.model.body_mass[:, ids]
            / env.sim.get_default_field("body_mass")[ids]
        )
        inertia = (
            env.sim.model.body_inertia[:, ids].sort(dim=-1).values
            / env.sim.get_default_field("body_inertia")[ids].sort(dim=-1).values
        )
        assert ((mass >= 0.8) & (mass <= 1.2)).all()
        torch.testing.assert_close(
            inertia, mass[..., None].expand_as(inertia), atol=3e-5, rtol=3e-5
        )
        torch.testing.assert_close(context[:, 37:46], (mass - 1.0) / 0.2)
        torch.testing.assert_close(
            context[:, 46:73], ((inertia - 1.0) / 0.2).flatten(1)
        )
        for event, block, limit in [
            ("base_com", slice(10, 13), 0.03),
            ("link_com", slice(13, 37), 0.015),
        ]:
            entity_cfg = env.event_manager.get_term_cfg(event).params["asset_cfg"]
            body_ids = ids[entity_cfg.body_ids]
            delta = (
                env.sim.model.body_ipos[:, body_ids]
                - env.sim.get_default_field("body_ipos")[body_ids]
            )
            assert delta.abs().max() <= limit + 1e-6
            torch.testing.assert_close(context[:, block], delta.flatten(1) / limit)

        common = env.event_manager.get_term_cfg("wheel_friction")
        difference = env.event_manager.get_term_cfg("wheel_friction_difference")
        common.func(env, None, **{**common.params, "ranges": (0.7, 0.7)})
        difference.func(env, None, **{**difference.params, "ranges": (0.03, 0.03)})
        friction_ids = robot.indexing.geom_ids[common.params["asset_cfg"].geom_ids]
        torch.testing.assert_close(
            env.sim.model.geom_friction[:, friction_ids, 0], torch.full((4, 2), 0.73)
        )

        gains = env.sim.model.actuator_gainprm.clone()
        biases = env.sim.model.actuator_biasprm.clone()
        pd = env.event_manager.get_term_cfg("pd_gains")
        for _ in range(2):
            pd.func(
                env,
                torch.tensor([1]),
                **{
                    **pd.params,
                    "stiffness_scale_range": (0.9, 0.9),
                    "damping_scale_range": (1.1, 1.1),
                },
            )
        torch.testing.assert_close(
            env.sim.model.actuator_gainprm[[0, 2, 3]], gains[[0, 2, 3]]
        )
        torch.testing.assert_close(
            env.sim.model.actuator_biasprm[[0, 2, 3]], biases[[0, 2, 3]]
        )
        updated = mdp.domain_randomization_delta_quantity(env)
        torch.testing.assert_close(updated[1, 73:79], torch.full((6,), -0.5))
        torch.testing.assert_close(updated[1, 79:87], torch.full((8,), 0.5))
        obs, rewards, _, _, _ = env.step(
            torch.zeros(4, env.action_manager.total_action_dim)
        )
        assert obs["dynamics_context"].shape == (4, 87)
        assert torch.isfinite(rewards).all()
    finally:
        env.close()
