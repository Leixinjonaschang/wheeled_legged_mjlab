# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: file-ignore[missing-type-function-argument, missing-type-args, missing-type-kwargs, missing-return-type-private-function, undocumented-public-function]

"""Tests for the retained velocity representation teacher-student PPO."""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import RepresentationVelocityTeacherStudentPPO
from rsl_rl.models import DepthRepresentationVelocityActorCritic, RepresentationVelocityActorCritic
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 6
PROPRIO_DIM = 28
COMMAND_DIM = 3
LIN_VEL_DIM = 3
CRITIC_DIM = 16
PRIVILEGED_DIM = 11
HISTORY_LENGTH = 5
NUM_ACTIONS = 2
DEPTH_SHAPE = (1, 32, 24)


def make_rep_obs() -> TensorDict:
    return TensorDict(
        {
            "proprio_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM),
            "actor_command": torch.randn(NUM_ENVS, COMMAND_DIM),
            "lin_vel_target": torch.randn(NUM_ENVS, LIN_VEL_DIM),
            "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
            "privileged_encoder": torch.randn(NUM_ENVS, PRIVILEGED_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def make_depth_rep_obs() -> TensorDict:
    obs = make_rep_obs()
    obs["depth_camera"] = torch.randn(NUM_ENVS, *DEPTH_SHAPE)
    return obs


def make_model(obs: TensorDict) -> RepresentationVelocityActorCritic:
    return RepresentationVelocityActorCritic(
        obs,
        {
            "proprio_history": ["proprio_history"],
            "actor_command": ["actor_command"],
            "lin_vel_target": ["lin_vel_target"],
            "critic": ["critic"],
            "privileged_encoder": ["privileged_encoder", "actor_command"],
        },
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=4,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def make_depth_model(obs: TensorDict) -> DepthRepresentationVelocityActorCritic:
    return DepthRepresentationVelocityActorCritic(
        obs,
        {
            "proprio_history": ["proprio_history"],
            "actor_command": ["actor_command"],
            "lin_vel_target": ["lin_vel_target"],
            "critic": ["critic"],
            "privileged_encoder": ["privileged_encoder", "actor_command"],
            "depth_encoder": ["depth_camera"],
        },
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=4,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def build_algorithm(*, depth: bool = False) -> tuple[RepresentationVelocityTeacherStudentPPO, TensorDict]:
    torch.manual_seed(12 if depth else 11)
    obs = make_depth_rep_obs() if depth else make_rep_obs()
    model = make_depth_model(obs) if depth else make_model(obs)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    algorithm = RepresentationVelocityTeacherStudentPPO(
        model,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        student_learning_rate=1.0e-3,
        schedule="fixed",
        desired_kl=0.01,
        num_representation_epochs=1,
        num_representation_mini_batches=2,
        representation_chunk_length=2,
    )
    return algorithm, obs


def fill_rollout(algorithm: RepresentationVelocityTeacherStudentPPO, obs: TensorDict) -> None:
    for _ in range(NUM_STEPS):
        algorithm.act(obs)
        next_obs = make_depth_rep_obs() if "depth_camera" in obs else make_rep_obs()
        algorithm.process_env_step(
            next_obs,
            torch.randn(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {},
        )
        obs = next_obs
    algorithm.compute_returns(obs)


def any_param_changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(before[name], parameter) for name, parameter in module.named_parameters())


def test_update_returns_student_losses_and_updates_parameter_groups() -> None:
    algorithm, obs = build_algorithm()
    fill_rollout(algorithm, obs)
    modules = {
        "actor": algorithm.actor.actor_head,
        "critic": algorithm.actor.critic_head,
        "privileged": algorithm.actor.privileged_encoder,
        "proprio": algorithm.actor.proprio_encoder,
        "latent": algorithm.actor.student_latent_head,
        "velocity": algorithm.actor.lin_vel_head,
    }
    before = {
        name: {key: value.detach().clone() for key, value in module.named_parameters()}
        for name, module in modules.items()
    }

    losses = algorithm.update()

    assert {"value", "surrogate", "entropy", "student", "representation", "lin_vel", "roughness"} <= losses.keys()
    assert all(any_param_changed(before[name], module) for name, module in modules.items())


def test_student_update_runs_after_all_ppo_minibatches() -> None:
    algorithm, obs = build_algorithm()
    fill_rollout(algorithm, obs)
    expected_ppo_calls = algorithm.num_learning_epochs * algorithm.num_mini_batches
    calls = {"ppo": 0, "student": 0}
    original_act_teacher = algorithm.actor.act_teacher
    original_compute_student_losses = algorithm.actor.compute_student_losses

    def counted_act_teacher(*args, **kwargs):
        calls["ppo"] += 1
        return original_act_teacher(*args, **kwargs)

    def counted_compute_student_losses(*args, **kwargs):
        assert calls["ppo"] == expected_ppo_calls
        calls["student"] += 1
        return original_compute_student_losses(*args, **kwargs)

    algorithm.actor.act_teacher = counted_act_teacher  # type: ignore[method-assign]
    algorithm.actor.compute_student_losses = counted_compute_student_losses  # type: ignore[method-assign]
    algorithm.update()

    assert calls == {"ppo": expected_ppo_calls, "student": expected_ppo_calls}


def test_plain_depth_update_has_no_predictor_dependency() -> None:
    algorithm, obs = build_algorithm(depth=True)
    fill_rollout(algorithm, obs)

    losses = algorithm.update()

    assert {"student", "representation", "lin_vel", "roughness"} <= losses.keys()
    assert all("latent_dynamics" not in name for name in losses)
    assert not hasattr(algorithm.actor, "latent_dynamics_predictors")


def test_student_loss_detaches_privileged_encoder_target() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    student_loss, _, _ = model.compute_student_losses(obs)
    model.zero_grad()
    student_loss.backward()

    assert any(parameter.grad is not None for parameter in model.proprio_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.privileged_encoder.parameters())


def test_multi_gpu_update_reduces_ppo_and_student_gradients() -> None:
    algorithm, obs = build_algorithm()
    fill_rollout(algorithm, obs)
    algorithm.is_multi_gpu = True
    ppo_ids = {id(parameter) for parameter in algorithm.actor.ppo_parameters()}
    student_ids = {id(parameter) for parameter in algorithm.actor.student_parameters()}
    reduced = {"ppo": False, "student": False}

    def fake_reduce_parameters(parameters=None) -> None:
        parameter_ids = {id(parameter) for parameter in parameters}
        if parameter_ids <= ppo_ids:
            reduced["ppo"] = True
        if parameter_ids <= student_ids:
            reduced["student"] = True

    algorithm.reduce_parameters = fake_reduce_parameters
    algorithm.update()

    assert reduced == {"ppo": True, "student": True}


def test_save_includes_student_optimizer() -> None:
    algorithm, _ = build_algorithm()

    saved = algorithm.save()

    assert "optimizer_state_dict" in saved
    assert "student_optimizer_state_dict" in saved


def test_unsupported_options_fail_loudly() -> None:
    obs = make_rep_obs()
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    with pytest.raises(ValueError, match="RND"):
        RepresentationVelocityTeacherStudentPPO(make_model(obs), storage, rnd_cfg={})
    with pytest.raises(ValueError, match="Symmetry"):
        RepresentationVelocityTeacherStudentPPO(make_model(obs), storage, symmetry_cfg={})
    with pytest.raises(ValueError, match="CNN encoder sharing"):
        RepresentationVelocityTeacherStudentPPO(make_model(obs), storage, share_cnn_encoders=True)
