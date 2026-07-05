# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for representation-level teacher-student PPO."""

from __future__ import annotations

import torch
import pytest
from tensordict import TensorDict

from rsl_rl.algorithms import RepresentationTeacherStudentPPO
from rsl_rl.models import RepresentationActorCritic
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 4
ACTOR_DIM = 6
CRITIC_DIM = 10
HISTORY_LENGTH = 5
NUM_ACTIONS = 2


def make_rep_obs() -> TensorDict:
    return TensorDict(
        {
            "teacher_actor": torch.randn(NUM_ENVS, ACTOR_DIM),
            "student_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, ACTOR_DIM),
            "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def make_model(obs: TensorDict) -> RepresentationActorCritic:
    return RepresentationActorCritic(
        obs,
        {
            "teacher_actor": ["teacher_actor"],
            "critic": ["critic"],
            "student_history": ["student_history"],
            "privileged_encoder": ["critic"],
        },
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=4,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def build_algorithm() -> tuple[RepresentationTeacherStudentPPO, TensorDict]:
    torch.manual_seed(7)
    obs = make_rep_obs()
    model = make_model(obs)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = RepresentationTeacherStudentPPO(
        model,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        proprio_encoder_learning_rate=1.0e-3,
        schedule="fixed",
        desired_kl=0.01,
    )
    return alg, obs


def fill_rollout(alg: RepresentationTeacherStudentPPO, obs: TensorDict) -> TensorDict:
    for _ in range(NUM_STEPS):
        alg.act(obs)
        next_obs = make_rep_obs()
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        alg.process_env_step(next_obs, rewards, dones, {})
        obs = next_obs
    alg.compute_returns(obs)
    return obs


def any_param_changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(before[name], param) for name, param in module.named_parameters())


def test_update_returns_representation_loss_and_updates_parameter_groups() -> None:
    alg, obs = build_algorithm()
    fill_rollout(alg, obs)

    actor_before = {n: p.detach().clone() for n, p in alg.actor.actor_head.named_parameters()}
    critic_before = {n: p.detach().clone() for n, p in alg.actor.critic_head.named_parameters()}
    privileged_before = {n: p.detach().clone() for n, p in alg.actor.privileged_encoder.named_parameters()}
    proprio_before = {n: p.detach().clone() for n, p in alg.actor.proprio_encoder.named_parameters()}

    losses = alg.update()

    assert {"value", "surrogate", "entropy", "representation"} <= set(losses)
    assert any_param_changed(actor_before, alg.actor.actor_head)
    assert any_param_changed(critic_before, alg.actor.critic_head)
    assert any_param_changed(privileged_before, alg.actor.privileged_encoder)
    assert any_param_changed(proprio_before, alg.actor.proprio_encoder)


def test_representation_loss_detaches_privileged_encoder_target() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    loss = model.compute_representation_loss(obs)
    model.zero_grad()
    loss.backward()

    assert any(param.grad is not None for param in model.proprio_encoder.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_multi_gpu_update_reduces_ppo_and_proprio_gradients() -> None:
    alg, obs = build_algorithm()
    fill_rollout(alg, obs)
    alg.is_multi_gpu = True

    ppo_param_ids = {id(param) for param in alg.actor.ppo_parameters()}
    proprio_param_ids = {id(param) for param in alg.actor.proprio_parameters()}
    reduced = {"ppo": False, "proprio": False}

    def fake_reduce_parameters(parameters=None) -> None:
        param_ids = {id(param) for param in parameters}
        if param_ids <= ppo_param_ids:
            reduced["ppo"] = True
        if param_ids <= proprio_param_ids:
            reduced["proprio"] = True

    alg.reduce_parameters = fake_reduce_parameters

    alg.update()

    assert reduced == {"ppo": True, "proprio": True}


def test_unsupported_options_fail_loudly() -> None:
    obs = make_rep_obs()
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    with pytest.raises(ValueError, match="RND"):
        RepresentationTeacherStudentPPO(make_model(obs), storage, rnd_cfg={})
    with pytest.raises(ValueError, match="Symmetry"):
        RepresentationTeacherStudentPPO(make_model(obs), storage, symmetry_cfg={})
    with pytest.raises(ValueError, match="CNN encoder sharing"):
        RepresentationTeacherStudentPPO(make_model(obs), storage, share_cnn_encoders=True)
