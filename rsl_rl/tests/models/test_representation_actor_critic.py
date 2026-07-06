# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the representation-level actor-critic model."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import RepresentationActorCritic

NUM_ENVS = 4
ACTOR_DIM = 6
CRITIC_DIM = 10
HISTORY_LENGTH = 5
LATENT_DIM = 3
NUM_ACTIONS = 2


def make_rep_obs(include_privileged: bool = True) -> TensorDict:
    data = {
        "student_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, ACTOR_DIM),
    }
    if include_privileged:
        data["teacher_actor"] = torch.randn(NUM_ENVS, ACTOR_DIM)
        data["critic"] = torch.randn(NUM_ENVS, CRITIC_DIM)
    return TensorDict(data, batch_size=[NUM_ENVS])


def make_model(obs: TensorDict | None = None) -> RepresentationActorCritic:
    obs = make_rep_obs() if obs is None else obs
    obs_groups = {
        "teacher_actor": ["teacher_actor"],
        "critic": ["critic"],
        "student_history": ["student_history"],
        "privileged_encoder": ["critic"],
    }
    return RepresentationActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def test_teacher_student_and_value_paths_have_expected_shapes() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    teacher_actions = model.act_teacher(obs, stochastic_output=True)
    student_actions = model(obs)
    values = model.evaluate_teacher(obs)
    privileged_latent = model.get_privileged_latent(obs)
    proprio_latent = model.get_proprio_latent(obs)

    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert student_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert values.shape == (NUM_ENVS, 1)
    assert privileged_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert proprio_latent.shape == (NUM_ENVS, LATENT_DIM)


def test_actor_history_dim_is_five_frames_of_actor_dim() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    assert model.proprio_encoder_obs_dim == HISTORY_LENGTH * model.student_actor_obs_dim


def test_student_actor_is_the_latest_frame_of_student_history() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    assert torch.equal(model.get_student_actor_obs(obs), obs["student_history"][:, -1, :])
    assert torch.equal(model.get_proprio_obs(obs), obs["student_history"].flatten(start_dim=1))


def test_student_inference_does_not_require_critic_observations() -> None:
    model = make_model(make_rep_obs())
    inference_obs = make_rep_obs(include_privileged=False)

    actions = model(inference_obs)

    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)


def test_exported_student_policy_uses_one_history_input() -> None:
    model = make_model(make_rep_obs())
    student_history = torch.randn(NUM_ENVS, HISTORY_LENGTH, ACTOR_DIM)
    inference_obs = TensorDict({"student_history": student_history}, batch_size=[NUM_ENVS])
    expected_actions = model(inference_obs)

    jit_policy = model.as_jit()
    onnx_policy = model.as_onnx(verbose=False)

    assert torch.allclose(jit_policy(student_history), expected_actions)
    assert torch.allclose(onnx_policy(student_history), expected_actions)
    assert onnx_policy.input_names == ["student_history"]
    assert onnx_policy.get_dummy_inputs()[0].shape == (1, HISTORY_LENGTH, ACTOR_DIM)
