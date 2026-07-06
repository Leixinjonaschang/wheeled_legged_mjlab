# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: D103

"""Tests for the velocity representation actor-critic model."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import RepresentationVelocityActorCritic

NUM_ENVS = 4
PROPRIO_DIM = 28
COMMAND_DIM = 3
LIN_VEL_DIM = 3
CRITIC_DIM = 16
PRIVILEGED_DIM = 11
HISTORY_LENGTH = 5
LATENT_DIM = 4
NUM_ACTIONS = 2


def make_rep_obs(include_privileged: bool = True) -> TensorDict:
    data = {
        "proprio_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM),
        "actor_command": torch.randn(NUM_ENVS, COMMAND_DIM),
    }
    if include_privileged:
        data.update(
            {
                "lin_vel_target": torch.randn(NUM_ENVS, LIN_VEL_DIM),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                "privileged_encoder": torch.randn(NUM_ENVS, PRIVILEGED_DIM),
            }
        )
    return TensorDict(data, batch_size=[NUM_ENVS])


def make_model(obs: TensorDict | None = None, *, obs_normalization: bool = False) -> RepresentationVelocityActorCritic:
    obs = make_rep_obs() if obs is None else obs
    obs_groups = {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "critic": ["critic"],
        "privileged_encoder": ["privileged_encoder"],
    }
    return RepresentationVelocityActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        obs_normalization=obs_normalization,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def test_teacher_student_value_and_velocity_paths_have_expected_shapes() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    teacher_actions = model.act_teacher(obs, stochastic_output=True)
    student_actions = model(obs)
    values = model.evaluate_teacher(obs)
    privileged_latent = model.get_privileged_latent(obs)
    proprio_latent, predicted_lin_vel = model.get_proprio_outputs(obs)

    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert student_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert values.shape == (NUM_ENVS, 1)
    assert privileged_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert proprio_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert predicted_lin_vel.shape == (NUM_ENVS, LIN_VEL_DIM)


def test_current_proprio_is_latest_frame_of_history() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    assert torch.equal(model.get_current_proprio(obs), obs["proprio_history"][:, -1, :])
    assert torch.equal(model.get_proprio_obs(obs), obs["proprio_history"].flatten(start_dim=1))


def test_teacher_actor_uses_predicted_velocity_and_privileged_latent() -> None:
    obs = make_rep_obs()
    obs["lin_vel_target"] = torch.full((NUM_ENVS, LIN_VEL_DIM), 100.0)
    obs["proprio_history"][:, -1, :] = torch.arange(
        NUM_ENVS * PROPRIO_DIM,
        dtype=obs["proprio_history"].dtype,
    ).reshape(NUM_ENVS, PROPRIO_DIM)
    model = make_model(obs)
    captured: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        expected_predicted_lin_vel = model.get_predicted_lin_vel(obs)

    def capture_actor(actor_obs: torch.Tensor, latent: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        captured["actor_obs"] = actor_obs.detach().clone()
        captured["latent"] = latent.detach().clone()
        captured["stochastic_output"] = torch.tensor(stochastic_output)
        return torch.zeros(NUM_ENVS, NUM_ACTIONS)

    model._actor = capture_actor  # type: ignore[method-assign]

    actions = model.act_teacher(obs, stochastic_output=True)

    lin_vel_slice = captured["actor_obs"][:, :LIN_VEL_DIM]
    proprio_slice = captured["actor_obs"][:, LIN_VEL_DIM : LIN_VEL_DIM + PROPRIO_DIM]
    command_slice = captured["actor_obs"][:, LIN_VEL_DIM + PROPRIO_DIM :]
    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert torch.equal(lin_vel_slice, expected_predicted_lin_vel)
    assert not torch.equal(lin_vel_slice, obs["lin_vel_target"])
    assert torch.equal(proprio_slice, obs["proprio_history"][:, -1, :])
    assert torch.equal(command_slice, obs["actor_command"])
    assert torch.equal(captured["latent"], model.get_privileged_latent(obs))
    assert captured["stochastic_output"].item() is True


def test_student_losses_only_update_student_encoder_side() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses(obs)
    model.zero_grad()
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.proprio_encoder.parameters())
    assert any(param.grad is not None for param in model.student_latent_head.parameters())
    assert any(param.grad is not None for param in model.lin_vel_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_normalization_update_supports_velocity_inputs() -> None:
    obs = make_rep_obs()
    model = make_model(obs, obs_normalization=True)

    model.update_normalization(obs)
    teacher_actions = model.act_teacher(obs)
    student_actions = model(obs)

    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert student_actions.shape == (NUM_ENVS, NUM_ACTIONS)


def test_student_inference_only_requires_history_and_command() -> None:
    model = make_model(make_rep_obs())
    inference_obs = make_rep_obs(include_privileged=False)

    actions = model(inference_obs)

    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)


def test_exported_student_policy_uses_history_and_command_inputs() -> None:
    model = make_model(make_rep_obs())
    proprio_history = torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM)
    actor_command = torch.randn(NUM_ENVS, COMMAND_DIM)
    inference_obs = TensorDict(
        {"proprio_history": proprio_history, "actor_command": actor_command},
        batch_size=[NUM_ENVS],
    )
    expected_actions = model(inference_obs)
    expected_predicted_lin_vel = model.get_predicted_lin_vel(inference_obs)

    jit_policy = model.as_jit()
    onnx_policy = model.as_onnx(verbose=False)
    jit_actions, jit_predicted_lin_vel = jit_policy(proprio_history, actor_command)
    onnx_actions, onnx_predicted_lin_vel = onnx_policy(proprio_history, actor_command)

    assert torch.allclose(jit_actions, expected_actions)
    assert torch.allclose(jit_predicted_lin_vel, expected_predicted_lin_vel)
    assert torch.allclose(onnx_actions, expected_actions)
    assert torch.allclose(onnx_predicted_lin_vel, expected_predicted_lin_vel)
    assert onnx_policy.input_names == ["proprio_history", "actor_command"]
    assert onnx_policy.output_names == ["actions", "predicted_lin_vel"]
    assert onnx_policy.get_dummy_inputs()[0].shape == (1, HISTORY_LENGTH, PROPRIO_DIM)
    assert onnx_policy.get_dummy_inputs()[1].shape == (1, COMMAND_DIM)
