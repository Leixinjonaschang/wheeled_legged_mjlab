# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: D103

"""Tests for the velocity representation actor-critic model."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import DepthRepresentationVelocityActorCritic, RepresentationVelocityActorCritic
from rsl_rl.models.depth_representation_velocity_predictor_actor_critic import (
    DepthRepresentationVelocityPredictorActorCritic,
)

NUM_ENVS = 4
PROPRIO_DIM = 28
COMMAND_DIM = 3
LIN_VEL_DIM = 3
CRITIC_DIM = 16
PRIVILEGED_DIM = 11
HISTORY_LENGTH = 5
LATENT_DIM = 4
NUM_ACTIONS = 2
DEPTH_SHAPE = (1, 32, 24)


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


def make_depth_rep_obs(include_privileged: bool = True, include_depth: bool = True) -> TensorDict:
    data = {
        "proprio_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM),
        "actor_command": torch.randn(NUM_ENVS, COMMAND_DIM),
    }
    if include_depth:
        data["depth_camera"] = torch.randn(NUM_ENVS, *DEPTH_SHAPE)
    if include_privileged:
        data.update(
            {
                "lin_vel_target": torch.randn(NUM_ENVS, LIN_VEL_DIM),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                "privileged_encoder": torch.randn(NUM_ENVS, PRIVILEGED_DIM),
            }
        )
    return TensorDict(data, batch_size=[NUM_ENVS])


def make_depth_model(obs: TensorDict | None = None) -> DepthRepresentationVelocityActorCritic:
    obs = make_depth_rep_obs() if obs is None else obs
    obs_groups = {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "critic": ["critic"],
        "privileged_encoder": ["privileged_encoder"],
        "depth_encoder": ["depth_camera"],
    }
    return DepthRepresentationVelocityActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def make_depth_predictor_model(
    obs: TensorDict | None = None,
) -> DepthRepresentationVelocityPredictorActorCritic:
    obs = make_depth_rep_obs() if obs is None else obs
    obs_groups = {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "critic": ["critic"],
        "privileged_encoder": ["privileged_encoder"],
        "depth_encoder": ["depth_camera"],
    }
    return DepthRepresentationVelocityPredictorActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        latent_dynamics_horizons=(1, 5),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def test_depth_model_has_no_latent_dynamics_interface() -> None:
    model = make_depth_model()

    assert not hasattr(model, "latent_dynamics_horizons")
    assert not hasattr(model, "latent_dynamics_predictors")
    assert not hasattr(model, "compute_latent_dynamics_loss")


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
    expected_proprio_obs = obs["proprio_history"].flatten(start_dim=1)
    assert torch.equal(model.get_proprio_obs(obs), expected_proprio_obs)


def test_encoder_outputs_do_not_depend_on_command() -> None:
    obs = make_rep_obs()
    model = make_model(obs)
    changed_command_obs = obs.clone()
    changed_command_obs["actor_command"] = torch.randn_like(obs["actor_command"])

    assert torch.equal(
        model.get_proprio_outputs(obs)[0],
        model.get_proprio_outputs(changed_command_obs)[0],
    )
    assert torch.equal(
        model.get_predicted_lin_vel(obs),
        model.get_predicted_lin_vel(changed_command_obs),
    )
    assert torch.equal(
        model.get_privileged_latent(obs),
        model.get_privileged_latent(changed_command_obs),
    )


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


def test_depth_teacher_student_value_and_velocity_paths_have_expected_shapes() -> None:
    obs = make_depth_rep_obs()
    model = make_depth_model(obs)

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


def test_depth_student_hidden_state_persists_and_resets_per_environment() -> None:
    obs = make_depth_rep_obs()
    model = make_depth_model(obs)

    model(obs)
    first_state = model.get_hidden_state().clone()
    model(obs)
    second_state = model.get_hidden_state().clone()
    assert not torch.equal(first_state, second_state)

    model.reset(torch.tensor([True, False, False, True]))
    reset_state = model.get_hidden_state()
    assert torch.equal(reset_state[[0, 3]], torch.zeros_like(reset_state[[0, 3]]))
    assert torch.equal(reset_state[[1, 2]], second_state[[1, 2]])

    model.reset()
    assert model.get_hidden_state() is None


def test_depth_student_losses_update_depth_and_student_encoder_side() -> None:
    obs = make_depth_rep_obs()
    model = make_depth_model(obs)

    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses(obs)
    model.zero_grad()
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.depth_encoder.parameters())
    assert any(param.grad is not None for param in model.depth_gru.parameters())
    assert any(param.grad is not None for param in model.proprio_encoder.parameters())
    assert any(param.grad is not None for param in model.student_latent_head.parameters())
    assert any(param.grad is not None for param in model.lin_vel_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_depth_latent_dynamics_prediction_is_normalized_and_has_isolated_gradients() -> None:
    obs_t = make_depth_rep_obs()
    obs_tp1 = make_depth_rep_obs()
    model = make_depth_predictor_model(obs_t)
    applied_action_block = torch.randn(NUM_ENVS, 5 * NUM_ACTIONS)

    latent_t = model.get_privileged_latent(obs_t)
    prediction = model.predict_privileged_latent(latent_t, applied_action_block, horizon=5)
    assert torch.allclose(prediction.norm(dim=-1), torch.ones(NUM_ENVS), atol=1.0e-6)
    assert set(model.latent_dynamics_predictors) == {"1", "5"}

    captured_latents: list[torch.Tensor] = []
    original_get_privileged_latent = model.get_privileged_latent

    def capture_latent(obs: TensorDict) -> torch.Tensor:
        latent = original_get_privileged_latent(obs)
        captured_latents.append(latent)
        return latent

    model.get_privileged_latent = capture_latent  # type: ignore[method-assign]
    model.zero_grad()
    dynamics_loss = model.compute_latent_dynamics_loss(
        obs_t,
        applied_action_block,
        obs_tp1,
        horizon=5,
    )
    dynamics_loss.backward()

    assert captured_latents[0].requires_grad
    assert not captured_latents[1].requires_grad
    assert all(param.grad is None for param in model.latent_dynamics_predictors["1"].parameters())
    assert any(param.grad is not None for param in model.latent_dynamics_predictors["5"].parameters())
    assert any(param.grad is not None for param in model.privileged_encoder.parameters())
    assert all(param.grad is None for param in model.actor_head.parameters())
    assert all(param.grad is None for param in model.critic_head.parameters())
    assert all(param.grad is None for param in model.depth_encoder.parameters())
    assert all(param.grad is None for param in model.depth_gru.parameters())
    assert all(param.grad is None for param in model.proprio_encoder.parameters())
    assert all(param.grad is None for param in model.student_latent_head.parameters())
    assert all(param.grad is None for param in model.lin_vel_head.parameters())


def test_depth_ppo_path_does_not_backpropagate_into_dynamics_predictor() -> None:
    obs = make_depth_rep_obs()
    model = make_depth_predictor_model(obs)

    model.zero_grad()
    actions = model.act_teacher(obs, stochastic_output=True)
    ppo_path_loss = actions.square().mean() + model.evaluate_teacher(obs).square().mean()
    ppo_path_loss.backward()

    assert any(param.grad is not None for param in model.privileged_encoder.parameters())
    assert all(param.grad is None for param in model.latent_dynamics_predictors.parameters())


def test_depth_latent_dynamics_can_detach_source_for_representation_ablation() -> None:
    obs_t = make_depth_rep_obs()
    obs_future = make_depth_rep_obs()
    model = make_depth_predictor_model(obs_t)
    applied_action_block = torch.randn(NUM_ENVS, 5 * NUM_ACTIONS)

    model.zero_grad()
    loss = model.compute_latent_dynamics_loss(
        obs_t,
        applied_action_block,
        obs_future,
        horizon=5,
        detach_source=True,
    )
    loss.backward()

    assert all(param.grad is None for param in model.privileged_encoder.parameters())
    assert any(param.grad is not None for param in model.latent_dynamics_predictors["5"].parameters())


def test_depth_sequence_student_losses_use_continuous_depth_state() -> None:
    model = make_depth_model()
    time_steps = 3
    step_observations = [make_depth_rep_obs() for _ in range(time_steps)]
    obs = TensorDict(
        {
            key: torch.stack([step_obs[key] for step_obs in step_observations])
            for key in step_observations[0].keys()
        },
        batch_size=[time_steps, NUM_ENVS],
    )
    dones = torch.zeros(time_steps, NUM_ENVS, 1)
    dones[1, 0] = 1.0
    hidden_state = torch.zeros(NUM_ENVS, model.depth_gru_hidden_dim)

    model.zero_grad()
    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses_sequence(
        obs,
        dones,
        hidden_state,
    )
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.depth_gru.parameters())
    assert all(param.grad is None for param in model.actor_head.parameters())
    assert all(param.grad is None for param in model.critic_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_depth_student_inference_requires_depth_observations() -> None:
    model = make_depth_model()
    inference_obs = make_depth_rep_obs(include_privileged=False, include_depth=False)

    try:
        model(inference_obs)
    except ValueError as exc:
        assert "depth observation group" in str(exc)
    else:
        raise AssertionError("student inference without depth should fail")


def test_depth_onnx_wrapper_matches_policy_outputs() -> None:
    model = make_depth_model()
    model.eval()
    obs = make_depth_rep_obs()
    hidden_state = torch.zeros(NUM_ENVS, model.depth_gru_hidden_dim)
    onnx_model = model.as_onnx(verbose=False)
    onnx_model.eval()

    with torch.inference_mode():
        expected_actions = model(obs, hidden_state=hidden_state)
        expected_predicted_lin_vel = model.get_proprio_outputs(obs, hidden_state=hidden_state)[1]
        actions, predicted_lin_vel, hidden_state_out = onnx_model(
            obs["proprio_history"],
            obs["actor_command"],
            obs["depth_camera"],
            hidden_state,
        )

    assert torch.allclose(actions, expected_actions, atol=1e-6)
    assert torch.allclose(predicted_lin_vel, expected_predicted_lin_vel, atol=1e-6)
    assert hidden_state_out.shape == hidden_state.shape
    assert onnx_model.input_names == ["proprio_history", "actor_command", "depth", "hidden_state_in"]
    assert onnx_model.output_names == ["actions", "predicted_lin_vel", "hidden_state_out"]
    assert onnx_model.get_dummy_inputs()[2].shape == (1, *DEPTH_SHAPE)


def test_depth_jit_wrapper_scripts_and_runs_single_robot_policy() -> None:
    model = make_depth_model()
    obs = make_depth_rep_obs()

    export_model = model.as_jit()
    assert all("latent_dynamics" not in name for name, _ in export_model.named_parameters())
    jit_model = torch.jit.script(export_model)
    actions, predicted_lin_vel = jit_model(
        obs["proprio_history"][:1],
        obs["actor_command"][:1],
        obs["depth_camera"][:1],
    )

    assert actions.shape == (1, NUM_ACTIONS)
    assert predicted_lin_vel.shape == (1, LIN_VEL_DIM)
