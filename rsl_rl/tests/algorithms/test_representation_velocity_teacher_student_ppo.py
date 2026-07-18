# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: ANN001, ANN002, ANN003, ANN202, D103

"""Tests for velocity representation teacher-student PPO."""

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
        latent_dynamics_horizons=(1, 5),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def build_algorithm() -> tuple[RepresentationVelocityTeacherStudentPPO, TensorDict]:
    torch.manual_seed(11)
    obs = make_rep_obs()
    model = make_model(obs)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = RepresentationVelocityTeacherStudentPPO(
        model,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        student_learning_rate=1.0e-3,
        schedule="fixed",
        desired_kl=0.01,
    )
    return alg, obs


def build_depth_algorithm() -> tuple[RepresentationVelocityTeacherStudentPPO, TensorDict]:
    torch.manual_seed(12)
    obs = make_depth_rep_obs()
    model = make_depth_model(obs)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    alg = RepresentationVelocityTeacherStudentPPO(
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
        latent_dynamics_loss_coef=1.0,
        latent_dynamics_horizons=(1, 5),
        latent_dynamics_horizon_weights=(1.0, 0.5),
        latent_rollout_horizon=5,
        latent_rollout_loss_coef=0.5,
        num_latent_dynamics_epochs=1,
        num_latent_dynamics_mini_batches=2,
    )
    return alg, obs


def fill_rollout(alg: RepresentationVelocityTeacherStudentPPO, obs: TensorDict) -> TensorDict:
    for _ in range(NUM_STEPS):
        alg.act(obs)
        next_obs = make_depth_rep_obs() if "depth_camera" in obs else make_rep_obs()
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        alg.process_env_step(
            next_obs,
            rewards,
            dones,
            {"applied_actions": torch.randn(NUM_ENVS, NUM_ACTIONS)},
        )
        obs = next_obs
    alg.compute_returns(obs)
    return obs


def any_param_changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(before[name], param) for name, param in module.named_parameters())


@pytest.mark.parametrize(
    ("baseline", "delta", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([1.0, 0.0], [0.0, 2.0], 0.0),
        ([1.0, 0.0], [0.0, 0.0], 0.0),
    ],
)
def test_gradient_delta_cosine(baseline: list[float], delta: list[float], expected: float) -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    baseline_tensor = torch.tensor(baseline)
    parameter.grad = baseline_tensor + torch.tensor(delta)

    cosine = RepresentationVelocityTeacherStudentPPO._gradient_delta_cosine(
        [parameter],
        {id(parameter): baseline_tensor},
    )

    assert cosine == pytest.approx(expected)


def test_gradient_delta_cosine_includes_dynamics_only_parameters() -> None:
    shared_parameter = torch.nn.Parameter(torch.zeros(2))
    dynamics_only_parameter = torch.nn.Parameter(torch.zeros(2))
    shared_baseline = torch.tensor([1.0, 0.0])
    shared_parameter.grad = torch.tensor([2.0, 0.0])
    dynamics_only_parameter.grad = torch.tensor([0.0, 1.0])

    cosine = RepresentationVelocityTeacherStudentPPO._gradient_delta_cosine(
        [shared_parameter, dynamics_only_parameter],
        {
            id(shared_parameter): shared_baseline,
            id(dynamics_only_parameter): None,
        },
    )

    assert cosine == pytest.approx(1.0 / 2.0**0.5)


def test_update_returns_student_losses_and_updates_parameter_groups() -> None:
    alg, obs = build_algorithm()
    fill_rollout(alg, obs)

    actor_before = {n: p.detach().clone() for n, p in alg.actor.actor_head.named_parameters()}
    critic_before = {n: p.detach().clone() for n, p in alg.actor.critic_head.named_parameters()}
    privileged_before = {n: p.detach().clone() for n, p in alg.actor.privileged_encoder.named_parameters()}
    proprio_before = {n: p.detach().clone() for n, p in alg.actor.proprio_encoder.named_parameters()}
    latent_head_before = {n: p.detach().clone() for n, p in alg.actor.student_latent_head.named_parameters()}
    lin_vel_head_before = {n: p.detach().clone() for n, p in alg.actor.lin_vel_head.named_parameters()}

    losses = alg.update()

    assert {"value", "surrogate", "entropy", "student", "representation", "lin_vel"} <= set(losses)
    assert any_param_changed(actor_before, alg.actor.actor_head)
    assert any_param_changed(critic_before, alg.actor.critic_head)
    assert any_param_changed(privileged_before, alg.actor.privileged_encoder)
    assert any_param_changed(proprio_before, alg.actor.proprio_encoder)
    assert any_param_changed(latent_head_before, alg.actor.student_latent_head)
    assert any_param_changed(lin_vel_head_before, alg.actor.lin_vel_head)


def test_student_update_runs_after_all_ppo_minibatches() -> None:
    alg, obs = build_algorithm()
    fill_rollout(alg, obs)
    expected_ppo_calls = alg.num_learning_epochs * alg.num_mini_batches
    calls = {"ppo": 0, "student": 0}
    original_act_teacher = alg.actor.act_teacher
    original_compute_student_losses = alg.actor.compute_student_losses

    def counted_act_teacher(*args, **kwargs):
        calls["ppo"] += 1
        return original_act_teacher(*args, **kwargs)

    def counted_compute_student_losses(*args, **kwargs):
        assert calls["ppo"] == expected_ppo_calls
        calls["student"] += 1
        return original_compute_student_losses(*args, **kwargs)

    alg.actor.act_teacher = counted_act_teacher  # type: ignore[method-assign]
    alg.actor.compute_student_losses = counted_compute_student_losses  # type: ignore[method-assign]

    alg.update()

    assert calls == {"ppo": expected_ppo_calls, "student": expected_ppo_calls}


def test_depth_student_update_uses_sequence_chunks() -> None:
    alg, obs = build_depth_algorithm()
    fill_rollout(alg, obs)
    calls = {"flat": 0, "sequence": 0}
    original_compute_student_losses = alg.actor.compute_student_losses
    original_compute_student_losses_sequence = alg.actor.compute_student_losses_sequence

    def counted_compute_student_losses(*args, **kwargs):
        calls["flat"] += 1
        return original_compute_student_losses(*args, **kwargs)

    def counted_compute_student_losses_sequence(*args, **kwargs):
        calls["sequence"] += 1
        return original_compute_student_losses_sequence(*args, **kwargs)

    alg.actor.compute_student_losses = counted_compute_student_losses  # type: ignore[method-assign]
    alg.actor.compute_student_losses_sequence = counted_compute_student_losses_sequence  # type: ignore[method-assign]

    losses = alg.update()

    assert {
        "student",
        "representation",
        "lin_vel",
        "Grad/ppo_total_norm",
        "Grad/privileged_encoder_ppo_norm",
        "Grad/ppo_clip_fraction",
        "Grad/combined_total_norm",
        "Grad/combined_clip_fraction",
        "Grad/dynamics_total_norm",
        "Grad/privileged_encoder_dynamics_norm",
        "Grad/privileged_encoder_dynamics_to_ppo_ratio",
        "Grad/privileged_encoder_ppo_dynamics_cosine",
        "Grad/dynamics_clip_fraction",
        "Update/privileged_encoder_norm_joint",
        "Update/privileged_encoder_norm_ppo_only",
        "Update/policy_kl_joint",
        "Update/policy_kl_ppo_only",
        "Update/joint_step_fraction",
        "latent_dynamics_loss",
        "latent_identity_loss",
        "latent_prediction_identity_ratio",
        "latent_shuffled_action_loss",
        "latent_shuffled_action_ratio",
        "latent_dynamics_valid_fraction",
        "latent_dynamics_loss_k1",
        "latent_dynamics_loss_k5",
        "latent_identity_loss_k1",
        "latent_identity_loss_k5",
        "latent_prediction_identity_ratio_k1",
        "latent_prediction_identity_ratio_k5",
        "latent_shuffled_action_loss_k1",
        "latent_shuffled_action_loss_k5",
        "latent_shuffled_action_ratio_k1",
        "latent_shuffled_action_ratio_k5",
        "latent_prediction_cosine_similarity_k1",
        "latent_prediction_cosine_similarity_k5",
        "latent_dynamics_valid_fraction_k1",
        "latent_dynamics_valid_fraction_k5",
        "latent_reversed_action_loss_k5",
        "latent_reversed_action_ratio_k5",
        "latent_rollout_loss",
        "latent_rollout_valid_fraction",
        "latent_direct_rollout_cosine_k5",
        "latent_direct_rollout_mse_k5",
    } <= set(losses)
    assert losses["Grad/privileged_encoder_ppo_norm"] > 0.0
    assert losses["Grad/privileged_encoder_dynamics_norm"] > 0.0
    assert losses["Grad/privileged_encoder_dynamics_to_ppo_ratio"] == pytest.approx(
        losses["Grad/privileged_encoder_dynamics_norm"] / (losses["Grad/privileged_encoder_ppo_norm"] + 1.0e-8)
    )
    assert -1.0 <= losses["Grad/privileged_encoder_ppo_dynamics_cosine"] <= 1.0
    assert 0.0 <= losses["Grad/ppo_clip_fraction"] <= 1.0
    assert 0.0 <= losses["Grad/combined_clip_fraction"] <= 1.0
    assert 0.0 <= losses["Grad/dynamics_clip_fraction"] <= 1.0
    assert losses["Update/privileged_encoder_norm_joint"] > 0.0
    assert losses["Update/privileged_encoder_norm_ppo_only"] > 0.0
    assert losses["Update/policy_kl_joint"] >= 0.0
    assert losses["Update/policy_kl_ppo_only"] >= 0.0
    assert losses["Update/joint_step_fraction"] == pytest.approx(0.5)
    for step in range(1, 6):
        assert {
            f"latent_rollout_loss_k{step}",
            f"latent_rollout_identity_ratio_k{step}",
            f"latent_rollout_shuffled_action_loss_k{step}",
            f"latent_rollout_shuffled_action_ratio_k{step}",
            f"latent_rollout_cosine_similarity_k{step}",
        } <= set(losses)
    assert calls["flat"] == 0
    assert calls["sequence"] > 0


def test_depth_dynamics_updates_predictor_and_records_applied_actions() -> None:
    alg, obs = build_depth_algorithm()
    alg.act(obs)

    assert alg.transition.applied_actions is None

    next_obs = make_depth_rep_obs()
    applied_actions = torch.tensor([[1.0, -1.0]]).repeat(NUM_ENVS, 1)
    alg.process_env_step(
        next_obs,
        torch.randn(NUM_ENVS),
        torch.zeros(NUM_ENVS),
        {"applied_actions": applied_actions},
    )
    assert alg.storage.applied_actions is not None
    assert torch.equal(alg.storage.applied_actions[0], applied_actions)

    for _ in range(1, NUM_STEPS):
        alg.act(next_obs)
        following_obs = make_depth_rep_obs()
        alg.process_env_step(
            following_obs,
            torch.randn(NUM_ENVS),
            torch.zeros(NUM_ENVS),
            {"applied_actions": torch.randn(NUM_ENVS, NUM_ACTIONS)},
        )
        next_obs = following_obs
    alg.compute_returns(next_obs)

    predictor_before = {
        horizon: {name: param.detach().clone() for name, param in predictor.named_parameters()}
        for horizon, predictor in alg.actor.latent_dynamics_predictors.items()
    }
    losses = alg.update()

    for horizon, predictor in alg.actor.latent_dynamics_predictors.items():
        assert any_param_changed(predictor_before[horizon], predictor)
    assert losses["latent_dynamics_valid_fraction"] == pytest.approx(1.0)
    assert losses["latent_dynamics_valid_fraction_k1"] == pytest.approx(1.0)
    assert losses["latent_dynamics_valid_fraction_k5"] == pytest.approx(1.0)
    assert losses["latent_dynamics_loss"] == pytest.approx(
        (losses["latent_dynamics_loss_k1"] + 0.5 * losses["latent_dynamics_loss_k5"]) / 1.5
    )


def test_depth_dynamics_uses_joint_optimizer_steps_only() -> None:
    alg, obs = build_depth_algorithm()
    fill_rollout(alg, obs)
    optimizer_steps = 0
    original_step = alg.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    alg.optimizer.step = counted_step

    alg.update()

    assert optimizer_steps == alg.num_learning_epochs * alg.num_mini_batches


def test_joint_dynamics_respects_detached_source_encoder() -> None:
    alg, obs = build_depth_algorithm()
    alg.latent_dynamics_detach_source = True
    fill_rollout(alg, obs)

    losses = alg.update()

    assert losses["Grad/privileged_encoder_dynamics_norm"] == pytest.approx(0.0)
    assert losses["Grad/privileged_encoder_ppo_dynamics_cosine"] == pytest.approx(0.0)
    assert losses["Grad/dynamics_total_norm"] > 0.0
    assert losses["Update/joint_step_fraction"] == pytest.approx(0.5)


def test_latent_dynamics_generator_filters_only_done_pairs() -> None:
    num_steps = 4
    num_envs = 2
    obs = TensorDict(
        {"state": torch.zeros(num_envs, 1)},
        batch_size=[num_envs],
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [1])
    state = torch.arange(num_steps * num_envs).view(num_steps, num_envs, 1).float()
    storage.observations["state"].copy_(state)
    storage.applied_actions = state.clone()
    storage.dones[0, 0] = 1

    batches = list(storage.latent_dynamics_mini_batch_generator(2, 1))
    pair_markers = {
        (int(current), int(future), int(action))
        for batch in batches
        for current, future, action in zip(
            batch.observations["state"],
            batch.next_observations["state"],
            batch.applied_actions,
            strict=True,
        )
    }

    assert pair_markers == {
        (1, 3, 1),
        (2, 4, 2),
        (3, 5, 3),
        (4, 6, 4),
        (5, 7, 5),
    }
    assert storage.latent_dynamics_valid_fraction == pytest.approx(5.0 / 6.0)
    assert storage.latent_dynamics_valid_fractions[1] == pytest.approx(5.0 / 6.0)


def test_latent_dynamics_generator_builds_horizon_action_block_and_masks_full_interval() -> None:
    num_steps = 8
    num_envs = 2
    obs = TensorDict(
        {"state": torch.zeros(num_envs, 1)},
        batch_size=[num_envs],
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [1])
    state = torch.arange(num_steps * num_envs).view(num_steps, num_envs, 1).float()
    storage.observations["state"].copy_(state)
    storage.applied_actions = state.clone()
    storage.dones[0, 0] = 1

    batches = list(storage.latent_dynamics_mini_batch_generator(2, 1, horizon=5))
    pair_markers = {
        (int(current), int(future), tuple(int(action) for action in action_block))
        for batch in batches
        for current, future, action_block in zip(
            batch.observations["state"],
            batch.next_observations["state"],
            batch.applied_actions,
            strict=True,
        )
    }

    assert pair_markers == {
        (1, 11, (1, 3, 5, 7, 9)),
        (2, 12, (2, 4, 6, 8, 10)),
        (3, 13, (3, 5, 7, 9, 11)),
        (4, 14, (4, 6, 8, 10, 12)),
        (5, 15, (5, 7, 9, 11, 13)),
    }
    assert storage.latent_dynamics_valid_fractions[5] == pytest.approx(5.0 / 6.0)


def test_latent_dynamics_sequence_generator_returns_all_ordered_steps() -> None:
    num_steps = 8
    num_envs = 2
    obs = TensorDict(
        {"state": torch.zeros(num_envs, 1)},
        batch_size=[num_envs],
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [1])
    state = torch.arange(num_steps * num_envs).view(num_steps, num_envs, 1).float()
    storage.observations["state"].copy_(state)
    storage.applied_actions = state.clone()
    storage.dones[0, 0] = 1

    batches = list(storage.latent_dynamics_sequence_mini_batch_generator(2, 1, rollout_horizon=5))
    sequence_markers = {
        (
            int(batch.observations["state"][sample]),
            tuple(int(value) for value in batch.future_observations["state"][:, sample]),
            tuple(int(value) for value in batch.applied_actions[:, sample]),
        )
        for batch in batches
        for sample in range(batch.observations.batch_size[0])
    }

    assert sequence_markers == {
        (1, (3, 5, 7, 9, 11), (1, 3, 5, 7, 9)),
        (2, (4, 6, 8, 10, 12), (2, 4, 6, 8, 10)),
        (3, (5, 7, 9, 11, 13), (3, 5, 7, 9, 11)),
        (4, (6, 8, 10, 12, 14), (4, 6, 8, 10, 12)),
        (5, (7, 9, 11, 13, 15), (5, 7, 9, 11, 13)),
    }
    assert storage.latent_dynamics_sequence_valid_fraction == pytest.approx(5.0 / 6.0)


def test_latent_rollout_keeps_recursive_gradient_chain() -> None:
    model = make_depth_model(make_depth_rep_obs())
    latent = torch.randn(NUM_ENVS, model.latent_dim, requires_grad=True)
    applied_actions = torch.randn(5, NUM_ENVS, NUM_ACTIONS)

    rollout = model.rollout_privileged_latent(latent, applied_actions)
    rollout[-1, :, 0].sum().backward()

    assert rollout.shape == (5, NUM_ENVS, model.latent_dim)
    assert latent.grad is not None and torch.count_nonzero(latent.grad) > 0
    assert any(
        param.grad is not None and torch.count_nonzero(param.grad) > 0
        for param in model.latent_dynamics_predictors["1"].parameters()
    )
    assert all(param.grad is None for param in model.latent_dynamics_predictors["5"].parameters())


def test_student_loss_detaches_privileged_encoder_target() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    student_loss, _, _ = model.compute_student_losses(obs)
    model.zero_grad()
    student_loss.backward()

    assert any(param.grad is not None for param in model.proprio_encoder.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_multi_gpu_update_reduces_ppo_and_student_gradients() -> None:
    alg, obs = build_algorithm()
    fill_rollout(alg, obs)
    alg.is_multi_gpu = True

    ppo_param_ids = {id(param) for param in alg.actor.ppo_parameters()}
    student_param_ids = {id(param) for param in alg.actor.student_parameters()}
    reduced = {"ppo": False, "student": False}

    def fake_reduce_parameters(parameters=None) -> None:
        param_ids = {id(param) for param in parameters}
        if param_ids <= ppo_param_ids:
            reduced["ppo"] = True
        if param_ids <= student_param_ids:
            reduced["student"] = True

    alg.reduce_parameters = fake_reduce_parameters

    alg.update()

    assert reduced == {"ppo": True, "student": True}


def test_multi_gpu_depth_update_reduces_latent_dynamics_gradients() -> None:
    alg, obs = build_depth_algorithm()
    fill_rollout(alg, obs)
    alg.is_multi_gpu = True

    dynamics_param_ids = {id(param) for param in alg.actor.latent_dynamics_parameters()}
    dynamics_reduced = False

    def fake_reduce_parameters(parameters=None) -> None:
        nonlocal dynamics_reduced
        param_ids = {id(param) for param in parameters}
        if param_ids == dynamics_param_ids:
            dynamics_reduced = True

    alg.reduce_parameters = fake_reduce_parameters

    alg.update()

    assert dynamics_reduced


def test_save_includes_student_optimizer() -> None:
    alg, _ = build_algorithm()

    saved = alg.save()

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
