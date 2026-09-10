# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for direct depth-student asymmetric PPO."""

# ruff: file-ignore[undocumented-public-function, lowercase-imported-as-non-lowercase]

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Iterable
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms.depth_predictor_ppo import DepthPredictorPPO
from rsl_rl.models import DepthActor, MLPModel
from rsl_rl.models.latent_dynamics_predictor import LatentDynamicsPredictor
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 12
NUM_ACTIONS = 2
LATENT_DIM = 4
HISTORY_LENGTH = 5
PROPRIO_DIM = 6
DEPTH_SHAPE = (1, 16, 16)


def make_obs() -> TensorDict:
    """Create one CPU-sized observation batch."""
    return TensorDict(
        {
            "proprio_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM),
            "actor_command": torch.randn(NUM_ENVS, 3),
            "lin_vel_target": torch.randn(NUM_ENVS, 3),
            "depth_camera": 0.2 + 2.0 * torch.rand(NUM_ENVS, *DEPTH_SHAPE),
            "wheel_roughness": torch.rand(NUM_ENVS, 2),
            "critic": torch.randn(NUM_ENVS, 7),
            "dynamics_context": torch.randn(NUM_ENVS, 2),
        },
        batch_size=[NUM_ENVS],
    )


def obs_groups() -> dict[str, list[str]]:
    """Return the asymmetric observation routing used by the task."""
    return {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "depth_encoder": ["depth_camera"],
        "wheel_roughness": ["wheel_roughness"],
        "critic": ["critic", "dynamics_context"],
    }


def make_actor(
    obs: TensorDict | None = None,
    groups: dict[str, list[str]] | None = None,
) -> DepthActor:
    """Create a small depth actor."""
    obs = make_obs() if obs is None else obs
    return DepthActor(
        obs,
        obs_groups() if groups is None else groups,
        NUM_ACTIONS,
        hidden_dims=(16, 8),
        encoder_hidden_dims=(16, 8),
        latent_dim=LATENT_DIM,
        depth_feature_dim=4,
        depth_gru_hidden_dim=5,
        depth_channels=(2, 2),
        depth_conv_strides=(2, 1),
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def build_algorithm(*, roughness_loss_coef: float = 0.2) -> tuple[DepthPredictorPPO, TensorDict]:
    """Create the complete CPU-sized algorithm."""
    torch.manual_seed(7)
    obs = make_obs()
    actor = make_actor(obs)
    critic = MLPModel(obs, obs_groups(), "critic", 1, hidden_dims=(16, 8))
    predictor = LatentDynamicsPredictor(
        LATENT_DIM,
        3,
        NUM_ACTIONS,
        horizons=(1, 5, 10),
        hidden_dims=(16, 8),
    )
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    algorithm = DepthPredictorPPO(
        actor,
        critic,
        predictor,
        storage,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_auxiliary_epochs=1,
        num_auxiliary_mini_batches=4,
        learning_rate=1.0e-3,
        predictor_learning_rate=1.0e-3,
        roughness_loss_coef=roughness_loss_coef,
        schedule="fixed",
    )
    return algorithm, obs


def stack_obs(observations: list[TensorDict]) -> TensorDict:
    """Stack observations into a time-major TensorDict."""
    return torch.stack(observations, dim=0)


def fill_rollout(
    algorithm: DepthPredictorPPO,
    obs: TensorDict,
    *,
    all_done: bool = False,
) -> TensorDict:
    """Collect a complete synthetic rollout and compute GAE."""
    for _ in range(NUM_STEPS):
        sampled_actions = algorithm.act(obs)
        next_obs = make_obs()
        dones = torch.ones(NUM_ENVS) if all_done else torch.zeros(NUM_ENVS)
        algorithm.process_env_step(
            next_obs,
            torch.randn(NUM_ENVS),
            dones,
            {"applied_actions": sampled_actions + 0.25},
        )
        obs = next_obs
    algorithm.compute_returns(obs)
    return obs


def has_nonzero_grad(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """Return whether any parameter has a nonzero gradient."""
    return any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)


def has_no_grad(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """Return whether every parameter has no gradient."""
    return all(parameter.grad is None for parameter in parameters)


def zero_modules(*modules: torch.nn.Module) -> None:
    """Clear gradients for a collection of modules."""
    for module in modules:
        module.zero_grad(set_to_none=True)


def test_structure_and_deployment_observation_isolation() -> None:
    algorithm, obs = build_algorithm()
    actor = algorithm.actor

    assert not hasattr(actor, "privileged_encoder")
    assert not hasattr(actor, "act_teacher")
    assert algorithm.critic.obs_groups == ["critic", "dynamics_context"]
    assert algorithm.predictor not in list(actor.modules())

    deployment_obs = TensorDict(
        {key: obs[key] for key in ("proprio_history", "actor_command", "depth_camera")},
        batch_size=obs.batch_size,
    )
    actions = actor(deployment_obs)
    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    deployment_actor = make_actor(
        deployment_obs,
        {
            "proprio_history": ["proprio_history"],
            "actor_command": ["actor_command"],
            "depth_encoder": ["depth_camera"],
            "critic": ["actor_command"],
        },
    )
    deployment_actor.load_state_dict(actor.state_dict(), strict=True)
    actor.reset()
    deployment_actor.reset()
    torch.testing.assert_close(actor(deployment_obs), deployment_actor(deployment_obs))


def test_each_loss_reaches_only_its_owned_modules() -> None:
    algorithm, _ = build_algorithm()
    actor, critic, predictor = (
        algorithm.actor,
        algorithm.critic,
        algorithm.predictor,
    )
    sequence = stack_obs([make_obs() for _ in range(NUM_STEPS)])
    dones = torch.zeros(NUM_STEPS, NUM_ENVS, 1)
    hidden = torch.zeros(NUM_ENVS, actor.depth_gru_hidden_dim)

    zero_modules(actor, critic, predictor)
    actions, _, _, _, _ = actor.forward_sequence(sequence, dones, hidden, stochastic_output=True)
    policy_loss = -actor.get_output_log_prob(actions.detach()).mean()
    policy_loss = policy_loss - 0.01 * actor.output_entropy.mean()
    policy_loss.backward()
    assert has_nonzero_grad(actor.depth_encoder.parameters())
    assert has_nonzero_grad(actor.depth_gru.parameters())
    assert has_nonzero_grad(actor.student_encoder.parameters())
    assert has_nonzero_grad(actor.latent_head.parameters())
    assert has_nonzero_grad(actor.lin_vel_head.parameters())
    assert has_nonzero_grad(actor.actor_head.parameters())
    assert has_nonzero_grad(actor.distribution.parameters())
    assert has_no_grad(actor.roughness_head.parameters())
    assert has_no_grad(critic.parameters())
    assert has_no_grad(predictor.parameters())

    zero_modules(actor, critic, predictor)
    critic(sequence).square().mean().backward()
    assert has_nonzero_grad(critic.parameters())
    assert has_no_grad(actor.parameters())
    assert has_no_grad(predictor.parameters())

    zero_modules(actor, critic, predictor)
    _, predicted_velocity, predicted_roughness, _ = actor.get_student_outputs_sequence(sequence, dones, hidden)
    F.mse_loss(predicted_velocity, actor.get_lin_vel_target(sequence)).backward()
    assert has_nonzero_grad(actor.depth_encoder.parameters())
    assert has_nonzero_grad(actor.depth_gru.parameters())
    assert has_nonzero_grad(actor.student_encoder.parameters())
    assert has_nonzero_grad(actor.lin_vel_head.parameters())
    assert has_no_grad(actor.latent_head.parameters())
    assert has_no_grad(actor.roughness_head.parameters())
    assert has_no_grad(actor.actor_head.parameters())
    assert has_no_grad(critic.parameters())
    assert has_no_grad(predictor.parameters())

    zero_modules(actor, critic, predictor)
    _, _, predicted_roughness, _ = actor.get_student_outputs_sequence(sequence, dones, hidden)
    F.smooth_l1_loss(predicted_roughness, actor.get_wheel_roughness(sequence)).backward()
    assert has_nonzero_grad(actor.depth_encoder.parameters())
    assert has_nonzero_grad(actor.depth_gru.parameters())
    assert has_nonzero_grad(actor.student_encoder.parameters())
    assert has_nonzero_grad(actor.roughness_head.parameters())
    assert has_no_grad(actor.latent_head.parameters())
    assert has_no_grad(actor.lin_vel_head.parameters())
    assert has_no_grad(actor.actor_head.parameters())


def test_dynamics_source_and_future_target_gradient_boundary() -> None:
    algorithm, _ = build_algorithm()
    actor, critic, predictor = (
        algorithm.actor,
        algorithm.critic,
        algorithm.predictor,
    )
    sequence = stack_obs([make_obs() for _ in range(NUM_STEPS)])
    dones = torch.zeros(NUM_STEPS, NUM_ENVS, 1)
    latent, _, _, _ = actor.get_student_outputs_sequence(
        sequence,
        dones,
        torch.zeros(NUM_ENVS, actor.depth_gru_hidden_dim),
    )
    latent.retain_grad()
    source = latent[0]
    future_target = latent[10].detach()
    source_velocity = actor.get_normalized_lin_vel_target(sequence)[0]
    future_velocity = actor.get_normalized_lin_vel_target(sequence)[10]
    applied_actions = torch.randn(NUM_ENVS, 10 * NUM_ACTIONS)

    zero_modules(actor, critic, predictor)
    predicted_latent, predicted_velocity = predictor(source, source_velocity, applied_actions, horizon=10)
    loss = (1.0 - F.cosine_similarity(predicted_latent, future_target, dim=-1)).mean()
    loss = loss + F.smooth_l1_loss(predicted_velocity, future_velocity)
    loss.backward()

    assert torch.count_nonzero(latent.grad[0]) > 0
    assert torch.count_nonzero(latent.grad[10]) == 0
    assert has_nonzero_grad(actor.depth_encoder.parameters())
    assert has_nonzero_grad(actor.depth_gru.parameters())
    assert has_nonzero_grad(actor.student_encoder.parameters())
    assert has_nonzero_grad(actor.latent_head.parameters())
    assert has_nonzero_grad(predictor.predictors["10"].parameters())
    assert has_no_grad(predictor.predictors["1"].parameters())
    assert has_no_grad(predictor.predictors["5"].parameters())
    assert has_no_grad(actor.lin_vel_head.parameters())
    assert has_no_grad(actor.roughness_head.parameters())
    assert has_no_grad(actor.actor_head.parameters())
    assert has_no_grad(critic.parameters())


def test_sequence_replay_matches_online_steps_and_resets_after_done() -> None:
    actor = make_actor()
    observations = [make_obs() for _ in range(6)]
    sequence = stack_obs(observations)
    dones = torch.zeros(6, NUM_ENVS, 1)
    dones[2, 0] = 1
    initial_hidden = torch.randn(NUM_ENVS, actor.depth_gru_hidden_dim)
    actor.reset(hidden_state=initial_hidden)

    online_actions = []
    online_velocities = []
    for step, step_obs in enumerate(observations):
        state_before = actor.get_hidden_state().clone()
        online_velocities.append(actor.get_predicted_lin_vel(step_obs))
        torch.testing.assert_close(actor.get_hidden_state(), state_before)
        online_actions.append(actor(step_obs))
        actor.reset(dones[step])
    online_final_hidden = actor.get_hidden_state().clone()

    replay_actions, _, replay_velocities, _, replay_final_hidden = actor.forward_sequence(
        sequence, dones, initial_hidden
    )
    torch.testing.assert_close(replay_actions, torch.stack(online_actions))
    torch.testing.assert_close(replay_velocities, torch.stack(online_velocities))
    torch.testing.assert_close(replay_final_hidden, online_final_hidden)


def test_done_boundary_cuts_recurrent_gradient_and_initial_state_gradient() -> None:
    actor = make_actor()
    sequence = stack_obs([make_obs() for _ in range(4)])
    depth = sequence["depth_camera"].detach().requires_grad_()
    sequence["depth_camera"] = depth
    dones = torch.zeros(4, NUM_ENVS, 1)
    dones[1] = 1
    initial_hidden = torch.randn(NUM_ENVS, actor.depth_gru_hidden_dim, requires_grad=True)

    actions, _, _, _, _ = actor.forward_sequence(sequence, dones, initial_hidden)
    actions[2].square().mean().backward()

    assert initial_hidden.grad is None
    assert torch.count_nonzero(depth.grad[:2]) == 0
    assert torch.count_nonzero(depth.grad[2]) > 0


def test_predictor_preserves_action_order_and_recursive_gradient_chain() -> None:
    predictor = LatentDynamicsPredictor(
        LATENT_DIM,
        3,
        NUM_ACTIONS,
        horizons=(1, 5),
        hidden_dims=(8,),
    )
    latent = torch.randn(NUM_ENVS, LATENT_DIM, requires_grad=True)
    velocity = torch.randn(NUM_ENVS, 3, requires_grad=True)
    actions = torch.arange(5 * NUM_ENVS * NUM_ACTIONS, dtype=torch.float32).reshape(5, NUM_ENVS, NUM_ACTIONS)
    captured_inputs: list[torch.Tensor] = []
    hook = predictor.predictors["5"].register_forward_pre_hook(
        lambda _module, args: captured_inputs.append(args[0].detach())
    )

    predictor(latent, velocity, actions.permute(1, 0, 2), horizon=5)
    hook.remove()
    torch.testing.assert_close(
        captured_inputs[0][..., -5 * NUM_ACTIONS :],
        actions.permute(1, 0, 2).flatten(start_dim=-2),
    )

    rollout_latent, rollout_velocity = predictor.rollout(latent, velocity, actions)
    (rollout_latent[-1].square().mean() + rollout_velocity[-1].square().mean()).backward()
    assert has_nonzero_grad(predictor.predictors["1"].parameters())
    assert has_no_grad(predictor.predictors["5"].parameters())
    assert latent.grad is not None and torch.count_nonzero(latent.grad) > 0
    assert velocity.grad is not None and torch.count_nonzero(velocity.grad) > 0


def test_prediction_masks_episode_boundaries_and_rollout_edges() -> None:
    algorithm, _ = build_algorithm()
    latent = F.normalize(torch.randn(NUM_STEPS, NUM_ENVS, LATENT_DIM), dim=-1)
    normalized_velocity = torch.randn(NUM_STEPS, NUM_ENVS, 3)
    applied_actions = torch.randn(NUM_STEPS, NUM_ENVS, NUM_ACTIONS)
    dones = torch.zeros(NUM_STEPS, NUM_ENVS, 1)
    dones[2, 0] = 1

    _, dynamics_valid, metrics = algorithm._compute_dynamics_objective(
        latent,
        normalized_velocity,
        applied_actions,
        dones,
    )

    assert dynamics_valid
    assert metrics["latent_dynamics_valid_fraction_k1"] == pytest.approx(43 / 44)
    assert metrics["latent_dynamics_valid_fraction_k5"] == pytest.approx(25 / 28)
    assert metrics["latent_dynamics_valid_fraction_k10"] == pytest.approx(6 / 8)
    assert metrics["latent_rollout_valid_fraction"] == pytest.approx(25 / 28)


def test_storage_sequence_batches_preserve_time_and_applied_action_order() -> None:
    obs = make_obs()
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    for step in range(NUM_STEPS):
        transition = RolloutStorage.Transition()
        step_obs = obs.clone()
        step_obs["critic"][:, 0] = step
        transition.observations = step_obs
        transition.actions = torch.zeros(NUM_ENVS, NUM_ACTIONS)
        transition.applied_actions = torch.stack([
            torch.full((NUM_ACTIONS,), 100.0 * step + env) for env in range(NUM_ENVS)
        ])
        transition.rewards = torch.zeros(NUM_ENVS)
        transition.dones = torch.zeros(NUM_ENVS)
        transition.values = torch.zeros(NUM_ENVS, 1)
        transition.actions_log_prob = torch.zeros(NUM_ENVS)
        transition.distribution_params = (
            torch.zeros(NUM_ENVS, NUM_ACTIONS),
            torch.ones(NUM_ENVS, NUM_ACTIONS),
        )
        transition.hidden_states = (
            torch.full((NUM_ENVS, 5), float(step)),
            None,
        )
        storage.add_transition(transition)

    batches = list(storage.sequence_mini_batch_generator(4, 1))
    assert len(batches) == 4
    seen_envs = set()
    for batch in batches:
        assert batch.observations.batch_size == torch.Size([NUM_STEPS, 1])
        torch.testing.assert_close(
            batch.observations["critic"][:, 0, 0],
            torch.arange(NUM_STEPS, dtype=torch.float32),
        )
        env_id = int(batch.applied_actions[0, 0, 0].item())
        seen_envs.add(env_id)
        torch.testing.assert_close(
            batch.applied_actions[:, 0, 0],
            100.0 * torch.arange(NUM_STEPS) + env_id,
        )
        torch.testing.assert_close(batch.hidden_states[0], torch.zeros(1, 5))
    assert seen_envs == set(range(NUM_ENVS))


def test_update_counts_normalization_checkpoint_and_actor_only_inference() -> None:
    algorithm, obs = build_algorithm()
    actor_normalizer = algorithm.actor.proprio_history_obs_normalizer
    count_before = actor_normalizer.count.item()
    fill_rollout(algorithm, obs)
    assert actor_normalizer.count.item() == count_before
    assert not torch.equal(algorithm.storage.actions, algorithm.storage.applied_actions)

    losses = algorithm.update()
    assert algorithm.last_num_ppo_updates == 20
    assert algorithm.last_num_auxiliary_updates == 4
    assert algorithm.last_num_predictor_updates == 4
    assert actor_normalizer.count.item() == count_before + NUM_STEPS * NUM_ENVS
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert {
        "lin_vel",
        "roughness",
        "latent_variance",
        "latent_dynamics_loss_k1",
        "latent_dynamics_loss_k5",
        "latent_dynamics_loss_k10",
        "latent_prediction_identity_ratio_k10",
        "latent_shuffled_action_ratio_k10",
        "latent_reversed_action_ratio_k10",
        "latent_rollout_loss",
        "latent_rollout_identity_ratio_k5",
        "latent_rollout_shuffled_action_ratio_k5",
        "Grad/student_backbone_ppo_norm",
        "Grad/student_backbone_prediction_norm",
        "Grad/student_backbone_ppo_prediction_cosine",
    } <= losses.keys()

    checkpoint = algorithm.save()
    assert set(checkpoint) == {
        "actor_state_dict",
        "critic_state_dict",
        "predictor_state_dict",
        "optimizer_state_dict",
        "predictor_optimizer_state_dict",
        "learning_rate",
        "predictor_learning_rate",
    }
    loaded, deployment_obs = build_algorithm()
    loaded.load(checkpoint, load_cfg=None, strict=True)
    loaded.eval_mode()
    algorithm.eval_mode()
    algorithm.actor.reset()
    loaded.actor.reset()
    torch.testing.assert_close(
        algorithm.get_policy()(deployment_obs),
        loaded.get_policy()(deployment_obs),
    )
    assert loaded.get_policy() is loaded.actor

    predictor_before_resume = {
        name: parameter.detach().clone() for name, parameter in loaded.predictor.named_parameters()
    }
    loaded.train_mode()
    fill_rollout(loaded, deployment_obs)
    resumed_losses = loaded.update()
    assert loaded.last_num_ppo_updates == 20
    assert loaded.last_num_predictor_updates == 4
    assert all(torch.isfinite(torch.tensor(value)) for value in resumed_losses.values())
    assert any(
        not torch.equal(parameter, predictor_before_resume[name])
        for name, parameter in loaded.predictor.named_parameters()
    )


def test_optimizers_are_disjoint_and_adaptive_kl_changes_only_main_lr() -> None:
    algorithm, obs = build_algorithm()
    main_ids = {id(parameter) for group in algorithm.optimizer.param_groups for parameter in group["params"]}
    predictor_ids = {
        id(parameter) for group in algorithm.predictor_optimizer.param_groups for parameter in group["params"]
    }
    expected_main_ids = {
        id(parameter) for module in (algorithm.actor, algorithm.critic) for parameter in module.parameters()
    }
    assert main_ids == expected_main_ids
    assert predictor_ids == {id(parameter) for parameter in algorithm.predictor.parameters()}
    assert main_ids.isdisjoint(predictor_ids)

    algorithm.schedule = "adaptive"
    algorithm.desired_kl = 1.0e-4
    algorithm.actor(obs, stochastic_output=True)
    new_mean, new_std = algorithm.actor.output_distribution_params
    old_params = (new_mean.detach() + 10.0, new_std.detach())
    new_params = (new_mean.detach(), new_std.detach())
    predictor_lr = algorithm.predictor_optimizer.param_groups[0]["lr"]

    algorithm._adapt_learning_rate(old_params, new_params)

    assert algorithm.optimizer.param_groups[0]["lr"] < 1.0e-3
    assert algorithm.predictor_optimizer.param_groups[0]["lr"] == predictor_lr


def test_globally_empty_prediction_batch_skips_only_predictor() -> None:
    algorithm, obs = build_algorithm()
    predictor_before = {name: parameter.detach().clone() for name, parameter in algorithm.predictor.named_parameters()}
    fill_rollout(algorithm, obs, all_done=True)
    losses = algorithm.update()

    assert algorithm.last_num_ppo_updates == 20
    assert algorithm.last_num_auxiliary_updates == 4
    assert algorithm.last_num_predictor_updates == 0
    assert losses["latent_dynamics_valid_fraction"] == pytest.approx(0.0)
    assert losses["latent_rollout_valid_fraction"] == pytest.approx(0.0)
    for name, parameter in algorithm.predictor.named_parameters():
        torch.testing.assert_close(parameter, predictor_before[name])


def test_global_masked_mean_keeps_a_zero_sample_rank_in_the_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    algorithm, _ = build_algorithm()
    algorithm.is_multi_gpu = True
    algorithm.gpu_world_size = 2
    remote_contributions = iter((6.0, 3.0))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        del op
        tensor.add_(next(remote_contributions))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    values = torch.tensor([2.0, 4.0], requires_grad=True)
    local_mask = torch.zeros(2, dtype=torch.bool)

    loss, metric, global_count = algorithm._global_masked_mean(values, local_mask)
    loss.backward()

    assert global_count == 3
    assert metric == pytest.approx(2.0)
    torch.testing.assert_close(values.grad, torch.zeros_like(values))


def test_export_wrappers_match_online_actor_and_exclude_training_heads() -> None:
    actor = make_actor()
    obs = make_obs()[:1]
    hidden = torch.randn(1, actor.depth_gru_hidden_dim)
    actor.reset(hidden_state=hidden)
    expected_velocity = actor.get_predicted_lin_vel(obs)
    expected_actions = actor(obs)
    expected_hidden = actor.get_hidden_state().clone()

    onnx_actor = actor.as_onnx(verbose=False)
    actions, velocity, hidden_out = onnx_actor(
        obs["proprio_history"],
        obs["actor_command"],
        actor.depth_preprocessor(obs["depth_camera"]),
        hidden,
    )
    torch.testing.assert_close(actions, expected_actions)
    torch.testing.assert_close(velocity, expected_velocity)
    torch.testing.assert_close(hidden_out, expected_hidden)
    assert onnx_actor.input_names == [
        "proprio_history",
        "actor_command",
        "depth",
        "hidden_state_in",
    ]
    assert onnx_actor.output_names == [
        "actions",
        "predicted_lin_vel",
        "hidden_state_out",
    ]
    assert all("roughness" not in name and "predictor" not in name for name, _ in onnx_actor.named_parameters())

    scripted = torch.jit.script(actor.as_jit())
    scripted_actions, scripted_velocity = scripted(
        obs["proprio_history"],
        obs["actor_command"],
        actor.depth_preprocessor(obs["depth_camera"]),
    )
    actor.reset()
    fresh_velocity = actor.get_predicted_lin_vel(obs)
    fresh_actions = actor(obs)
    torch.testing.assert_close(scripted_actions, fresh_actions)
    torch.testing.assert_close(scripted_velocity, fresh_velocity)


def test_actor_only_onnx_export_has_the_deployment_interface(tmp_path: Path) -> None:
    import onnx

    export_actor = make_actor().as_onnx(verbose=False)
    export_actor.eval()
    export_path = tmp_path / "depth_actor.onnx"

    torch.onnx.export(
        export_actor,
        export_actor.get_dummy_inputs(),
        export_path,
        opset_version=18,
        input_names=export_actor.input_names,
        output_names=export_actor.output_names,
    )

    graph = onnx.load(export_path).graph
    assert [value.name for value in graph.input] == export_actor.input_names
    assert [value.name for value in graph.output] == export_actor.output_names
    assert all("roughness" not in value.name and "predictor" not in value.name for value in graph.initializer)


@pytest.mark.parametrize("roughness_loss_coef", (0.2, 0.0))
def test_roughness_ablation_changes_only_its_loss_coefficient(
    roughness_loss_coef: float,
) -> None:
    algorithm, _ = build_algorithm(roughness_loss_coef=roughness_loss_coef)
    assert algorithm.roughness_loss_coef == roughness_loss_coef
