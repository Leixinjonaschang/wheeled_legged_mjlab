# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asymmetric PPO for a directly trained recurrent depth student."""

# ruff: file-ignore[lowercase-imported-as-non-lowercase]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import DepthActor, MLPModel
from rsl_rl.models.latent_dynamics_predictor import LatentDynamicsPredictor
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class DepthPredictorPPO:
    """Train a depth actor, privileged critic, and latent predictor jointly."""

    actor: DepthActor
    critic: MLPModel
    predictor: LatentDynamicsPredictor

    def __init__(
        self,
        actor: DepthActor,
        critic: MLPModel,
        predictor: LatentDynamicsPredictor,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        num_auxiliary_epochs: int = 1,
        num_auxiliary_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1.0e-3,
        predictor_learning_rate: float = 1.0e-3,
        lin_vel_loss_coef: float = 1.0,
        roughness_loss_coef: float = 0.2,
        latent_dynamics_loss_coef: float = 3.0,
        latent_dynamics_velocity_loss_coef: float = 1.0,
        latent_dynamics_horizons: tuple[int, ...] | list[int] = (1, 5, 10),
        latent_dynamics_horizon_weights: tuple[float, ...] | list[float] = (
            1.0,
            0.75,
            0.5,
        ),
        latent_rollout_horizon: int = 5,
        latent_rollout_loss_coef: float = 0.75,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        share_cnn_encoders: bool = False,
        inference_only: bool = False,
    ) -> None:
        """Initialize models, disjoint optimizers, and update scheduling."""
        if rnd_cfg is not None:
            raise ValueError("RND is not supported by DepthPredictorPPO.")
        if symmetry_cfg is not None:
            raise ValueError("Symmetry augmentation is not supported by DepthPredictorPPO.")
        if share_cnn_encoders:
            raise ValueError("CNN encoder sharing is not supported by DepthPredictorPPO.")
        if actor.distribution is None:
            raise ValueError("DepthPredictorPPO requires a stochastic actor distribution.")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.predictor = predictor.to(device)
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self._raw_predictor = self.predictor

        self.latent_dynamics_horizons = tuple(int(horizon) for horizon in latent_dynamics_horizons)
        self.latent_dynamics_horizon_weights = tuple(float(weight) for weight in latent_dynamics_horizon_weights)
        if self.latent_dynamics_horizons != predictor.horizons:
            raise ValueError(
                "Algorithm and predictor horizons must match, got "
                f"{self.latent_dynamics_horizons} and {predictor.horizons}."
            )
        if len(self.latent_dynamics_horizons) != len(self.latent_dynamics_horizon_weights):
            raise ValueError("Each dynamics horizon must have exactly one weight.")
        if any(weight <= 0.0 for weight in self.latent_dynamics_horizon_weights):
            raise ValueError("Dynamics horizon weights must be positive.")
        if latent_dynamics_loss_coef < 0.0:
            raise ValueError("latent_dynamics_loss_coef must be non-negative.")
        if latent_dynamics_velocity_loss_coef < 0.0:
            raise ValueError("latent_dynamics_velocity_loss_coef must be non-negative.")
        if latent_rollout_horizon <= 0:
            raise ValueError("latent_rollout_horizon must be positive.")
        if latent_rollout_loss_coef < 0.0:
            raise ValueError("latent_rollout_loss_coef must be non-negative.")
        if latent_rollout_loss_coef > 0.0 and 1 not in predictor.horizons:
            raise ValueError("Autoregressive rollout requires a horizon-one predictor.")
        maximum_horizon = max(max(self.latent_dynamics_horizons), latent_rollout_horizon)
        if storage.num_transitions_per_env <= maximum_horizon:
            raise ValueError(
                "Rollout length must exceed every prediction horizon, got "
                f"{storage.num_transitions_per_env} and {maximum_horizon}."
            )
        if num_auxiliary_epochs <= 0 or num_auxiliary_mini_batches <= 0:
            raise ValueError("Auxiliary epochs and mini-batches must be positive.")
        num_ppo_updates = num_learning_epochs * num_mini_batches
        num_auxiliary_updates = num_auxiliary_epochs * num_auxiliary_mini_batches
        if num_auxiliary_updates > num_ppo_updates:
            raise ValueError("Auxiliary updates must not outnumber PPO updates.")
        if not inference_only and storage.num_envs < max(num_mini_batches, num_auxiliary_mini_batches):
            raise ValueError("Full-sequence mini-batches require at least one environment per batch.")
        if not inference_only and roughness_loss_coef > 0.0 and actor.wheel_roughness_group is None:
            raise ValueError("Positive roughness supervision requires a wheel_roughness group.")

        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_auxiliary_epochs = num_auxiliary_epochs
        self.num_auxiliary_mini_batches = num_auxiliary_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.predictor_learning_rate = predictor_learning_rate
        self.lin_vel_loss_coef = lin_vel_loss_coef
        self.roughness_loss_coef = roughness_loss_coef
        self.latent_dynamics_loss_coef = latent_dynamics_loss_coef
        self.latent_dynamics_velocity_loss_coef = latent_dynamics_velocity_loss_coef
        self.latent_dynamics_horizon_weight_by_horizon = dict(
            zip(
                self.latent_dynamics_horizons,
                self.latent_dynamics_horizon_weights,
                strict=True,
            )
        )
        self.latent_rollout_horizon = latent_rollout_horizon
        self.latent_rollout_loss_coef = latent_rollout_loss_coef
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.rnd = None
        self.inference_only = inference_only

        self._main_parameters = list(chain(self.actor.parameters(), self.critic.parameters()))
        self._predictor_parameters = list(self.predictor.parameters())
        main_ids = [id(parameter) for parameter in self._main_parameters]
        predictor_ids = [id(parameter) for parameter in self._predictor_parameters]
        if len(main_ids) != len(set(main_ids)):
            raise ValueError("Main optimizer parameter ownership contains duplicates.")
        if len(predictor_ids) != len(set(predictor_ids)):
            raise ValueError("Predictor optimizer parameter ownership contains duplicates.")
        if not set(main_ids).isdisjoint(predictor_ids):
            raise ValueError("Main and predictor optimizer parameters must be disjoint.")

        optimizer_class = resolve_optimizer(optimizer)
        self.optimizer = optimizer_class(  # type: ignore
            self._main_parameters, lr=learning_rate
        )
        self.predictor_optimizer = optimizer_class(  # type: ignore
            self._predictor_parameters, lr=predictor_learning_rate
        )
        self.last_num_ppo_updates = 0
        self.last_num_auxiliary_updates = 0
        self.last_num_predictor_updates = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample exclusively from the deployable depth actor."""
        self.transition.hidden_states = (
            self.actor.get_hidden_state(),
            self.critic.get_hidden_state(),
        )
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(
            parameter.detach() for parameter in self.actor.output_distribution_params
        )
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Store a transition while leaving normalization frozen."""
        if "applied_actions" not in extras:
            raise ValueError("Depth dynamics training requires applied_actions from the environment.")
        self.transition.applied_actions = extras["applied_actions"].to(self.device).detach()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE with the independent privileged critic."""
        storage = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(storage.num_transitions_per_env)):
            next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
            next_is_not_terminal = 1.0 - storage.dones[step].float()
            delta = storage.rewards[step] + next_is_not_terminal * self.gamma * next_values - storage.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            storage.returns[step] = advantage + storage.values[step]
        storage.advantages = storage.returns - storage.values
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages = (storage.advantages - storage.advantages.mean()) / (storage.advantages.std() + 1.0e-8)

    def update(self) -> dict[str, float]:
        """Run twenty PPO steps and interleave four full-sequence auxiliary steps."""
        if self.inference_only:
            raise RuntimeError("An inference-only depth actor cannot be trained.")
        totals: defaultdict[str, float] = defaultdict(float)
        metric_counts: defaultdict[str, int] = defaultdict(int)
        ppo_updates = self.num_learning_epochs * self.num_mini_batches
        auxiliary_updates = self.num_auxiliary_epochs * self.num_auxiliary_mini_batches
        auxiliary_indices = {((index + 1) * ppo_updates - 1) // auxiliary_updates for index in range(auxiliary_updates)}
        auxiliary_generator = iter(
            self.storage.sequence_mini_batch_generator(
                self.num_auxiliary_mini_batches,
                self.num_auxiliary_epochs,
            )
        )

        self.last_num_ppo_updates = 0
        self.last_num_auxiliary_updates = 0
        self.last_num_predictor_updates = 0
        generator = self.storage.sequence_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for update_index, batch in enumerate(generator):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1.0e-8)

            self.actor.forward_sequence(
                batch.observations,
                batch.dones,
                self._as_tensor_hidden_state(batch.hidden_states[0]),
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy
            self._adapt_learning_rate(batch.old_distribution_params, distribution_params)

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob, -1))
            advantages = torch.squeeze(batch.advantages, -1)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch.returns).pow(2),
                    (value_clipped - batch.returns).pow(2),
                ).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            auxiliary_loss = None
            dynamics_loss = None
            dynamics_valid = False
            if update_index in auxiliary_indices:
                auxiliary_batch = next(auxiliary_generator)
                (
                    auxiliary_loss,
                    dynamics_loss,
                    dynamics_valid,
                    auxiliary_metrics,
                ) = self._compute_auxiliary_objective(auxiliary_batch)
                for name, value in auxiliary_metrics.items():
                    totals[name] += value
                    metric_counts[name] += 1
                self.last_num_auxiliary_updates += 1

            self.optimizer.zero_grad(set_to_none=True)
            self.predictor_optimizer.zero_grad(set_to_none=True)
            ppo_loss.backward()
            backbone_parameters = list(self.actor.student_backbone_parameters())
            ppo_backbone_gradients = self._clone_gradients(backbone_parameters)
            totals["Grad/student_backbone_ppo_norm"] += self._snapshot_norm(ppo_backbone_gradients)
            metric_counts["Grad/student_backbone_ppo_norm"] += 1

            if auxiliary_loss is not None:
                auxiliary_loss.backward(retain_graph=dynamics_valid)
                before_prediction = self._clone_gradients(backbone_parameters)
                if dynamics_loss is not None and dynamics_valid:
                    dynamics_loss.backward()
                    prediction_gradients = self._gradient_delta(backbone_parameters, before_prediction)
                    totals["Grad/student_backbone_prediction_norm"] += self._snapshot_norm(prediction_gradients)
                    totals["Grad/student_backbone_ppo_prediction_cosine"] += self._snapshot_cosine(
                        ppo_backbone_gradients, prediction_gradients
                    )
                    metric_counts["Grad/student_backbone_prediction_norm"] += 1
                    metric_counts["Grad/student_backbone_ppo_prediction_cosine"] += 1

            if self._distributed:
                self.reduce_parameters(self._main_parameters)
                if dynamics_valid:
                    self.reduce_parameters(self._predictor_parameters)

            main_grad_norm = nn.utils.clip_grad_norm_(self._main_parameters, self.max_grad_norm).item()
            totals["Grad/main_total_norm"] += main_grad_norm
            totals["Grad/main_clip_fraction"] += float(main_grad_norm > self.max_grad_norm)
            metric_counts["Grad/main_total_norm"] += 1
            metric_counts["Grad/main_clip_fraction"] += 1
            self.optimizer.step()
            if dynamics_valid:
                predictor_grad_norm = nn.utils.clip_grad_norm_(self._predictor_parameters, self.max_grad_norm).item()
                totals["Grad/predictor_total_norm"] += predictor_grad_norm
                totals["Grad/predictor_clip_fraction"] += float(predictor_grad_norm > self.max_grad_norm)
                metric_counts["Grad/predictor_total_norm"] += 1
                metric_counts["Grad/predictor_clip_fraction"] += 1
                self.predictor_optimizer.step()
                self.last_num_predictor_updates += 1

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            metric_counts["value"] += 1
            metric_counts["surrogate"] += 1
            metric_counts["entropy"] += 1
            self.last_num_ppo_updates += 1

        # Statistics are intentionally updated only after every replay/optimizer step.
        rollout_observations = self.storage.observations.flatten(0, 1)
        self._raw_actor.update_normalization(rollout_observations)
        self._raw_critic.update_normalization(rollout_observations)

        loss_dict = {name: value / max(metric_counts[name], 1) for name, value in totals.items()}
        loss_dict.update({
            "Learning/ppo_lr": self.optimizer.param_groups[0]["lr"],
            "Learning/predictor_lr": self.predictor_optimizer.param_groups[0]["lr"],
            "Update/ppo_steps": float(self.last_num_ppo_updates),
            "Update/auxiliary_steps": float(self.last_num_auxiliary_updates),
            "Update/predictor_steps": float(self.last_num_predictor_updates),
        })
        self.storage.clear()
        return loss_dict

    def _compute_auxiliary_objective(
        self, batch: RolloutStorage.Batch
    ) -> tuple[torch.Tensor, torch.Tensor | None, bool, dict[str, float]]:
        observations = batch.observations
        dones = batch.dones
        applied_actions = batch.applied_actions
        if observations is None or dones is None or applied_actions is None:
            raise ValueError("Auxiliary sequence batches are incomplete.")
        latent, predicted_lin_vel, predicted_roughness, _ = self.actor.get_student_outputs_sequence(
            observations,
            dones,
            self._as_tensor_hidden_state(batch.hidden_states[0]),
        )
        true_lin_vel = self.actor.get_lin_vel_target(observations)
        lin_vel_values = (predicted_lin_vel - true_lin_vel).square().mean(dim=-1)
        lin_vel_loss, lin_vel_metric, _ = self._global_masked_mean(
            lin_vel_values, torch.ones_like(lin_vel_values, dtype=torch.bool)
        )
        if self.actor.wheel_roughness_group is not None:
            true_roughness = self.actor.get_wheel_roughness(observations)
            roughness_values = F.smooth_l1_loss(predicted_roughness, true_roughness, reduction="none").mean(dim=-1)
            roughness_loss, roughness_metric, _ = self._global_masked_mean(
                roughness_values,
                torch.ones_like(roughness_values, dtype=torch.bool),
            )
        else:
            roughness_loss = lin_vel_loss.new_zeros(())
            roughness_metric = 0.0

        supervised_loss = self.lin_vel_loss_coef * lin_vel_loss + self.roughness_loss_coef * roughness_loss
        normalized_true_lin_vel = self.actor.get_normalized_lin_vel_target(observations)
        dynamics_loss, dynamics_valid, dynamics_metrics = self._compute_dynamics_objective(
            latent,
            normalized_true_lin_vel,
            applied_actions,
            dones,
        )
        metrics = {
            "student": self.lin_vel_loss_coef * lin_vel_metric + self.roughness_loss_coef * roughness_metric,
            "lin_vel": lin_vel_metric,
            "roughness": roughness_metric,
            "latent_variance": self._global_latent_variance(latent),
            **dynamics_metrics,
        }
        return supervised_loss, dynamics_loss, dynamics_valid, metrics

    def _compute_dynamics_objective(
        self,
        latent: torch.Tensor,
        normalized_true_lin_vel: torch.Tensor,
        applied_actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor | None, bool, dict[str, float]]:
        """Build direct and autoregressive losses from one replayed sequence."""
        time_steps, batch_size, _ = latent.shape
        transition_valid = ~dones.squeeze(-1).bool()
        metrics: dict[str, float] = {}
        direct_loss = latent.sum() * 0.0
        active_weight = 0.0
        any_valid = False
        direct_metrics: dict[int, tuple[float, float, int]] = {}

        for horizon in self.latent_dynamics_horizons:
            num_starts = time_steps - horizon
            valid = torch.stack([transition_valid[offset : offset + num_starts] for offset in range(horizon)]).all(
                dim=0
            )
            action_block = torch.stack(
                [applied_actions[offset : offset + num_starts] for offset in range(horizon)],
                dim=-2,
            ).flatten(start_dim=-2)
            source_latent = latent[:num_starts]
            source_velocity = normalized_true_lin_vel[:num_starts]
            target_latent = latent[horizon:].detach()
            target_velocity = normalized_true_lin_vel[horizon:].detach()
            predicted_latent, predicted_velocity = self.predictor(
                source_latent,
                source_velocity,
                action_block,
                horizon,
            )
            representation_values = 1.0 - F.cosine_similarity(predicted_latent, target_latent, dim=-1)
            velocity_values = F.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none").mean(dim=-1)
            total_values = representation_values + self.latent_dynamics_velocity_loss_coef * velocity_values
            horizon_loss, horizon_metric, global_count = self._global_masked_mean(total_values, valid)
            _, representation_metric, _ = self._global_masked_mean(representation_values, valid)
            _, velocity_metric, _ = self._global_masked_mean(velocity_values, valid)
            direct_metrics[horizon] = (
                representation_metric,
                velocity_metric,
                global_count,
            )
            if global_count > 0:
                weight = self.latent_dynamics_horizon_weight_by_horizon[horizon]
                direct_loss = direct_loss + weight * horizon_loss
                active_weight += weight
                any_valid = True

            eligible_count = self._global_integer(num_starts * batch_size)
            metrics[f"latent_dynamics_loss_k{horizon}"] = horizon_metric
            metrics[f"latent_dynamics_representation_loss_k{horizon}"] = representation_metric
            metrics[f"latent_dynamics_velocity_loss_k{horizon}"] = velocity_metric
            metrics[f"latent_dynamics_valid_fraction_k{horizon}"] = global_count / max(eligible_count, 1)
            self._add_direct_control_metrics(
                metrics,
                horizon,
                horizon_metric,
                source_latent,
                source_velocity,
                target_latent,
                target_velocity,
                action_block,
                predicted_latent,
                valid,
            )

        if active_weight > 0.0:
            direct_loss = direct_loss / active_weight
        configured_weight = sum(self.latent_dynamics_horizon_weights)
        metrics["latent_dynamics_loss"] = sum(
            self.latent_dynamics_horizon_weight_by_horizon[horizon] * metrics[f"latent_dynamics_loss_k{horizon}"]
            for horizon in self.latent_dynamics_horizons
            if direct_metrics[horizon][2] > 0
        ) / max(active_weight, 1.0e-8)
        metrics["latent_dynamics_representation_loss"] = sum(
            self.latent_dynamics_horizon_weight_by_horizon[horizon] * direct_metrics[horizon][0]
            for horizon in self.latent_dynamics_horizons
            if direct_metrics[horizon][2] > 0
        ) / max(active_weight, 1.0e-8)
        metrics["latent_dynamics_velocity_loss"] = sum(
            self.latent_dynamics_horizon_weight_by_horizon[horizon] * direct_metrics[horizon][1]
            for horizon in self.latent_dynamics_horizons
            if direct_metrics[horizon][2] > 0
        ) / max(active_weight, 1.0e-8)
        metrics["latent_identity_loss"] = sum(
            self.latent_dynamics_horizon_weight_by_horizon[horizon] * metrics[f"latent_identity_loss_k{horizon}"]
            for horizon in self.latent_dynamics_horizons
            if direct_metrics[horizon][2] > 0
        ) / max(active_weight, 1.0e-8)
        metrics["latent_prediction_identity_ratio"] = metrics["latent_dynamics_loss"] / (
            metrics["latent_identity_loss"] + 1.0e-8
        )
        metrics["latent_shuffled_action_loss"] = sum(
            self.latent_dynamics_horizon_weight_by_horizon[horizon] * metrics[f"latent_shuffled_action_loss_k{horizon}"]
            for horizon in self.latent_dynamics_horizons
            if direct_metrics[horizon][2] > 0
        ) / max(active_weight, 1.0e-8)
        metrics["latent_shuffled_action_ratio"] = metrics["latent_shuffled_action_loss"] / (
            metrics["latent_dynamics_loss"] + 1.0e-8
        )
        metrics["latent_prediction_cosine_similarity"] = sum(
            self.latent_dynamics_horizon_weight_by_horizon[horizon]
            * metrics[f"latent_prediction_cosine_similarity_k{horizon}"]
            for horizon in self.latent_dynamics_horizons
            if direct_metrics[horizon][2] > 0
        ) / max(active_weight, 1.0e-8)
        metrics["latent_dynamics_valid_fraction"] = (
            sum(
                self.latent_dynamics_horizon_weight_by_horizon[horizon]
                * metrics[f"latent_dynamics_valid_fraction_k{horizon}"]
                for horizon in self.latent_dynamics_horizons
            )
            / configured_weight
        )

        rollout_loss, rollout_valid, rollout_metrics = self._compute_rollout_objective(
            latent,
            normalized_true_lin_vel,
            applied_actions,
            transition_valid,
        )
        metrics.update(rollout_metrics)
        any_valid = any_valid or rollout_valid
        if not any_valid or self.latent_dynamics_loss_coef <= 0.0:
            return None, False, metrics
        total_loss = direct_loss
        if rollout_valid:
            total_loss = total_loss + self.latent_rollout_loss_coef * rollout_loss
        return self.latent_dynamics_loss_coef * total_loss, True, metrics

    def _compute_rollout_objective(
        self,
        latent: torch.Tensor,
        normalized_true_lin_vel: torch.Tensor,
        applied_actions: torch.Tensor,
        transition_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, dict[str, float]]:
        horizon = self.latent_rollout_horizon
        time_steps, batch_size, _ = latent.shape
        num_starts = time_steps - horizon
        valid = torch.stack([transition_valid[offset : offset + num_starts] for offset in range(horizon)]).all(dim=0)
        action_sequence = torch.stack([applied_actions[offset : offset + num_starts] for offset in range(horizon)])
        source_latent = latent[:num_starts]
        source_velocity = normalized_true_lin_vel[:num_starts]
        rollout_latent, rollout_velocity = self.predictor.rollout(source_latent, source_velocity, action_sequence)
        rollout_loss = latent.sum() * 0.0
        metrics: dict[str, float] = {}
        global_count = 0
        shuffled_sequence = self._shuffle_sequence_samples(action_sequence)
        with torch.no_grad():
            shuffled_latent, shuffled_velocity = self.predictor.rollout(
                source_latent, source_velocity, shuffled_sequence
            )
        for step in range(horizon):
            target_latent = latent[step + 1 : num_starts + step + 1].detach()
            target_velocity = normalized_true_lin_vel[step + 1 : num_starts + step + 1].detach()
            representation_values = 1.0 - F.cosine_similarity(rollout_latent[step], target_latent, dim=-1)
            velocity_values = F.smooth_l1_loss(rollout_velocity[step], target_velocity, reduction="none").mean(dim=-1)
            step_values = representation_values + self.latent_dynamics_velocity_loss_coef * velocity_values
            step_loss, step_metric, global_count = self._global_masked_mean(step_values, valid)
            _, representation_metric, _ = self._global_masked_mean(representation_values, valid)
            _, velocity_metric, _ = self._global_masked_mean(velocity_values, valid)
            rollout_loss = rollout_loss + step_loss
            metrics[f"latent_rollout_loss_k{step + 1}"] = step_metric
            metrics[f"latent_rollout_representation_loss_k{step + 1}"] = representation_metric
            metrics[f"latent_rollout_velocity_loss_k{step + 1}"] = velocity_metric
            with torch.no_grad():
                identity_values = 1.0 - F.cosine_similarity(source_latent, target_latent, dim=-1)
                identity_values = identity_values + (
                    self.latent_dynamics_velocity_loss_coef
                    * F.smooth_l1_loss(source_velocity, target_velocity, reduction="none").mean(dim=-1)
                )
                shuffled_values = 1.0 - F.cosine_similarity(shuffled_latent[step], target_latent, dim=-1)
                shuffled_values = shuffled_values + (
                    self.latent_dynamics_velocity_loss_coef
                    * F.smooth_l1_loss(
                        shuffled_velocity[step],
                        target_velocity,
                        reduction="none",
                    ).mean(dim=-1)
                )
                _, identity_metric, _ = self._global_masked_mean(identity_values, valid)
                _, shuffled_metric, _ = self._global_masked_mean(shuffled_values, valid)
                cosine_values = F.cosine_similarity(rollout_latent[step], target_latent, dim=-1)
                _, cosine_metric, _ = self._global_masked_mean(cosine_values, valid)
            metrics[f"latent_rollout_identity_loss_k{step + 1}"] = identity_metric
            metrics[f"latent_rollout_identity_ratio_k{step + 1}"] = step_metric / (identity_metric + 1.0e-8)
            metrics[f"latent_rollout_shuffled_action_loss_k{step + 1}"] = shuffled_metric
            metrics[f"latent_rollout_shuffled_action_ratio_k{step + 1}"] = shuffled_metric / (step_metric + 1.0e-8)
            metrics[f"latent_rollout_cosine_similarity_k{step + 1}"] = cosine_metric
        rollout_loss = rollout_loss / horizon
        metrics["latent_rollout_loss"] = (
            sum(metrics[f"latent_rollout_loss_k{step + 1}"] for step in range(horizon)) / horizon
        )
        metrics["latent_rollout_representation_loss"] = (
            sum(metrics[f"latent_rollout_representation_loss_k{step + 1}"] for step in range(horizon)) / horizon
        )
        metrics["latent_rollout_velocity_loss"] = (
            sum(metrics[f"latent_rollout_velocity_loss_k{step + 1}"] for step in range(horizon)) / horizon
        )
        eligible_count = self._global_integer(num_starts * batch_size)
        metrics["latent_rollout_valid_fraction"] = global_count / max(eligible_count, 1)

        if horizon in self.predictor.horizons:
            direct_actions = action_sequence.permute(1, 2, 0, 3).flatten(start_dim=-2)
            with torch.no_grad():
                direct_latent, direct_velocity = self.predictor(
                    source_latent,
                    source_velocity,
                    direct_actions,
                    horizon,
                )
                mse_values = (direct_latent - rollout_latent[-1]).square().mean(dim=-1)
                cosine_values = F.cosine_similarity(direct_latent, rollout_latent[-1], dim=-1)
                velocity_values = F.smooth_l1_loss(direct_velocity, rollout_velocity[-1], reduction="none").mean(dim=-1)
                _, mse_metric, _ = self._global_masked_mean(mse_values, valid)
                _, cosine_metric, _ = self._global_masked_mean(cosine_values, valid)
                _, velocity_metric, _ = self._global_masked_mean(velocity_values, valid)
            metrics[f"latent_direct_rollout_mse_k{horizon}"] = mse_metric
            metrics[f"latent_direct_rollout_cosine_k{horizon}"] = cosine_metric
            metrics[f"latent_direct_rollout_velocity_loss_k{horizon}"] = velocity_metric
        return rollout_loss, global_count > 0, metrics

    def _add_direct_control_metrics(
        self,
        metrics: dict[str, float],
        horizon: int,
        prediction_metric: float,
        source_latent: torch.Tensor,
        source_velocity: torch.Tensor,
        target_latent: torch.Tensor,
        target_velocity: torch.Tensor,
        action_block: torch.Tensor,
        predicted_latent: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            identity_values = 1.0 - F.cosine_similarity(source_latent, target_latent, dim=-1)
            identity_values = identity_values + (
                self.latent_dynamics_velocity_loss_coef
                * F.smooth_l1_loss(source_velocity, target_velocity, reduction="none").mean(dim=-1)
            )
            shuffled_actions = self._shuffle_samples(action_block)
            shuffled_latent, shuffled_velocity = self.predictor(
                source_latent,
                source_velocity,
                shuffled_actions,
                horizon,
            )
            shuffled_values = 1.0 - F.cosine_similarity(shuffled_latent, target_latent, dim=-1)
            shuffled_values = shuffled_values + (
                self.latent_dynamics_velocity_loss_coef
                * F.smooth_l1_loss(shuffled_velocity, target_velocity, reduction="none").mean(dim=-1)
            )
            cosine_values = F.cosine_similarity(predicted_latent, target_latent, dim=-1)
            _, identity_metric, _ = self._global_masked_mean(identity_values, valid)
            _, shuffled_metric, _ = self._global_masked_mean(shuffled_values, valid)
            _, cosine_metric, _ = self._global_masked_mean(cosine_values, valid)
            metrics[f"latent_identity_loss_k{horizon}"] = identity_metric
            metrics[f"latent_prediction_identity_ratio_k{horizon}"] = prediction_metric / (identity_metric + 1.0e-8)
            metrics[f"latent_shuffled_action_loss_k{horizon}"] = shuffled_metric
            metrics[f"latent_shuffled_action_ratio_k{horizon}"] = shuffled_metric / (prediction_metric + 1.0e-8)
            metrics[f"latent_prediction_cosine_similarity_k{horizon}"] = cosine_metric
            if horizon > 1:
                reversed_actions = (
                    action_block.unflatten(-1, (horizon, self.predictor.action_dim)).flip(-2).flatten(start_dim=-2)
                )
                reversed_latent, reversed_velocity = self.predictor(
                    source_latent,
                    source_velocity,
                    reversed_actions,
                    horizon,
                )
                reversed_values = 1.0 - F.cosine_similarity(reversed_latent, target_latent, dim=-1)
                reversed_values = reversed_values + (
                    self.latent_dynamics_velocity_loss_coef
                    * F.smooth_l1_loss(reversed_velocity, target_velocity, reduction="none").mean(dim=-1)
                )
                _, reversed_metric, _ = self._global_masked_mean(reversed_values, valid)
                metrics[f"latent_reversed_action_loss_k{horizon}"] = reversed_metric
                metrics[f"latent_reversed_action_ratio_k{horizon}"] = reversed_metric / (prediction_metric + 1.0e-8)

    @property
    def _distributed(self) -> bool:
        return self.is_multi_gpu and torch.distributed.is_initialized()

    def _global_masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float, int]:
        mask_values = mask.to(dtype=values.dtype)
        local_sum = (values * mask_values).sum()
        global_sum = local_sum.detach().clone()
        global_count_tensor = mask_values.sum().detach().clone()
        if self._distributed:
            torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(global_count_tensor, op=torch.distributed.ReduceOp.SUM)
        global_count = int(global_count_tensor.item())
        if global_count == 0:
            return local_sum * 0.0, 0.0, 0
        gradient_scale = (self.gpu_world_size if self._distributed else 1) / global_count
        return (
            local_sum * gradient_scale,
            (global_sum / global_count_tensor).item(),
            global_count,
        )

    def _global_integer(self, local_value: int) -> int:
        value = torch.tensor(local_value, device=self.device, dtype=torch.long)
        if self._distributed:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return int(value.item())

    def _global_latent_variance(self, latent: torch.Tensor) -> float:
        flat = latent.detach().reshape(-1, latent.shape[-1])
        count = torch.tensor(flat.shape[0], device=flat.device, dtype=flat.dtype)
        value_sum = flat.sum(dim=0)
        square_sum = flat.square().sum(dim=0)
        if self._distributed:
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(value_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(square_sum, op=torch.distributed.ReduceOp.SUM)
        variance = square_sum / count - (value_sum / count).square()
        return variance.clamp_min(0.0).mean().item()

    def _adapt_learning_rate(
        self,
        old_distribution_params: tuple[torch.Tensor, ...],
        new_distribution_params: tuple[torch.Tensor, ...],
    ) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = self.actor.get_kl_divergence(old_distribution_params, new_distribution_params)
            local_sum = kl.sum()
            count = torch.tensor(kl.numel(), device=kl.device)
            if self._distributed:
                torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            kl_mean = local_sum / count
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            if self._distributed:
                learning_rate = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(learning_rate, src=0)
                self.learning_rate = learning_rate.item()
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = self.learning_rate

    @staticmethod
    def _as_tensor_hidden_state(hidden_state: HiddenState) -> torch.Tensor | None:
        if hidden_state is None or isinstance(hidden_state, torch.Tensor):
            return hidden_state
        raise ValueError("DepthActor storage must contain a tensor hidden state.")

    @staticmethod
    def _shuffle_samples(values: torch.Tensor) -> torch.Tensor:
        flat = values.flatten(0, 1)
        permutation = torch.randperm(flat.shape[0], device=flat.device)
        return flat[permutation].view_as(values)

    @staticmethod
    def _shuffle_sequence_samples(values: torch.Tensor) -> torch.Tensor:
        time_steps = values.shape[0]
        flat = values.flatten(1, 2)
        permutation = torch.randperm(flat.shape[1], device=flat.device)
        return flat[:, permutation].reshape(time_steps, *values.shape[1:])

    @staticmethod
    def _clone_gradients(
        parameters: Iterable[torch.nn.Parameter],
    ) -> dict[int, torch.Tensor | None]:
        return {
            id(parameter): (None if parameter.grad is None else parameter.grad.detach().clone())
            for parameter in parameters
        }

    @staticmethod
    def _gradient_delta(
        parameters: Iterable[torch.nn.Parameter],
        baseline: dict[int, torch.Tensor | None],
    ) -> dict[int, torch.Tensor | None]:
        result = {}
        for parameter in parameters:
            before = baseline[id(parameter)]
            after = parameter.grad
            if before is None and after is None:
                result[id(parameter)] = None
            elif before is None:
                result[id(parameter)] = after.detach().clone()
            elif after is None:
                result[id(parameter)] = -before
            else:
                result[id(parameter)] = after.detach() - before
        return result

    @staticmethod
    def _snapshot_norm(snapshot: dict[int, torch.Tensor | None]) -> float:
        norms = [value.norm(2) for value in snapshot.values() if value is not None]
        if not norms:
            return 0.0
        return torch.stack(norms).norm(2).item()

    @staticmethod
    def _snapshot_cosine(
        left: dict[int, torch.Tensor | None],
        right: dict[int, torch.Tensor | None],
    ) -> float:
        dot = None
        left_norm = None
        right_norm = None
        for parameter_id in left:
            left_value = left[parameter_id]
            right_value = right[parameter_id]
            if left_value is None or right_value is None:
                continue
            item_dot = (left_value * right_value).sum()
            item_left_norm = left_value.square().sum()
            item_right_norm = right_value.square().sum()
            dot = item_dot if dot is None else dot + item_dot
            left_norm = item_left_norm if left_norm is None else left_norm + item_left_norm
            right_norm = item_right_norm if right_norm is None else right_norm + item_right_norm
        if dot is None or left_norm is None or right_norm is None:
            return 0.0
        denominator = torch.sqrt(left_norm * right_norm)
        if denominator.item() <= 0.0:
            return 0.0
        return torch.clamp(dot / denominator, -1.0, 1.0).item()

    def train_mode(self) -> None:
        """Put all three learnable modules in training mode."""
        self.actor.train()
        self.critic.train()
        self.predictor.train()

    def eval_mode(self) -> None:
        """Put all three learnable modules in evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.predictor.eval()

    def save(self) -> dict:
        """Return only the new architecture's complete resumable state."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "predictor_state_dict": self._raw_predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "predictor_optimizer_state_dict": self.predictor_optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "predictor_learning_rate": self.predictor_learning_rate,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load a new-architecture checkpoint without legacy migration."""
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "predictor": True,
                "optimizer": True,
                "iteration": True,
            }
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("predictor"):
            self._raw_predictor.load_state_dict(loaded_dict["predictor_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.predictor_optimizer.load_state_dict(loaded_dict["predictor_optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            self.predictor_learning_rate = self.predictor_optimizer.param_groups[0]["lr"]
        return bool(load_cfg.get("iteration", False))

    def get_policy(self) -> DepthActor:
        """Return only the deployable actor."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile all forward modules when a torch compile mode is requested."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self.predictor = compile_model(self._raw_predictor, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DepthPredictorPPO:
        """Construct the three independent modules from their own configs."""
        algorithm_class: type[DepthPredictorPPO] = resolve_callable(  # type: ignore
            cfg["algorithm"].pop("class_name")
        )
        actor_class: type[DepthActor] = resolve_callable(  # type: ignore
            cfg["actor"].pop("class_name")
        )
        critic_class: type[MLPModel] = resolve_callable(  # type: ignore
            cfg["critic"].pop("class_name")
        )
        configured_obs_groups = cfg["obs_groups"]
        available_obs_groups = {
            set_name: list(groups)
            for set_name, groups in configured_obs_groups.items()
            if all(group in obs for group in groups)
        }
        deployment_sets = ("proprio_history", "actor_command", "depth_encoder")
        missing_deployment_sets = [set_name for set_name in deployment_sets if set_name not in available_obs_groups]
        if missing_deployment_sets:
            raise ValueError(
                f"Depth actor observations are incomplete; missing observation sets {missing_deployment_sets}."
            )
        training_sets = ("lin_vel_target", "critic")
        inference_only = any(set_name not in available_obs_groups for set_name in training_sets)
        if inference_only:
            # play/export constructs the actor without computing privileged or
            # supervised observation groups. A throwaway critic is still needed
            # by the generic runner, but is neither loaded nor evaluated.
            available_obs_groups["critic"] = available_obs_groups["actor_command"]
            default_sets = [*deployment_sets, "critic"]
        else:
            default_sets = [
                *deployment_sets,
                "lin_vel_target",
                "wheel_roughness",
                "critic",
            ]
        cfg["obs_groups"] = resolve_obs_groups(obs, available_obs_groups, default_sets)
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)

        # These generic config fields select other RSL-RL model families and are
        # intentionally not part of the dedicated DepthActor/MLP constructors.
        for key in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            cfg["actor"].pop(key, None)
            cfg["critic"].pop(key, None)

        actor = actor_class(obs, cfg["obs_groups"], env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        predictor_hidden_dims = cfg["algorithm"].pop("latent_dynamics_hidden_dims")
        predictor_activation = cfg["algorithm"].pop("latent_dynamics_activation")
        predictor = LatentDynamicsPredictor(
            latent_dim=actor.latent_dim,
            lin_vel_dim=actor.lin_vel_dim,
            action_dim=env.num_actions,
            horizons=cfg["algorithm"]["latent_dynamics_horizons"],
            hidden_dims=predictor_hidden_dims,
            activation=predictor_activation,
            normalize_latent=actor.normalize_latent,
        ).to(device)
        print(f"Depth Actor Model: {actor}")
        print(f"Privileged Critic Model: {critic}")
        print(f"Latent Dynamics Predictor: {predictor}")

        # Seed normalization once. It remains frozen through rollout and replay.
        actor.update_normalization(obs)
        critic.update_normalization(obs)
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )
        algorithm = algorithm_class(
            actor,
            critic,
            predictor,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
            inference_only=inference_only,
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm

    def broadcast_parameters(self) -> None:
        """Broadcast actor, critic, and predictor state across ranks."""
        states = [
            self._raw_actor.state_dict(),
            self._raw_critic.state_dict(),
            self._raw_predictor.state_dict(),
        ]
        torch.distributed.broadcast_object_list(states, src=0)
        self._raw_actor.load_state_dict(states[0])
        self._raw_critic.load_state_dict(states[1])
        self._raw_predictor.load_state_dict(states[2])

    def reduce_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Average one deterministic parameter set across all ranks."""
        parameters = list(parameters)
        active_parameters = [parameter for parameter in parameters if parameter.grad is not None]
        if not active_parameters:
            return
        flat_gradients = torch.cat([parameter.grad.view(-1) for parameter in active_parameters])
        torch.distributed.all_reduce(flat_gradients, op=torch.distributed.ReduceOp.SUM)
        flat_gradients /= self.gpu_world_size
        offset = 0
        for parameter in active_parameters:
            numel = parameter.numel()
            parameter.grad.copy_(flat_gradients[offset : offset + numel].view_as(parameter.grad))
            offset += numel
