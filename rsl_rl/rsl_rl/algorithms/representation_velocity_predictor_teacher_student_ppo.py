# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: D102, D107, N812

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config
from rsl_rl.models import RepresentationVelocityActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class RepresentationVelocityPredictorTeacherStudentPPO:
    """Teacher-student PPO with joint latent-and-velocity dynamics prediction."""

    def __init__(
        self,
        model: RepresentationVelocityActorCritic,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        student_learning_rate: float = 0.001,
        num_student_substeps: int = 1,
        num_representation_epochs: int | None = None,
        num_representation_mini_batches: int | None = None,
        representation_chunk_length: int = 12,
        representation_loss_coef: float = 1.0,
        lin_vel_loss_coef: float = 1.0,
        latent_dynamics_loss_coef: float = 0.0,
        latent_dynamics_velocity_loss_coef: float = 1.0,
        latent_dynamics_use_ema_target: bool = False,
        latent_dynamics_ema_decay: float = 0.99,
        latent_dynamics_horizons: tuple[int, ...] | list[int] = (1,),
        latent_dynamics_horizon_weights: tuple[float, ...] | list[float] = (1.0,),
        latent_dynamics_detach_source: bool = False,
        latent_rollout_horizon: int = 5,
        latent_rollout_loss_coef: float = 0.0,
        num_latent_dynamics_epochs: int = 1,
        num_latent_dynamics_mini_batches: int = 4,
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
    ) -> None:
        if rnd_cfg is not None:
            raise ValueError("RND is not supported by RepresentationVelocityPredictorTeacherStudentPPO.")
        if symmetry_cfg is not None:
            raise ValueError(
                "Symmetry augmentation is not supported by RepresentationVelocityPredictorTeacherStudentPPO."
            )
        if share_cnn_encoders:
            raise ValueError(
                "CNN encoder sharing is not supported by RepresentationVelocityPredictorTeacherStudentPPO."
            )
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.actor = model.to(self.device)
        self.critic = self.actor
        self._raw_actor = self.actor
        self._raw_critic = self.actor
        if latent_dynamics_loss_coef < 0.0:
            raise ValueError("latent_dynamics_loss_coef must be non-negative.")
        if latent_dynamics_velocity_loss_coef < 0.0:
            raise ValueError("latent_dynamics_velocity_loss_coef must be non-negative.")
        if not 0.0 <= latent_dynamics_ema_decay < 1.0:
            raise ValueError("latent_dynamics_ema_decay must be in [0, 1).")
        self.latent_dynamics_horizons = tuple(int(horizon) for horizon in latent_dynamics_horizons)
        self.latent_dynamics_horizon_weights = tuple(float(weight) for weight in latent_dynamics_horizon_weights)
        if not self.latent_dynamics_horizons:
            raise ValueError("latent_dynamics_horizons must not be empty.")
        if any(horizon <= 0 for horizon in self.latent_dynamics_horizons):
            raise ValueError("latent_dynamics_horizons must contain only positive integers.")
        if len(set(self.latent_dynamics_horizons)) != len(self.latent_dynamics_horizons):
            raise ValueError("latent_dynamics_horizons must not contain duplicates.")
        if len(self.latent_dynamics_horizons) != len(self.latent_dynamics_horizon_weights):
            raise ValueError("Each latent dynamics horizon must have exactly one weight.")
        if any(weight <= 0.0 for weight in self.latent_dynamics_horizon_weights):
            raise ValueError("latent_dynamics_horizon_weights must contain only positive values.")
        if latent_rollout_horizon <= 0:
            raise ValueError("latent_rollout_horizon must be positive.")
        if latent_rollout_loss_coef < 0.0:
            raise ValueError("latent_rollout_loss_coef must be non-negative.")
        self.latent_dynamics_horizon_weight_by_horizon = dict(
            zip(
                self.latent_dynamics_horizons,
                self.latent_dynamics_horizon_weights,
                strict=True,
            )
        )
        self.latent_dynamics_enabled = latent_dynamics_loss_coef > 0.0
        self.latent_rollout_enabled = self.latent_dynamics_enabled and latent_rollout_loss_coef > 0.0
        if self.latent_dynamics_enabled and not hasattr(self.actor, "compute_latent_dynamics_losses"):
            raise ValueError("Latent dynamics is only supported by a model with a latent dynamics predictor.")
        if self.latent_dynamics_enabled and tuple(self.actor.latent_dynamics_horizons) != self.latent_dynamics_horizons:
            raise ValueError(
                "Model and algorithm latent dynamics horizons must match, got "
                f"{tuple(self.actor.latent_dynamics_horizons)} and {self.latent_dynamics_horizons}."
            )
        if self.latent_dynamics_enabled and (num_latent_dynamics_epochs <= 0 or num_latent_dynamics_mini_batches <= 0):
            raise ValueError("Latent dynamics epochs and mini-batches must be positive.")
        if self.latent_dynamics_enabled and (
            num_latent_dynamics_epochs * num_latent_dynamics_mini_batches > num_learning_epochs * num_mini_batches
        ):
            raise ValueError(
                "Joint latent dynamics updates must not outnumber PPO updates; got "
                f"{num_latent_dynamics_epochs * num_latent_dynamics_mini_batches} and "
                f"{num_learning_epochs * num_mini_batches}."
            )
        if self.latent_rollout_enabled and 1 not in self.latent_dynamics_horizons:
            raise ValueError("Autoregressive latent rollout requires horizon 1 in latent_dynamics_horizons.")
        if self.latent_rollout_enabled and not hasattr(self.actor, "rollout_privileged_state"):
            raise ValueError("Latent rollout is only supported by a model with a one-step rollout method.")

        optimizer_cls = resolve_optimizer(optimizer)
        self.optimizer = optimizer_cls(self.actor.ppo_parameters(), lr=learning_rate)  # type: ignore
        self.student_optimizer = optimizer_cls(  # type: ignore
            self.actor.student_parameters(),
            lr=student_learning_rate,
        )

        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.student_learning_rate = student_learning_rate
        self.num_student_substeps = num_student_substeps
        self.num_representation_epochs = (
            num_student_substeps if num_representation_epochs is None else num_representation_epochs
        )
        self.num_representation_mini_batches = (
            num_mini_batches if num_representation_mini_batches is None else num_representation_mini_batches
        )
        self.representation_chunk_length = representation_chunk_length
        self.representation_loss_coef = representation_loss_coef
        self.lin_vel_loss_coef = lin_vel_loss_coef
        self.latent_dynamics_loss_coef = latent_dynamics_loss_coef
        self.latent_dynamics_velocity_loss_coef = latent_dynamics_velocity_loss_coef
        self.latent_dynamics_use_ema_target = latent_dynamics_use_ema_target
        self.latent_dynamics_ema_decay = latent_dynamics_ema_decay
        self.latent_dynamics_detach_source = latent_dynamics_detach_source
        self.latent_rollout_horizon = latent_rollout_horizon
        self.latent_rollout_loss_coef = latent_rollout_loss_coef
        self.num_latent_dynamics_epochs = num_latent_dynamics_epochs
        self.num_latent_dynamics_mini_batches = num_latent_dynamics_mini_batches
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.rnd = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor.act_teacher(obs, stochastic_output=True).detach()
        self.transition.values = self.actor.evaluate_teacher(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.latent_dynamics_enabled:
            if "applied_actions" not in extras:
                raise ValueError(
                    "Latent dynamics training requires applied_actions from the environment."
                )
            self.transition.applied_actions = (
                extras["applied_actions"].to(self.device).detach()
            )
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
        st = self.storage
        last_values = self.actor.evaluate_teacher(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_ppo_grad_norm = 0.0
        mean_privileged_encoder_ppo_grad_norm = 0.0
        ppo_grad_clip_count = 0
        mean_latent_dynamics_loss = {horizon: 0.0 for horizon in self.latent_dynamics_horizons}
        mean_latent_dynamics_representation_loss = {
            horizon: 0.0 for horizon in self.latent_dynamics_horizons
        }
        mean_latent_dynamics_velocity_loss = {
            horizon: 0.0 for horizon in self.latent_dynamics_horizons
        }
        mean_latent_identity_loss = {horizon: 0.0 for horizon in self.latent_dynamics_horizons}
        mean_latent_shuffled_action_loss = {horizon: 0.0 for horizon in self.latent_dynamics_horizons}
        mean_latent_reversed_action_loss = {horizon: 0.0 for horizon in self.latent_dynamics_horizons if horizon > 1}
        mean_latent_prediction_cosine = {horizon: 0.0 for horizon in self.latent_dynamics_horizons}
        latent_dynamics_samples = {horizon: 0 for horizon in self.latent_dynamics_horizons}
        rollout_steps = tuple(range(1, self.latent_rollout_horizon + 1))
        mean_latent_rollout_loss = {step: 0.0 for step in rollout_steps}
        mean_latent_rollout_representation_loss = {step: 0.0 for step in rollout_steps}
        mean_latent_rollout_velocity_loss = {step: 0.0 for step in rollout_steps}
        mean_latent_rollout_identity_loss = {step: 0.0 for step in rollout_steps}
        mean_latent_rollout_shuffled_action_loss = {step: 0.0 for step in rollout_steps}
        mean_latent_rollout_cosine = {step: 0.0 for step in rollout_steps}
        mean_latent_direct_rollout_mse = 0.0
        mean_latent_direct_rollout_cosine = 0.0
        mean_latent_direct_rollout_velocity_loss = 0.0
        latent_rollout_samples = 0
        mean_latent_dynamics_grad_norm = 0.0
        mean_privileged_encoder_dynamics_grad_norm = 0.0
        mean_privileged_encoder_ppo_dynamics_cosine = 0.0
        latent_dynamics_grad_clip_count = 0
        latent_dynamics_updates = 0
        mean_combined_grad_norm = 0.0
        combined_grad_clip_count = 0
        mean_joint_encoder_update_norm = 0.0
        mean_ppo_only_encoder_update_norm = 0.0
        mean_joint_policy_kl = 0.0
        mean_ppo_only_policy_kl = 0.0
        joint_update_diagnostics = 0
        ppo_only_update_diagnostics = 0

        num_ppo_updates = self.num_learning_epochs * self.num_mini_batches
        num_configured_dynamics_updates = (
            self.num_latent_dynamics_epochs * self.num_latent_dynamics_mini_batches
            if self.latent_dynamics_enabled
            else 0
        )
        # Spread auxiliary batches across PPO so shared parameters receive one joint Adam step,
        # rather than extra dynamics-only steps that would reuse PPO's optimizer momentum.
        dynamics_update_indices = {
            ((update + 1) * num_ppo_updates - 1) // num_configured_dynamics_updates
            for update in range(num_configured_dynamics_updates)
        }
        ppo_only_diagnostic_indices = {
            update - 1 for update in dynamics_update_indices if update > 0 and update - 1 not in dynamics_update_indices
        }
        dynamics_generator = iter(self._latent_dynamics_batch_generator())

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for update_index, batch in enumerate(generator):
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor.act_teacher(
                batch.observations,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.actor.evaluate_teacher(
                batch.observations,
                hidden_state=batch.hidden_states[0],
            )
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            dynamics_batches = next(dynamics_generator, None) if update_index in dynamics_update_indices else None
            dynamics_loss = None
            if dynamics_batches is not None:
                horizon_batches, rollout_batch = dynamics_batches
                (
                    dynamics_loss,
                    horizon_metrics,
                    rollout_metrics,
                    direct_rollout_mse,
                    direct_rollout_cosine,
                    direct_rollout_velocity_loss,
                    rollout_batch_size,
                ) = self._compute_latent_dynamics_objective(horizon_batches, rollout_batch)
                for horizon, metrics in horizon_metrics.items():
                    (
                        horizon_loss,
                        representation_loss,
                        velocity_loss,
                        identity_loss,
                        shuffled_action_loss,
                        reversed_action_loss,
                        prediction_cosine,
                        batch_size,
                    ) = metrics
                    mean_latent_dynamics_loss[horizon] += horizon_loss * batch_size
                    mean_latent_dynamics_representation_loss[horizon] += representation_loss * batch_size
                    mean_latent_dynamics_velocity_loss[horizon] += velocity_loss * batch_size
                    mean_latent_identity_loss[horizon] += identity_loss * batch_size
                    mean_latent_shuffled_action_loss[horizon] += shuffled_action_loss * batch_size
                    if horizon > 1:
                        mean_latent_reversed_action_loss[horizon] += reversed_action_loss * batch_size
                    mean_latent_prediction_cosine[horizon] += prediction_cosine * batch_size
                    latent_dynamics_samples[horizon] += batch_size
                for rollout_step, metrics in rollout_metrics.items():
                    (
                        rollout_loss,
                        representation_loss,
                        velocity_loss,
                        identity_loss,
                        shuffled_action_loss,
                        rollout_cosine,
                    ) = metrics
                    mean_latent_rollout_loss[rollout_step] += rollout_loss * rollout_batch_size
                    mean_latent_rollout_representation_loss[rollout_step] += (
                        representation_loss * rollout_batch_size
                    )
                    mean_latent_rollout_velocity_loss[rollout_step] += velocity_loss * rollout_batch_size
                    mean_latent_rollout_identity_loss[rollout_step] += identity_loss * rollout_batch_size
                    mean_latent_rollout_shuffled_action_loss[rollout_step] += shuffled_action_loss * rollout_batch_size
                    mean_latent_rollout_cosine[rollout_step] += rollout_cosine * rollout_batch_size
                mean_latent_direct_rollout_mse += direct_rollout_mse * rollout_batch_size
                mean_latent_direct_rollout_cosine += direct_rollout_cosine * rollout_batch_size
                mean_latent_direct_rollout_velocity_loss += (
                    direct_rollout_velocity_loss * rollout_batch_size
                )
                latent_rollout_samples += rollout_batch_size

            # Compare each joint step with the neighboring PPO-only step at the parameter-update level.
            collect_update_diagnostics = dynamics_loss is not None or update_index in ppo_only_diagnostic_indices
            encoder_before = (
                self._clone_parameters(self.actor.privileged_encoder.parameters())
                if collect_update_diagnostics
                else None
            )
            distribution_params_before = (
                tuple(param.detach().clone() for param in distribution_params) if collect_update_diagnostics else None
            )

            self.optimizer.zero_grad()
            ppo_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor.ppo_parameters())
            ppo_grad_norm = self._grad_norm(self.actor.ppo_parameters())
            privileged_encoder_ppo_grad_norm = self._grad_norm(self.actor.privileged_encoder.parameters())
            mean_ppo_grad_norm += ppo_grad_norm
            mean_privileged_encoder_ppo_grad_norm += privileged_encoder_ppo_grad_norm
            ppo_grad_clip_count += ppo_grad_norm > self.max_grad_norm

            if dynamics_loss is not None:
                dynamics_parameters = list(self.actor.latent_dynamics_parameters())
                baseline_gradients = self._clone_gradients(dynamics_parameters)
                dynamics_loss.backward()
                if self.is_multi_gpu:
                    self._reduce_gradient_delta(dynamics_parameters, baseline_gradients)
                latent_dynamics_grad_norm = self._gradient_delta_norm(
                    dynamics_parameters,
                    baseline_gradients,
                )
                privileged_encoder_dynamics_grad_norm = self._gradient_delta_norm(
                    list(self.actor.privileged_encoder.parameters()),
                    baseline_gradients,
                )
                privileged_encoder_ppo_dynamics_cosine = self._gradient_delta_cosine(
                    self.actor.privileged_encoder.parameters(),
                    baseline_gradients,
                )
                mean_latent_dynamics_grad_norm += latent_dynamics_grad_norm
                mean_privileged_encoder_dynamics_grad_norm += privileged_encoder_dynamics_grad_norm
                mean_privileged_encoder_ppo_dynamics_cosine += privileged_encoder_ppo_dynamics_cosine
                latent_dynamics_grad_clip_count += latent_dynamics_grad_norm > self.max_grad_norm
                latent_dynamics_updates += 1

            combined_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.ppo_parameters(),
                self.max_grad_norm,
            ).item()
            mean_combined_grad_norm += combined_grad_norm
            combined_grad_clip_count += combined_grad_norm > self.max_grad_norm
            self.optimizer.step()
            if self.latent_dynamics_enabled and self.latent_dynamics_use_ema_target:
                self._raw_actor.update_latent_dynamics_target(self.latent_dynamics_ema_decay)

            if collect_update_diagnostics:
                encoder_update_norm = self._parameter_delta_norm(
                    self.actor.privileged_encoder.parameters(),
                    encoder_before,
                )
                policy_kl = self._policy_step_kl(
                    batch,
                    original_batch_size,
                    distribution_params_before,
                )
                if dynamics_loss is not None:
                    mean_joint_encoder_update_norm += encoder_update_norm
                    mean_joint_policy_kl += policy_kl
                    joint_update_diagnostics += 1
                else:
                    mean_ppo_only_encoder_update_norm += encoder_update_norm
                    mean_ppo_only_policy_kl += policy_kl
                    ppo_only_update_diagnostics += 1

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        mean_student_loss = 0.0
        mean_representation_loss = 0.0
        mean_lin_vel_loss = 0.0
        student_updates = 0
        if hasattr(self.actor, "compute_student_losses_sequence"):
            generator = self.storage.representation_chunk_generator(
                self.num_representation_mini_batches,
                self.num_representation_epochs,
                self.representation_chunk_length,
            )
            for batch in generator:
                for _ in range(self.num_student_substeps):
                    _, representation_loss, lin_vel_loss = self.actor.compute_student_losses_sequence(
                        batch.observations,
                        batch.dones,
                        hidden_state=batch.hidden_states[0],
                    )
                    student_loss = (
                        self.representation_loss_coef * representation_loss + self.lin_vel_loss_coef * lin_vel_loss
                    )
                    self.student_optimizer.zero_grad()
                    student_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters(self.actor.student_parameters())
                    nn.utils.clip_grad_norm_(self.actor.student_parameters(), self.max_grad_norm)
                    self.student_optimizer.step()

                    mean_student_loss += student_loss.item()
                    mean_representation_loss += representation_loss.item()
                    mean_lin_vel_loss += lin_vel_loss.item()
                    student_updates += 1
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            for batch in generator:
                for _ in range(self.num_student_substeps):
                    _, representation_loss, lin_vel_loss = self.actor.compute_student_losses(batch.observations)
                    student_loss = (
                        self.representation_loss_coef * representation_loss + self.lin_vel_loss_coef * lin_vel_loss
                    )
                    self.student_optimizer.zero_grad()
                    student_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters(self.actor.student_parameters())
                    nn.utils.clip_grad_norm_(self.actor.student_parameters(), self.max_grad_norm)
                    self.student_optimizer.step()

                    mean_student_loss += student_loss.item()
                    mean_representation_loss += representation_loss.item()
                    mean_lin_vel_loss += lin_vel_loss.item()
                    student_updates += 1

        num_ppo_updates = self.num_learning_epochs * self.num_mini_batches
        mean_ppo_grad_norm /= num_ppo_updates
        mean_privileged_encoder_ppo_grad_norm /= num_ppo_updates
        mean_combined_grad_norm /= num_ppo_updates
        loss_dict = {
            "value": mean_value_loss / num_ppo_updates,
            "surrogate": mean_surrogate_loss / num_ppo_updates,
            "entropy": mean_entropy / num_ppo_updates,
            "student": mean_student_loss / student_updates,
            "representation": mean_representation_loss / student_updates,
            "lin_vel": mean_lin_vel_loss / student_updates,
            "Grad/ppo_total_norm": mean_ppo_grad_norm,
            "Grad/privileged_encoder_ppo_norm": mean_privileged_encoder_ppo_grad_norm,
            "Grad/ppo_clip_fraction": ppo_grad_clip_count / num_ppo_updates,
            "Grad/combined_total_norm": mean_combined_grad_norm,
            "Grad/combined_clip_fraction": combined_grad_clip_count / num_ppo_updates,
            "Update/privileged_encoder_norm_joint": (mean_joint_encoder_update_norm / max(joint_update_diagnostics, 1)),
            "Update/privileged_encoder_norm_ppo_only": (
                mean_ppo_only_encoder_update_norm / max(ppo_only_update_diagnostics, 1)
            ),
            "Update/policy_kl_joint": mean_joint_policy_kl / max(joint_update_diagnostics, 1),
            "Update/policy_kl_ppo_only": mean_ppo_only_policy_kl / max(ppo_only_update_diagnostics, 1),
            "Update/joint_step_fraction": joint_update_diagnostics / num_ppo_updates,
        }
        if self.latent_dynamics_enabled:
            if latent_dynamics_updates > 0:
                mean_latent_dynamics_grad_norm /= latent_dynamics_updates
                mean_privileged_encoder_dynamics_grad_norm /= latent_dynamics_updates
                mean_privileged_encoder_ppo_dynamics_cosine /= latent_dynamics_updates
            loss_dict.update({
                "Grad/dynamics_total_norm": mean_latent_dynamics_grad_norm,
                "Grad/privileged_encoder_dynamics_norm": mean_privileged_encoder_dynamics_grad_norm,
                "Grad/privileged_encoder_dynamics_to_ppo_ratio": (
                    mean_privileged_encoder_dynamics_grad_norm / (mean_privileged_encoder_ppo_grad_norm + 1.0e-8)
                ),
                "Grad/privileged_encoder_ppo_dynamics_cosine": mean_privileged_encoder_ppo_dynamics_cosine,
                "Grad/dynamics_clip_fraction": (latent_dynamics_grad_clip_count / max(latent_dynamics_updates, 1)),
            })
            for horizon in self.latent_dynamics_horizons:
                if latent_dynamics_samples[horizon] > 0:
                    mean_latent_dynamics_loss[horizon] /= latent_dynamics_samples[horizon]
                    mean_latent_dynamics_representation_loss[horizon] /= latent_dynamics_samples[horizon]
                    mean_latent_dynamics_velocity_loss[horizon] /= latent_dynamics_samples[horizon]
                    mean_latent_identity_loss[horizon] /= latent_dynamics_samples[horizon]
                    mean_latent_shuffled_action_loss[horizon] /= latent_dynamics_samples[horizon]
                    if horizon > 1:
                        mean_latent_reversed_action_loss[horizon] /= latent_dynamics_samples[horizon]
                    mean_latent_prediction_cosine[horizon] /= latent_dynamics_samples[horizon]

            available_horizons = [
                horizon for horizon in self.latent_dynamics_horizons if latent_dynamics_samples[horizon] > 0
            ]
            available_weight = sum(
                self.latent_dynamics_horizon_weight_by_horizon[horizon] for horizon in available_horizons
            )

            def weighted_horizon_mean(values: dict[int, float]) -> float:
                if not available_weight:
                    return 0.0
                return (
                    sum(
                        self.latent_dynamics_horizon_weight_by_horizon[horizon] * values[horizon]
                        for horizon in available_horizons
                    )
                    / available_weight
                )

            aggregate_dynamics_loss = weighted_horizon_mean(mean_latent_dynamics_loss)
            aggregate_representation_loss = weighted_horizon_mean(mean_latent_dynamics_representation_loss)
            aggregate_velocity_loss = weighted_horizon_mean(mean_latent_dynamics_velocity_loss)
            aggregate_identity_loss = weighted_horizon_mean(mean_latent_identity_loss)
            aggregate_shuffled_action_loss = weighted_horizon_mean(mean_latent_shuffled_action_loss)
            aggregate_prediction_cosine = weighted_horizon_mean(mean_latent_prediction_cosine)
            configured_weight = sum(self.latent_dynamics_horizon_weights)
            aggregate_valid_fraction = (
                sum(
                    self.latent_dynamics_horizon_weight_by_horizon[horizon]
                    * self.storage.latent_dynamics_valid_fractions.get(horizon, 0.0)
                    for horizon in self.latent_dynamics_horizons
                )
                / configured_weight
            )
            loss_dict.update({
                "latent_dynamics_loss": aggregate_dynamics_loss,
                "latent_dynamics_representation_loss": aggregate_representation_loss,
                "latent_dynamics_velocity_loss": aggregate_velocity_loss,
                "latent_identity_loss": aggregate_identity_loss,
                "latent_prediction_identity_ratio": aggregate_dynamics_loss / (aggregate_identity_loss + 1.0e-8),
                "latent_shuffled_action_loss": aggregate_shuffled_action_loss,
                "latent_shuffled_action_ratio": aggregate_shuffled_action_loss / (aggregate_dynamics_loss + 1.0e-8),
                "latent_prediction_cosine_similarity": aggregate_prediction_cosine,
                "latent_dynamics_valid_fraction": aggregate_valid_fraction,
            })
            for horizon in self.latent_dynamics_horizons:
                loss_dict.update({
                    f"latent_dynamics_loss_k{horizon}": mean_latent_dynamics_loss[horizon],
                    f"latent_dynamics_representation_loss_k{horizon}": (
                        mean_latent_dynamics_representation_loss[horizon]
                    ),
                    f"latent_dynamics_velocity_loss_k{horizon}": mean_latent_dynamics_velocity_loss[horizon],
                    f"latent_identity_loss_k{horizon}": mean_latent_identity_loss[horizon],
                    f"latent_prediction_identity_ratio_k{horizon}": mean_latent_dynamics_loss[horizon]
                    / (mean_latent_identity_loss[horizon] + 1.0e-8),
                    f"latent_shuffled_action_loss_k{horizon}": mean_latent_shuffled_action_loss[horizon],
                    f"latent_shuffled_action_ratio_k{horizon}": mean_latent_shuffled_action_loss[horizon]
                    / (mean_latent_dynamics_loss[horizon] + 1.0e-8),
                    f"latent_prediction_cosine_similarity_k{horizon}": mean_latent_prediction_cosine[horizon],
                    f"latent_dynamics_valid_fraction_k{horizon}": self.storage.latent_dynamics_valid_fractions.get(
                        horizon,
                        0.0,
                    ),
                })
                if horizon > 1:
                    loss_dict.update({
                        f"latent_reversed_action_loss_k{horizon}": mean_latent_reversed_action_loss[horizon],
                        f"latent_reversed_action_ratio_k{horizon}": mean_latent_reversed_action_loss[horizon]
                        / (mean_latent_dynamics_loss[horizon] + 1.0e-8),
                    })
        if self.latent_rollout_enabled:
            if latent_rollout_samples > 0:
                for step in rollout_steps:
                    mean_latent_rollout_loss[step] /= latent_rollout_samples
                    mean_latent_rollout_representation_loss[step] /= latent_rollout_samples
                    mean_latent_rollout_velocity_loss[step] /= latent_rollout_samples
                    mean_latent_rollout_identity_loss[step] /= latent_rollout_samples
                    mean_latent_rollout_shuffled_action_loss[step] /= latent_rollout_samples
                    mean_latent_rollout_cosine[step] /= latent_rollout_samples
                mean_latent_direct_rollout_mse /= latent_rollout_samples
                mean_latent_direct_rollout_cosine /= latent_rollout_samples
                mean_latent_direct_rollout_velocity_loss /= latent_rollout_samples

            loss_dict.update({
                "latent_rollout_loss": sum(mean_latent_rollout_loss.values()) / len(rollout_steps),
                "latent_rollout_representation_loss": (
                    sum(mean_latent_rollout_representation_loss.values()) / len(rollout_steps)
                ),
                "latent_rollout_velocity_loss": (
                    sum(mean_latent_rollout_velocity_loss.values()) / len(rollout_steps)
                ),
                "latent_rollout_valid_fraction": self.storage.latent_dynamics_sequence_valid_fraction,
            })
            for step in rollout_steps:
                loss_dict.update({
                    f"latent_rollout_loss_k{step}": mean_latent_rollout_loss[step],
                    f"latent_rollout_representation_loss_k{step}": (
                        mean_latent_rollout_representation_loss[step]
                    ),
                    f"latent_rollout_velocity_loss_k{step}": mean_latent_rollout_velocity_loss[step],
                    f"latent_rollout_identity_ratio_k{step}": mean_latent_rollout_loss[step]
                    / (mean_latent_rollout_identity_loss[step] + 1.0e-8),
                    f"latent_rollout_shuffled_action_loss_k{step}": (mean_latent_rollout_shuffled_action_loss[step]),
                    f"latent_rollout_shuffled_action_ratio_k{step}": (
                        mean_latent_rollout_shuffled_action_loss[step] / (mean_latent_rollout_loss[step] + 1.0e-8)
                    ),
                    f"latent_rollout_cosine_similarity_k{step}": mean_latent_rollout_cosine[step],
                })
            if self.latent_rollout_horizon in self.latent_dynamics_horizons:
                loss_dict.update({
                    f"latent_direct_rollout_cosine_k{self.latent_rollout_horizon}": (mean_latent_direct_rollout_cosine),
                    f"latent_direct_rollout_mse_k{self.latent_rollout_horizon}": (mean_latent_direct_rollout_mse),
                    f"latent_direct_rollout_velocity_loss_k{self.latent_rollout_horizon}": (
                        mean_latent_direct_rollout_velocity_loss
                    ),
                })
        self.storage.clear()
        return loss_dict

    def _latent_dynamics_batch_generator(
        self,
    ) -> Iterable[tuple[dict[int, RolloutStorage.Batch], RolloutStorage.Batch | None]]:
        dynamics_generators = {
            horizon: iter(
                self.storage.latent_dynamics_mini_batch_generator(
                    self.num_latent_dynamics_mini_batches,
                    self.num_latent_dynamics_epochs,
                    horizon=horizon,
                )
            )
            for horizon in self.latent_dynamics_horizons
        }
        rollout_generator = (
            iter(
                self.storage.latent_dynamics_sequence_mini_batch_generator(
                    self.num_latent_dynamics_mini_batches,
                    self.num_latent_dynamics_epochs,
                    rollout_horizon=self.latent_rollout_horizon,
                )
            )
            if self.latent_rollout_enabled
            else None
        )
        while dynamics_generators or rollout_generator is not None:
            horizon_batches = {}
            for horizon, generator in tuple(dynamics_generators.items()):
                try:
                    horizon_batches[horizon] = next(generator)
                except StopIteration:  # noqa: PERF203
                    del dynamics_generators[horizon]
            rollout_batch = None
            if rollout_generator is not None:
                try:
                    rollout_batch = next(rollout_generator)
                except StopIteration:
                    rollout_generator = None
            if horizon_batches or rollout_batch is not None:
                yield horizon_batches, rollout_batch

    def _compute_latent_dynamics_objective(
        self,
        horizon_batches: dict[int, RolloutStorage.Batch],
        rollout_batch: RolloutStorage.Batch | None,
    ) -> tuple[
        torch.Tensor,
        dict[int, tuple[float, float, float, float, float, float, float, int]],
        dict[int, tuple[float, float, float, float, float, float]],
        float,
        float,
        float,
        int,
    ]:
        active_weight = sum(self.latent_dynamics_horizon_weight_by_horizon[horizon] for horizon in horizon_batches)
        weighted_loss = torch.zeros((), device=self.device)
        horizon_metrics = {}
        for horizon, batch in horizon_batches.items():
            observations_t = batch.observations
            observations_future = batch.next_observations
            applied_action_block = batch.applied_actions
            representation_loss, velocity_loss = self.actor.compute_latent_dynamics_losses(
                observations_t,
                applied_action_block,
                observations_future,
                horizon=horizon,
                detach_source=self.latent_dynamics_detach_source,
                use_ema_target=self.latent_dynamics_use_ema_target,
            )
            dynamics_loss = (
                representation_loss
                + self.latent_dynamics_velocity_loss_coef * velocity_loss
            )
            weighted_loss = weighted_loss + (self.latent_dynamics_horizon_weight_by_horizon[horizon] * dynamics_loss)

            with torch.no_grad():
                latent_t = self.actor.get_privileged_latent(observations_t)
                normalized_lin_vel_t = self.actor.get_normalized_lin_vel_target(observations_t)
                latent_future, normalized_lin_vel_future = self.actor.get_latent_dynamics_target(
                    observations_future,
                    use_ema_target=self.latent_dynamics_use_ema_target,
                )
                predicted_latent_future, predicted_normalized_lin_vel_future = (
                    self.actor.predict_privileged_state(
                        latent_t,
                        normalized_lin_vel_t,
                        applied_action_block,
                        horizon,
                    )
                )
                shuffled_action_block = applied_action_block[
                    torch.randperm(applied_action_block.shape[0], device=applied_action_block.device)
                ]
                shuffled_latent_prediction, shuffled_velocity_prediction = self.actor.predict_privileged_state(
                    latent_t,
                    normalized_lin_vel_t,
                    shuffled_action_block,
                    horizon,
                )
                identity_loss = (
                    (1.0 - F.cosine_similarity(latent_t, latent_future, dim=-1)).mean()
                    + self.latent_dynamics_velocity_loss_coef
                    * F.smooth_l1_loss(normalized_lin_vel_t, normalized_lin_vel_future)
                )
                shuffled_action_loss = (
                    (
                        1.0
                        - F.cosine_similarity(
                            shuffled_latent_prediction,
                            latent_future,
                            dim=-1,
                        )
                    ).mean()
                    + self.latent_dynamics_velocity_loss_coef
                    * F.smooth_l1_loss(
                        shuffled_velocity_prediction,
                        normalized_lin_vel_future,
                    )
                )
                reversed_action_loss = torch.zeros((), device=self.device)
                if horizon > 1:
                    reversed_action_block = (
                        applied_action_block
                        .reshape(
                            applied_action_block.shape[0],
                            horizon,
                            self.actor.latent_dynamics_action_dim,
                        )
                        .flip(dims=(1,))
                        .flatten(start_dim=1)
                    )
                    reversed_latent_prediction, reversed_velocity_prediction = self.actor.predict_privileged_state(
                        latent_t,
                        normalized_lin_vel_t,
                        reversed_action_block,
                        horizon,
                    )
                    reversed_action_loss = (
                        (
                            1.0
                            - F.cosine_similarity(
                                reversed_latent_prediction,
                                latent_future,
                                dim=-1,
                            )
                        ).mean()
                        + self.latent_dynamics_velocity_loss_coef
                        * F.smooth_l1_loss(
                            reversed_velocity_prediction,
                            normalized_lin_vel_future,
                        )
                    )
                prediction_cosine = F.cosine_similarity(
                    predicted_latent_future,
                    latent_future,
                    dim=-1,
                ).mean()

            horizon_metrics[horizon] = (
                dynamics_loss.item(),
                representation_loss.item(),
                velocity_loss.item(),
                identity_loss.item(),
                shuffled_action_loss.item(),
                reversed_action_loss.item(),
                prediction_cosine.item(),
                observations_t.batch_size[0],
            )

        if horizon_batches:
            weighted_loss = weighted_loss / active_weight

        rollout_metrics = {}
        direct_rollout_mse = 0.0
        direct_rollout_cosine = 0.0
        direct_rollout_velocity_loss = 0.0
        rollout_batch_size = 0
        if rollout_batch is not None:
            observations_t = rollout_batch.observations
            future_observations = rollout_batch.future_observations
            applied_action_sequence = rollout_batch.applied_actions

            latent_t = self.actor.get_privileged_latent(observations_t)
            normalized_lin_vel_t = self.actor.get_normalized_lin_vel_target(observations_t)
            if self.latent_dynamics_detach_source:
                latent_t = latent_t.detach()
            with torch.no_grad():
                future_latents, future_normalized_lin_vels = self.actor.get_latent_dynamics_target(
                    future_observations,
                    use_ema_target=self.latent_dynamics_use_ema_target,
                )
            rollout_latent_predictions, rollout_velocity_predictions = self.actor.rollout_privileged_state(
                latent_t,
                normalized_lin_vel_t,
                applied_action_sequence,
            )
            rollout_representation_losses = torch.stack([
                (
                    1.0
                    - F.cosine_similarity(
                        rollout_latent_predictions[step],
                        future_latents[step],
                        dim=-1,
                    )
                ).mean()
                for step in range(self.latent_rollout_horizon)
            ])
            rollout_velocity_losses = torch.stack([
                F.smooth_l1_loss(
                    rollout_velocity_predictions[step],
                    future_normalized_lin_vels[step],
                )
                for step in range(self.latent_rollout_horizon)
            ])
            rollout_step_losses = (
                rollout_representation_losses
                + self.latent_dynamics_velocity_loss_coef * rollout_velocity_losses
            )
            weighted_loss = weighted_loss + self.latent_rollout_loss_coef * rollout_step_losses.mean()

            with torch.no_grad():
                rollout_batch_size = observations_t.batch_size[0]
                shuffled_action_sequence = applied_action_sequence[
                    :,
                    torch.randperm(rollout_batch_size, device=applied_action_sequence.device),
                ]
                shuffled_rollout_latents, shuffled_rollout_velocities = self.actor.rollout_privileged_state(
                    latent_t,
                    normalized_lin_vel_t,
                    shuffled_action_sequence,
                )
                for step in range(self.latent_rollout_horizon):
                    identity_loss = (
                        (
                            1.0
                            - F.cosine_similarity(
                                latent_t,
                                future_latents[step],
                                dim=-1,
                            )
                        ).mean()
                        + self.latent_dynamics_velocity_loss_coef
                        * F.smooth_l1_loss(
                            normalized_lin_vel_t,
                            future_normalized_lin_vels[step],
                        )
                    )
                    shuffled_action_loss = (
                        (
                            1.0
                            - F.cosine_similarity(
                                shuffled_rollout_latents[step],
                                future_latents[step],
                                dim=-1,
                            )
                        ).mean()
                        + self.latent_dynamics_velocity_loss_coef
                        * F.smooth_l1_loss(
                            shuffled_rollout_velocities[step],
                            future_normalized_lin_vels[step],
                        )
                    )
                    rollout_cosine = F.cosine_similarity(
                        rollout_latent_predictions[step],
                        future_latents[step],
                        dim=-1,
                    ).mean()
                    rollout_metrics[step + 1] = (
                        rollout_step_losses[step].item(),
                        rollout_representation_losses[step].item(),
                        rollout_velocity_losses[step].item(),
                        identity_loss.item(),
                        shuffled_action_loss.item(),
                        rollout_cosine.item(),
                    )

                if self.latent_rollout_horizon in self.latent_dynamics_horizons:
                    direct_action_block = applied_action_sequence.transpose(0, 1).flatten(start_dim=1)
                    direct_latent_prediction, direct_velocity_prediction = self.actor.predict_privileged_state(
                        latent_t,
                        normalized_lin_vel_t,
                        direct_action_block,
                        self.latent_rollout_horizon,
                    )
                    direct_rollout_mse = F.mse_loss(
                        direct_latent_prediction,
                        rollout_latent_predictions[-1],
                    ).item()
                    direct_rollout_cosine = (
                        F
                        .cosine_similarity(
                            direct_latent_prediction,
                            rollout_latent_predictions[-1],
                            dim=-1,
                        )
                        .mean()
                        .item()
                    )
                    direct_rollout_velocity_loss = F.smooth_l1_loss(
                        direct_velocity_prediction,
                        rollout_velocity_predictions[-1],
                    ).item()

        return (
            self.latent_dynamics_loss_coef * weighted_loss,
            horizon_metrics,
            rollout_metrics,
            direct_rollout_mse,
            direct_rollout_cosine,
            direct_rollout_velocity_loss,
            rollout_batch_size,
        )

    @staticmethod
    def _clone_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in parameters]

    @staticmethod
    def _clone_gradients(
        parameters: Iterable[torch.nn.Parameter],
    ) -> dict[int, torch.Tensor | None]:
        return {
            id(parameter): None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        }

    @staticmethod
    def _gradient_delta_norm(
        parameters: Iterable[torch.nn.Parameter],
        baseline_gradients: dict[int, torch.Tensor | None],
    ) -> float:
        delta_norms = []
        for parameter in parameters:
            baseline = baseline_gradients.get(id(parameter))
            if parameter.grad is None:
                continue
            delta = parameter.grad.detach() if baseline is None else parameter.grad.detach() - baseline
            delta_norms.append(delta.norm(2))
        if not delta_norms:
            return 0.0
        return torch.stack(delta_norms).norm(2).item()

    @staticmethod
    def _gradient_delta_cosine(
        parameters: Iterable[torch.nn.Parameter],
        baseline_gradients: dict[int, torch.Tensor | None],
    ) -> float:
        """Return cosine similarity between baseline and subsequently added gradients."""
        dot_product = None
        baseline_squared_norm = None
        delta_squared_norm = None
        for parameter in parameters:
            baseline = baseline_gradients.get(id(parameter))
            current = None if parameter.grad is None else parameter.grad.detach()
            if baseline is None and current is None:
                continue
            if baseline is None:
                assert current is not None
                delta = current
                parameter_dot = torch.zeros((), device=current.device, dtype=current.dtype)
                parameter_baseline_squared_norm = torch.zeros((), device=current.device, dtype=current.dtype)
            else:
                delta = -baseline if current is None else current - baseline
                parameter_dot = torch.sum(baseline * delta)
                parameter_baseline_squared_norm = torch.sum(baseline.square())
            parameter_delta_squared_norm = torch.sum(delta.square())
            if dot_product is None:
                dot_product = parameter_dot
                baseline_squared_norm = parameter_baseline_squared_norm
                delta_squared_norm = parameter_delta_squared_norm
            else:
                dot_product = dot_product + parameter_dot
                baseline_squared_norm = baseline_squared_norm + parameter_baseline_squared_norm
                delta_squared_norm = delta_squared_norm + parameter_delta_squared_norm

        if dot_product is None or baseline_squared_norm is None or delta_squared_norm is None:
            return 0.0
        denominator = torch.sqrt(baseline_squared_norm * delta_squared_norm)
        if denominator.item() == 0.0:
            return 0.0
        return torch.clamp(dot_product / denominator, min=-1.0, max=1.0).item()

    def _reduce_gradient_delta(
        self,
        parameters: list[torch.nn.Parameter],
        baseline_gradients: dict[int, torch.Tensor | None],
    ) -> None:
        for parameter in parameters:
            baseline = baseline_gradients[id(parameter)]
            if parameter.grad is not None and baseline is not None:
                parameter.grad.sub_(baseline)
        self.reduce_parameters(parameters)
        reduced_gradients = self._clone_gradients(parameters)
        for parameter in parameters:
            baseline = baseline_gradients[id(parameter)]
            reduced = reduced_gradients[id(parameter)]
            if baseline is None:
                parameter.grad = reduced
            elif reduced is None:
                parameter.grad = baseline
            else:
                parameter.grad = baseline + reduced

    @staticmethod
    def _parameter_delta_norm(
        parameters: Iterable[torch.nn.Parameter],
        before: list[torch.Tensor],
    ) -> float:
        delta_norms = [
            (parameter.detach() - previous).norm(2) for parameter, previous in zip(parameters, before, strict=True)
        ]
        if not delta_norms:
            return 0.0
        return torch.stack(delta_norms).norm(2).item()

    @torch.inference_mode()
    def _policy_step_kl(
        self,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
        distribution_params_before: tuple[torch.Tensor, ...],
    ) -> float:
        hidden_state_before = self.actor.get_hidden_state()
        if isinstance(hidden_state_before, torch.Tensor):
            hidden_state_before = hidden_state_before.detach().clone()
        self.actor.act_teacher(
            batch.observations,
            hidden_state=batch.hidden_states[0],
            stochastic_output=False,
        )
        distribution_params_after = tuple(
            parameter[:original_batch_size].detach() for parameter in self.actor.output_distribution_params
        )
        policy_kl = self.actor.get_kl_divergence(
            distribution_params_before,
            distribution_params_after,
        ).mean()
        self.actor.reset(hidden_state=hidden_state_before)
        if self.is_multi_gpu and torch.distributed.is_initialized():
            torch.distributed.all_reduce(policy_kl, op=torch.distributed.ReduceOp.SUM)
            policy_kl /= self.gpu_world_size
        return policy_kl.item()

    @staticmethod
    def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
        grad_norms = [param.grad.detach().norm(2) for param in parameters if param.grad is not None]
        if not grad_norms:
            return 0.0
        return torch.stack(grad_norms).norm(2).item()

    def train_mode(self) -> None:
        self.actor.train()

    def eval_mode(self) -> None:
        self.actor.eval()

    def save(self) -> dict:
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "student_optimizer_state_dict": self.student_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor") or load_cfg.get("critic"):
            key = "actor_state_dict" if "actor_state_dict" in loaded_dict else "critic_state_dict"
            self._raw_actor.load_state_dict(loaded_dict[key], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "student_optimizer_state_dict" in loaded_dict:
                self.student_optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])
            elif "proprio_optimizer_state_dict" in loaded_dict:
                self.student_optimizer.load_state_dict(loaded_dict["proprio_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> RepresentationVelocityActorCritic:
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = self.actor

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> RepresentationVelocityPredictorTeacherStudentPPO:
        alg_class: type[RepresentationVelocityPredictorTeacherStudentPPO] = resolve_callable(  # type: ignore
            cfg["algorithm"].pop("class_name")
        )
        model_class: type[RepresentationVelocityActorCritic] = resolve_callable(  # type: ignore
            cfg["actor"].pop("class_name")
        )

        default_sets = ["proprio_history", "actor_command", "lin_vel_target", "critic", "privileged_encoder"]
        if cfg["algorithm"].get("latent_dynamics_loss_coef", 0.0) > 0.0:
            cfg["actor"]["latent_dynamics_horizons"] = cfg["algorithm"].get(
                "latent_dynamics_horizons",
                (1,),
            )
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("RND is not supported by RepresentationVelocityPredictorTeacherStudentPPO.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError(
                "Symmetry augmentation is not supported by RepresentationVelocityPredictorTeacherStudentPPO."
            )
        cfg["algorithm"]["symmetry_cfg"] = None
        if cfg["algorithm"].get("share_cnn_encoders", False):
            raise ValueError(
                "CNN encoder sharing is not supported by RepresentationVelocityPredictorTeacherStudentPPO."
            )
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        model = model_class(obs, cfg["obs_groups"], env.num_actions, **cfg["actor"]).to(device)
        print(f"Representation Velocity Actor-Critic Model: {model}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg = alg_class(
            model,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def broadcast_parameters(self) -> None:
        model_params = [self._raw_actor.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])

    def reduce_parameters(self, parameters: Iterable[torch.nn.Parameter] | None = None) -> None:
        all_params = list(self.actor.ppo_parameters() if parameters is None else parameters)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
