# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: D102, D107

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config
from rsl_rl.models import RepresentationVelocityActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class RepresentationVelocityTeacherStudentPPO:
    """PPO on privileged latents plus delayed student velocity representation learning."""

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
        representation_chunk_length: int = 24,
        representation_loss_coef: float = 1.0,
        lin_vel_loss_coef: float = 1.0,
        roughness_loss_coef: float = 0.2,
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
            raise ValueError("RND is not supported by RepresentationVelocityTeacherStudentPPO.")
        if symmetry_cfg is not None:
            raise ValueError("Symmetry augmentation is not supported by RepresentationVelocityTeacherStudentPPO.")
        if share_cnn_encoders:
            raise ValueError("CNN encoder sharing is not supported by RepresentationVelocityTeacherStudentPPO.")
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
        self.roughness_loss_coef = roughness_loss_coef
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
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.actor.update_normalization(obs)
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

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

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
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor.ppo_parameters())
            nn.utils.clip_grad_norm_(self.actor.ppo_parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        mean_student_loss = 0.0
        mean_representation_loss = 0.0
        mean_lin_vel_loss = 0.0
        mean_roughness_loss = 0.0
        student_updates = 0
        if hasattr(self.actor, "compute_student_losses_sequence"):
            generator = self.storage.representation_chunk_generator(
                self.num_representation_mini_batches,
                self.num_representation_epochs,
                self.representation_chunk_length,
            )
            for batch in generator:
                for _ in range(self.num_student_substeps):
                    if hasattr(self.actor, "compute_student_losses_sequence_with_roughness"):
                        _, representation_loss, lin_vel_loss, roughness_loss = (
                            self.actor.compute_student_losses_sequence_with_roughness(
                                batch.observations,
                                batch.dones,
                                hidden_state=batch.hidden_states[0],
                            )
                        )
                    else:
                        _, representation_loss, lin_vel_loss = self.actor.compute_student_losses_sequence(
                            batch.observations,
                            batch.dones,
                            hidden_state=batch.hidden_states[0],
                        )
                        roughness_loss = representation_loss.new_zeros(())
                    student_loss = (
                        self.representation_loss_coef * representation_loss
                        + self.lin_vel_loss_coef * lin_vel_loss
                        + self.roughness_loss_coef * roughness_loss
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
                    mean_roughness_loss += roughness_loss.item()
                    student_updates += 1
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            for batch in generator:
                for _ in range(self.num_student_substeps):
                    if hasattr(self.actor, "compute_student_losses_with_roughness"):
                        _, representation_loss, lin_vel_loss, roughness_loss = self.actor.compute_student_losses_with_roughness(
                            batch.observations
                        )
                    else:
                        _, representation_loss, lin_vel_loss = self.actor.compute_student_losses(batch.observations)
                        roughness_loss = representation_loss.new_zeros(())
                    student_loss = (
                        self.representation_loss_coef * representation_loss
                        + self.lin_vel_loss_coef * lin_vel_loss
                        + self.roughness_loss_coef * roughness_loss
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
                    mean_roughness_loss += roughness_loss.item()
                    student_updates += 1

        num_ppo_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_ppo_updates,
            "surrogate": mean_surrogate_loss / num_ppo_updates,
            "entropy": mean_entropy / num_ppo_updates,
            "student": mean_student_loss / student_updates,
            "representation": mean_representation_loss / student_updates,
            "lin_vel": mean_lin_vel_loss / student_updates,
            "roughness": mean_roughness_loss / student_updates,
        }
        self.storage.clear()
        return loss_dict

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
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            if "student_optimizer_state_dict" in loaded_dict:
                self.student_optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])
                self.student_learning_rate = self.student_optimizer.param_groups[0]["lr"]
            elif "proprio_optimizer_state_dict" in loaded_dict:
                self.student_optimizer.load_state_dict(loaded_dict["proprio_optimizer_state_dict"])
                self.student_learning_rate = self.student_optimizer.param_groups[0]["lr"]
        return load_cfg.get("iteration", False)

    def get_policy(self) -> RepresentationVelocityActorCritic:
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = self.actor

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> RepresentationVelocityTeacherStudentPPO:
        alg_class: type[RepresentationVelocityTeacherStudentPPO] = resolve_callable(  # type: ignore
            cfg["algorithm"].pop("class_name")
        )
        model_class: type[RepresentationVelocityActorCritic] = resolve_callable(  # type: ignore
            cfg["actor"].pop("class_name")
        )

        default_sets = ["proprio_history", "actor_command", "lin_vel_target", "critic", "privileged_encoder"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("RND is not supported by RepresentationVelocityTeacherStudentPPO.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("Symmetry augmentation is not supported by RepresentationVelocityTeacherStudentPPO.")
        cfg["algorithm"]["symmetry_cfg"] = None
        if cfg["algorithm"].get("share_cnn_encoders", False):
            raise ValueError("CNN encoder sharing is not supported by RepresentationVelocityTeacherStudentPPO.")
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
