# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class RepresentationActorCritic(nn.Module):
    """Actor-critic with privileged and proprioceptive representation encoders."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] | None = None,
        latent_dim: int = 32,
        activation: str = "elu",
        obs_normalization: bool = False,
        normalize_latent: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.teacher_actor_obs_groups, self.teacher_actor_obs_dim = self._get_obs_dim(
            obs, obs_groups, "teacher_actor"
        )
        self.critic_obs_groups, self.critic_obs_dim = self._get_obs_dim(obs, obs_groups, "critic")
        (
            self.student_history_obs_groups,
            self.student_history_length,
            self.student_actor_obs_dim,
        ) = self._get_history_shape(obs, obs_groups, "student_history")
        self.proprio_encoder_obs_dim = self.student_history_length * self.student_actor_obs_dim
        if self.student_actor_obs_dim != self.teacher_actor_obs_dim:
            raise ValueError(
                "Student history frame dimension must match the teacher actor observation dimension, "
                f"got {self.student_actor_obs_dim} and {self.teacher_actor_obs_dim}."
            )
        self.privileged_encoder_obs_groups, self.privileged_encoder_obs_dim = self._get_obs_dim(
            obs, obs_groups, "privileged_encoder"
        )
        self.obs_groups = self.student_history_obs_groups
        self.obs_dim = self.proprio_encoder_obs_dim
        self.latent_dim = latent_dim
        self.normalize_latent = normalize_latent

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.teacher_actor_obs_normalizer = EmpiricalNormalization(self.teacher_actor_obs_dim)
            self.student_actor_obs_normalizer = EmpiricalNormalization(self.student_actor_obs_dim)
            self.critic_obs_normalizer = EmpiricalNormalization(self.critic_obs_dim)
            self.proprio_obs_normalizer = EmpiricalNormalization(self.proprio_encoder_obs_dim)
            self.privileged_obs_normalizer = EmpiricalNormalization(self.privileged_encoder_obs_dim)
        else:
            self.teacher_actor_obs_normalizer = nn.Identity()
            self.student_actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()
            self.proprio_obs_normalizer = nn.Identity()
            self.privileged_obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        encoder_hidden_dims = hidden_dims if encoder_hidden_dims is None else encoder_hidden_dims
        self.privileged_encoder = MLP(self.privileged_encoder_obs_dim, latent_dim, encoder_hidden_dims, activation)
        self.proprio_encoder = MLP(self.proprio_encoder_obs_dim, latent_dim, encoder_hidden_dims, activation)
        self.actor_head = MLP(self.teacher_actor_obs_dim + latent_dim, actor_output_dim, hidden_dims, activation)
        self.critic_head = MLP(self.critic_obs_dim + latent_dim, 1, hidden_dims, activation)

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_head)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run the deployable student policy path."""
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        actor_obs = self.get_student_actor_obs(obs)
        latent = self.get_proprio_latent(obs)
        return self._actor(actor_obs, latent, stochastic_output=stochastic_output)

    def act_teacher(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run PPO updates with deployable actor observations and privileged latent."""
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        actor_obs = self.get_student_actor_obs(obs)
        latent = self.get_privileged_latent(obs)
        return self._actor(actor_obs, latent, stochastic_output=stochastic_output)

    def evaluate_teacher(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Evaluate the critic with the privileged latent."""
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        critic_obs = self.get_critic_obs(obs)
        latent = self.get_privileged_latent(obs)
        return self.critic_head(torch.cat((critic_obs, latent), dim=-1))

    def compute_representation_loss(self, obs: TensorDict) -> torch.Tensor:
        """Align proprioceptive latents to detached privileged latents."""
        proprio_latent = self.get_proprio_latent(obs)
        with torch.no_grad():
            privileged_latent = self.get_privileged_latent(obs)
        return (1.0 - F.cosine_similarity(proprio_latent, privileged_latent, dim=-1)).mean()

    def ppo_parameters(self):
        """Yield parameters optimized by PPO."""
        yield from self.privileged_encoder.parameters()
        yield from self.actor_head.parameters()
        yield from self.critic_head.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def proprio_parameters(self):
        """Yield parameters optimized by representation alignment."""
        yield from self.proprio_encoder.parameters()

    def get_teacher_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.teacher_actor_obs_normalizer(self._cat_obs(obs, self.teacher_actor_obs_groups))

    def get_student_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        student_history = self._cat_obs(obs, self.student_history_obs_groups)
        return self.student_actor_obs_normalizer(student_history[:, -1, :])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_obs_normalizer(self._cat_obs(obs, self.critic_obs_groups))

    def get_proprio_obs(self, obs: TensorDict) -> torch.Tensor:
        student_history = self._cat_obs(obs, self.student_history_obs_groups)
        return self.proprio_obs_normalizer(student_history.flatten(start_dim=1))

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.privileged_obs_normalizer(self._cat_obs(obs, self.privileged_encoder_obs_groups))

    def get_proprio_latent(self, obs: TensorDict) -> torch.Tensor:
        latent = self.proprio_encoder(self.get_proprio_obs(obs))
        return self._normalize_latent(latent)

    def get_privileged_latent(self, obs: TensorDict) -> torch.Tensor:
        latent = self.privileged_encoder(self.get_privileged_obs(obs))
        return self._normalize_latent(latent)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _TorchRepresentationActorCritic(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxRepresentationActorCritic(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            student_history = self._cat_obs(obs, self.student_history_obs_groups)
            self.teacher_actor_obs_normalizer.update(  # type: ignore
                self._cat_obs(obs, self.teacher_actor_obs_groups)
            )
            self.student_actor_obs_normalizer.update(student_history[:, -1, :])  # type: ignore
            self.critic_obs_normalizer.update(self._cat_obs(obs, self.critic_obs_groups))  # type: ignore
            self.proprio_obs_normalizer.update(student_history.flatten(start_dim=1))  # type: ignore
            self.privileged_obs_normalizer.update(self._cat_obs(obs, self.privileged_encoder_obs_groups))  # type: ignore

    def _actor(self, actor_obs: torch.Tensor, latent: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        mlp_output = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(latent, p=2.0, dim=-1) if self.normalize_latent else latent

    def _cat_obs(self, obs: TensorDict, obs_groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in obs_groups], dim=-1)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The representation model only supports 1D observations, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_history_shape(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int, int]:
        active_obs_groups = obs_groups[obs_set]
        history_length: int | None = None
        frame_dim = 0
        for obs_group in active_obs_groups:
            group_obs = obs[obs_group]
            if len(group_obs.shape) != 3:
                raise ValueError(
                    "Student history observations must have shape (batch, history, features), "
                    f"got {group_obs.shape} for '{obs_group}'."
                )
            if history_length is None:
                history_length = group_obs.shape[-2]
            elif group_obs.shape[-2] != history_length:
                raise ValueError(
                    "All student history groups must use the same history length, "
                    f"got {history_length} and {group_obs.shape[-2]}."
                )
            frame_dim += group_obs.shape[-1]
        if history_length is None:
            raise ValueError("At least one student history observation group is required.")
        return active_obs_groups, history_length, frame_dim


class _TorchRepresentationActorCritic(nn.Module):
    """TorchScript wrapper for student policy inference."""

    def __init__(self, model: RepresentationActorCritic) -> None:
        super().__init__()
        self.student_actor_obs_normalizer = copy.deepcopy(model.student_actor_obs_normalizer)
        self.proprio_obs_normalizer = copy.deepcopy(model.proprio_obs_normalizer)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, student_history: torch.Tensor) -> torch.Tensor:
        actor_obs = self.student_actor_obs_normalizer(student_history[:, -1, :])
        proprio_obs = self.proprio_obs_normalizer(student_history.flatten(start_dim=1))
        latent = self.proprio_encoder(proprio_obs)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        out = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxRepresentationActorCritic(nn.Module):
    """ONNX wrapper for student policy inference."""

    is_recurrent: bool = False

    def __init__(self, model: RepresentationActorCritic, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.student_actor_obs_normalizer = copy.deepcopy(model.student_actor_obs_normalizer)
        self.proprio_obs_normalizer = copy.deepcopy(model.proprio_obs_normalizer)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.history_length = model.student_history_length
        self.actor_input_size = model.student_actor_obs_dim

    def forward(self, student_history: torch.Tensor) -> torch.Tensor:
        actor_obs = self.student_actor_obs_normalizer(student_history[:, -1, :])
        proprio_obs = self.proprio_obs_normalizer(student_history.flatten(start_dim=1))
        latent = self.proprio_encoder(proprio_obs)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        out = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.history_length, self.actor_input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["student_history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
