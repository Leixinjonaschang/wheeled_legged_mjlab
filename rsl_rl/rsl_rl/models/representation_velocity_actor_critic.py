# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: ANN201, D102, D107, N812

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


def _build_actor_obs(
    lin_vel_normalizer: nn.Module,
    current_proprio_normalizer: nn.Module,
    command_normalizer: nn.Module,
    predicted_lin_vel: torch.Tensor,
    current_proprio: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            lin_vel_normalizer(predicted_lin_vel),
            current_proprio_normalizer(current_proprio),
            command_normalizer(command),
        ),
        dim=-1,
    )


class RepresentationVelocityActorCritic(nn.Module):
    """Actor-critic with privileged and proprioceptive velocity representations."""

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
        (
            self.proprio_history_obs_groups,
            self.proprio_history_length,
            self.current_proprio_dim,
        ) = self._get_history_shape(obs, obs_groups, "proprio_history")
        self.command_obs_groups, self.command_dim = self._get_obs_dim(obs, obs_groups, "actor_command")
        self.lin_vel_target_obs_groups, self.lin_vel_dim = self._get_obs_dim(obs, obs_groups, "lin_vel_target")
        self.critic_obs_groups, self.critic_obs_dim = self._get_obs_dim(obs, obs_groups, "critic")
        self.privileged_encoder_obs_groups, self.privileged_encoder_obs_dim = self._get_obs_dim(
            obs, obs_groups, "privileged_encoder"
        )
        if self.lin_vel_dim != 3:
            raise ValueError(f"lin_vel_target must have dimension 3, got {self.lin_vel_dim}.")

        self.proprio_history_obs_dim = self.proprio_history_length * self.current_proprio_dim
        self.proprio_encoder_obs_dim = self.proprio_history_obs_dim + self.command_dim
        self.actor_obs_dim = self.lin_vel_dim + self.current_proprio_dim + self.command_dim
        self.obs_groups = self.proprio_history_obs_groups
        self.obs_dim = self.proprio_encoder_obs_dim
        self.latent_dim = latent_dim
        self.normalize_latent = normalize_latent

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.proprio_history_obs_normalizer = EmpiricalNormalization(self.proprio_history_obs_dim)
            self.current_proprio_obs_normalizer = EmpiricalNormalization(self.current_proprio_dim)
            self.command_obs_normalizer = EmpiricalNormalization(self.command_dim)
            self.lin_vel_normalizer = EmpiricalNormalization(self.lin_vel_dim)
            self.critic_obs_normalizer = EmpiricalNormalization(self.critic_obs_dim)
            self.privileged_obs_normalizer = EmpiricalNormalization(self.privileged_encoder_obs_dim)
        else:
            self.proprio_history_obs_normalizer = nn.Identity()
            self.current_proprio_obs_normalizer = nn.Identity()
            self.command_obs_normalizer = nn.Identity()
            self.lin_vel_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()
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
        encoder_feature_dim = self._encoder_feature_dim(encoder_hidden_dims)
        self.privileged_encoder = MLP(self.privileged_encoder_obs_dim, latent_dim, encoder_hidden_dims, activation)
        self.proprio_encoder = MLP(self.proprio_encoder_obs_dim, encoder_feature_dim, encoder_hidden_dims, activation)
        self.student_latent_head = nn.Linear(encoder_feature_dim, latent_dim)
        self.lin_vel_head = nn.Linear(encoder_feature_dim, self.lin_vel_dim)
        self.actor_head = MLP(self.actor_obs_dim + latent_dim, actor_output_dim, hidden_dims, activation)
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
        latent, predicted_lin_vel = self.get_proprio_outputs(obs)
        actor_obs = self.get_actor_obs_from_prediction(obs, predicted_lin_vel)
        return self._actor(actor_obs, latent, stochastic_output=stochastic_output)

    def act_teacher(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run PPO with predicted velocity actor inputs and privileged latent."""
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        with torch.no_grad():
            predicted_lin_vel = self.get_predicted_lin_vel(obs)
        actor_obs = self.get_actor_obs_from_prediction(obs, predicted_lin_vel)
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

    def compute_student_losses(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return combined student, latent-alignment, and velocity-regression losses."""
        proprio_latent, predicted_lin_vel = self.get_proprio_outputs(obs)
        with torch.no_grad():
            privileged_latent = self.get_privileged_latent(obs)
        representation_loss = F.mse_loss(proprio_latent, privileged_latent)
        lin_vel_loss = F.mse_loss(predicted_lin_vel, self.get_lin_vel_target(obs))
        return representation_loss + lin_vel_loss, representation_loss, lin_vel_loss

    def compute_representation_loss(self, obs: TensorDict) -> torch.Tensor:
        return self.compute_student_losses(obs)[1]

    def compute_lin_vel_loss(self, obs: TensorDict) -> torch.Tensor:
        return self.compute_student_losses(obs)[2]

    def ppo_parameters(self):
        """Yield parameters optimized by PPO."""
        yield from self.privileged_encoder.parameters()
        yield from self.actor_head.parameters()
        yield from self.critic_head.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def student_parameters(self):
        """Yield parameters optimized by student representation learning."""
        yield from self.proprio_encoder.parameters()
        yield from self.student_latent_head.parameters()
        yield from self.lin_vel_head.parameters()

    def proprio_parameters(self):
        """Compatibility alias for student encoder parameters."""
        yield from self.student_parameters()

    def get_actor_obs_from_prediction(self, obs: TensorDict, predicted_lin_vel: torch.Tensor) -> torch.Tensor:
        return _build_actor_obs(
            self.lin_vel_normalizer,
            self.current_proprio_obs_normalizer,
            self.command_obs_normalizer,
            predicted_lin_vel,
            self.get_current_proprio(obs),
            self.get_command(obs),
        )

    def get_current_proprio(self, obs: TensorDict) -> torch.Tensor:
        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        return proprio_history[:, -1, :]

    def get_command(self, obs: TensorDict) -> torch.Tensor:
        return self._cat_obs(obs, self.command_obs_groups)

    def get_lin_vel_target(self, obs: TensorDict) -> torch.Tensor:
        return self._cat_obs(obs, self.lin_vel_target_obs_groups)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_obs_normalizer(self._cat_obs(obs, self.critic_obs_groups))

    def get_proprio_obs(self, obs: TensorDict) -> torch.Tensor:
        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        proprio_obs = self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=1))
        return torch.cat((proprio_obs, self.command_obs_normalizer(self.get_command(obs))), dim=-1)

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.privileged_obs_normalizer(self._cat_obs(obs, self.privileged_encoder_obs_groups))

    def get_proprio_outputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.proprio_encoder(self.get_proprio_obs(obs))
        latent = self.student_latent_head(features)
        predicted_lin_vel = self.lin_vel_head(features)
        return self._normalize_latent(latent), predicted_lin_vel

    def get_proprio_latent(self, obs: TensorDict) -> torch.Tensor:
        return self.get_proprio_outputs(obs)[0]

    def get_predicted_lin_vel(self, obs: TensorDict) -> torch.Tensor:
        return self.get_proprio_outputs(obs)[1]

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
        return _TorchRepresentationVelocityActorCritic(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxRepresentationVelocityActorCritic(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
            self.proprio_history_obs_normalizer.update(proprio_history.flatten(start_dim=1))  # type: ignore
            self.current_proprio_obs_normalizer.update(proprio_history[:, -1, :])  # type: ignore
            self.command_obs_normalizer.update(self.get_command(obs))  # type: ignore
            self.lin_vel_normalizer.update(self.get_lin_vel_target(obs))  # type: ignore
            self.critic_obs_normalizer.update(self._cat_obs(obs, self.critic_obs_groups))  # type: ignore
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
                    "Proprio history observations must have shape (batch, history, features), "
                    f"got {group_obs.shape} for '{obs_group}'."
                )
            if history_length is None:
                history_length = group_obs.shape[-2]
            elif group_obs.shape[-2] != history_length:
                raise ValueError(
                    "All proprio history groups must use the same history length, "
                    f"got {history_length} and {group_obs.shape[-2]}."
                )
            frame_dim += group_obs.shape[-1]
        if history_length is None:
            raise ValueError("At least one proprio history observation group is required.")
        return active_obs_groups, history_length, frame_dim

    @staticmethod
    def _encoder_feature_dim(encoder_hidden_dims: tuple[int, ...] | list[int]) -> int:
        if len(encoder_hidden_dims) == 0:
            raise ValueError("encoder_hidden_dims must contain at least one dimension.")
        return encoder_hidden_dims[-1]


class _TorchRepresentationVelocityActorCritic(nn.Module):
    """TorchScript wrapper for student velocity policy inference."""

    def __init__(self, model: RepresentationVelocityActorCritic) -> None:
        super().__init__()
        self.proprio_history_obs_normalizer = copy.deepcopy(model.proprio_history_obs_normalizer)
        self.current_proprio_obs_normalizer = copy.deepcopy(model.current_proprio_obs_normalizer)
        self.command_obs_normalizer = copy.deepcopy(model.command_obs_normalizer)
        self.lin_vel_normalizer = copy.deepcopy(model.lin_vel_normalizer)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.student_latent_head = copy.deepcopy(model.student_latent_head)
        self.lin_vel_head = copy.deepcopy(model.lin_vel_head)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, proprio_history: torch.Tensor, actor_command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio_obs = self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=1))
        student_obs = torch.cat((proprio_obs, self.command_obs_normalizer(actor_command)), dim=-1)
        features = self.proprio_encoder(student_obs)
        latent = self.student_latent_head(features)
        predicted_lin_vel = self.lin_vel_head(features)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        actor_obs = _build_actor_obs(
            self.lin_vel_normalizer,
            self.current_proprio_obs_normalizer,
            self.command_obs_normalizer,
            predicted_lin_vel,
            proprio_history[:, -1, :],
            actor_command,
        )
        out = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        return self.deterministic_output(out), predicted_lin_vel

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxRepresentationVelocityActorCritic(nn.Module):
    """ONNX wrapper for student velocity policy inference."""

    is_recurrent: bool = False

    def __init__(self, model: RepresentationVelocityActorCritic, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.proprio_history_obs_normalizer = copy.deepcopy(model.proprio_history_obs_normalizer)
        self.current_proprio_obs_normalizer = copy.deepcopy(model.current_proprio_obs_normalizer)
        self.command_obs_normalizer = copy.deepcopy(model.command_obs_normalizer)
        self.lin_vel_normalizer = copy.deepcopy(model.lin_vel_normalizer)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.student_latent_head = copy.deepcopy(model.student_latent_head)
        self.lin_vel_head = copy.deepcopy(model.lin_vel_head)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.history_length = model.proprio_history_length
        self.proprio_input_size = model.current_proprio_dim
        self.command_input_size = model.command_dim

    def forward(self, proprio_history: torch.Tensor, actor_command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio_obs = self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=1))
        student_obs = torch.cat((proprio_obs, self.command_obs_normalizer(actor_command)), dim=-1)
        features = self.proprio_encoder(student_obs)
        latent = self.student_latent_head(features)
        predicted_lin_vel = self.lin_vel_head(features)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        actor_obs = _build_actor_obs(
            self.lin_vel_normalizer,
            self.current_proprio_obs_normalizer,
            self.command_obs_normalizer,
            predicted_lin_vel,
            proprio_history[:, -1, :],
            actor_command,
        )
        out = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        return self.deterministic_output(out), predicted_lin_vel

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.history_length, self.proprio_input_size),
            torch.zeros(1, self.command_input_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["proprio_history", "actor_command"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "predicted_lin_vel"]
