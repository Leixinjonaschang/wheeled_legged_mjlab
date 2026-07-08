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

from rsl_rl.models.representation_velocity_actor_critic import (
    RepresentationVelocityActorCritic,
)
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories


class DepthRepresentationVelocityActorCritic(RepresentationVelocityActorCritic):
    """Velocity representation model with a recurrent depth-image student input."""

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
        depth_feature_dim: int = 64,
        depth_gru_hidden_dim: int = 64,
        depth_channels: tuple[int, ...] | list[int] = (16, 32, 32),
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            output_dim,
            hidden_dims=hidden_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            obs_normalization=obs_normalization,
            normalize_latent=normalize_latent,
            distribution_cfg=distribution_cfg,
        )
        self.depth_obs_group, self.depth_shape = self._get_depth_group_and_shape(obs, obs_groups, "depth_encoder")
        self.depth_feature_dim = depth_feature_dim
        self.depth_gru_hidden_dim = depth_gru_hidden_dim
        self.depth_encoder = _DepthCNN(
            input_shape=self.depth_shape,
            channels=depth_channels,
            output_dim=depth_feature_dim,
            activation=activation,
        )
        self.depth_gru = nn.GRUCell(depth_feature_dim, depth_gru_hidden_dim)

        encoder_hidden_dims = hidden_dims if encoder_hidden_dims is None else encoder_hidden_dims
        encoder_feature_dim = self._encoder_feature_dim(encoder_hidden_dims)
        self.proprio_depth_encoder_obs_dim = self.proprio_encoder_obs_dim + depth_gru_hidden_dim
        self.proprio_encoder = MLP(
            self.proprio_depth_encoder_obs_dim,
            encoder_feature_dim,
            encoder_hidden_dims,
            activation,
        )
        self._student_hidden_state: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run the deployable student policy path."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent, predicted_lin_vel, _ = self._get_student_outputs(
            obs,
            hidden_state,
            update_hidden_state=hidden_state is None,
            use_internal_state=True,
        )
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
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        with torch.no_grad():
            _, predicted_lin_vel, _ = self._get_student_outputs(
                obs,
                hidden_state,
                update_hidden_state=hidden_state is None,
                use_internal_state=True,
            )
        actor_obs = self.get_actor_obs_from_prediction(obs, predicted_lin_vel)
        latent = self.get_privileged_latent(obs)
        return self._actor(actor_obs, latent, stochastic_output=stochastic_output)

    def compute_student_losses(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return combined student, latent-alignment, and velocity-regression losses."""
        proprio_latent, predicted_lin_vel, _ = self._get_student_outputs(obs)
        with torch.no_grad():
            privileged_latent = self.get_privileged_latent(obs)
        representation_loss = F.mse_loss(proprio_latent, privileged_latent)
        lin_vel_loss = F.mse_loss(predicted_lin_vel, self.get_lin_vel_target(obs))
        return representation_loss + lin_vel_loss, representation_loss, lin_vel_loss

    def compute_student_losses_sequence(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        hidden_state: HiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute student losses over a continuous rollout chunk."""
        if len(obs.batch_size) != 2:
            raise ValueError(f"Expected sequence observations with [time, batch], got {obs.batch_size}")
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("Depth velocity representation expects a tensor GRU hidden state")

        time_steps, batch_size = obs.batch_size
        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        proprio_obs = self.proprio_history_obs_normalizer(
            self._flatten_proprio_history(proprio_history)
        )
        depth_latent, _ = self._encode_depth_sequence(obs[self.depth_obs_group], dones, hidden_state)
        student_input = torch.cat((proprio_obs, depth_latent), dim=-1)
        features = self.proprio_encoder(student_input.flatten(0, 1))
        latent = self.student_latent_head(features).view(time_steps, batch_size, self.latent_dim)
        predicted_lin_vel = self.lin_vel_head(features).view(time_steps, batch_size, self.lin_vel_dim)
        latent = self._normalize_latent(latent)

        privileged_obs = self.privileged_obs_normalizer(
            self._cat_obs(obs, self.privileged_encoder_obs_groups)
        )
        with torch.no_grad():
            privileged_latent = self._normalize_latent(
                self.privileged_encoder(privileged_obs.flatten(0, 1))
            ).view(time_steps, batch_size, self.latent_dim)
        representation_loss = F.mse_loss(latent, privileged_latent)
        lin_vel_loss = F.mse_loss(predicted_lin_vel, self.get_lin_vel_target(obs))
        return representation_loss + lin_vel_loss, representation_loss, lin_vel_loss

    def student_parameters(self):
        """Yield parameters optimized by student representation learning."""
        yield from self.depth_encoder.parameters()
        yield from self.depth_gru.parameters()
        yield from super().student_parameters()

    def get_proprio_outputs(
        self,
        obs: TensorDict,
        hidden_state: HiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, predicted_lin_vel, _ = self._get_student_outputs(obs, hidden_state)
        return latent, predicted_lin_vel

    def get_depth_obs(self, obs: TensorDict) -> torch.Tensor:
        if self.depth_obs_group not in obs:
            raise ValueError(f"missing depth observation group '{self.depth_obs_group}'")
        depth = obs[self.depth_obs_group]
        expected_shape = (*obs.batch_size, *self.depth_shape)
        if tuple(depth.shape) != expected_shape:
            raise ValueError(f"expected depth shape {expected_shape}, got {tuple(depth.shape)}")
        return depth

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self._student_hidden_state = None if hidden_state is None else hidden_state.detach()
        elif hidden_state is not None:
            raise NotImplementedError("Resetting done environments with a custom hidden state is not supported")
        elif self._student_hidden_state is not None:
            done_mask = dones.to(device=self._student_hidden_state.device, dtype=torch.bool).view(-1)
            self._student_hidden_state[done_mask] = 0.0

    def get_hidden_state(self) -> HiddenState:
        return self._student_hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._student_hidden_state is None:
            return
        if dones is None:
            self._student_hidden_state = self._student_hidden_state.detach()
            return
        done_mask = dones.to(device=self._student_hidden_state.device, dtype=torch.bool).view(-1)
        self._student_hidden_state[done_mask] = self._student_hidden_state[done_mask].detach()

    def as_jit(self) -> nn.Module:
        return _TorchDepthRepresentationVelocityActorCritic(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxDepthRepresentationVelocityActorCritic(self, verbose)

    def _get_student_outputs(
        self,
        obs: TensorDict,
        hidden_state: HiddenState = None,
        *,
        update_hidden_state: bool = False,
        use_internal_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("Depth velocity representation expects a tensor GRU hidden state")
        active_hidden_state = (
            self._student_hidden_state
            if use_internal_state and hidden_state is None
            else hidden_state
        )
        depth_latent, next_hidden_state = self._encode_depth(
            self.get_depth_obs(obs),
            active_hidden_state,
        )
        if update_hidden_state:
            self._student_hidden_state = next_hidden_state.detach()
        proprio_depth_obs = torch.cat((self.get_proprio_obs(obs), depth_latent), dim=-1)
        features = self.proprio_encoder(proprio_depth_obs)
        latent = self.student_latent_head(features)
        predicted_lin_vel = self.lin_vel_head(features)
        return self._normalize_latent(latent), predicted_lin_vel, next_hidden_state

    def _encode_depth(
        self,
        depth: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth.ndim != 4:
            raise ValueError(f"depth must have shape [batch, C, H, W], got {tuple(depth.shape)}")
        batch_size = depth.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size,
                self.depth_gru_hidden_dim,
                device=depth.device,
                dtype=depth.dtype,
            )
        expected_hidden_shape = (batch_size, self.depth_gru_hidden_dim)
        if tuple(hidden_state.shape) != expected_hidden_shape:
            raise ValueError(f"expected hidden_state shape {expected_hidden_shape}, got {tuple(hidden_state.shape)}")
        next_hidden_state = self.depth_gru(self.depth_encoder(depth), hidden_state)
        return next_hidden_state, next_hidden_state

    def _encode_depth_sequence(
        self,
        depth: torch.Tensor,
        dones: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth.ndim != 5:
            raise ValueError(f"depth sequence must have shape [time, batch, C, H, W], got {tuple(depth.shape)}")
        if tuple(depth.shape[2:]) != self.depth_shape:
            raise ValueError(f"expected depth shape {self.depth_shape}, got {tuple(depth.shape[2:])}")
        if tuple(dones.shape[:2]) != tuple(depth.shape[:2]):
            raise ValueError(f"dones must share [time, batch] with depth, got {tuple(dones.shape)}")

        batch_size = depth.shape[1]
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size,
                self.depth_gru_hidden_dim,
                device=depth.device,
                dtype=depth.dtype,
            )
        latents = []
        for step in range(depth.shape[0]):
            latent, hidden_state = self._encode_depth(depth[step], hidden_state)
            latents.append(latent)
            done_mask = dones[step].to(device=hidden_state.device, dtype=torch.bool).view(-1, 1)
            hidden_state = torch.where(done_mask, torch.zeros_like(hidden_state), hidden_state)
        return torch.stack(latents), hidden_state

    @staticmethod
    def _flatten_proprio_history(proprio_history: torch.Tensor) -> torch.Tensor:
        if proprio_history.ndim == 3:
            return proprio_history.flatten(start_dim=1)
        if proprio_history.ndim == 4:
            return proprio_history.flatten(start_dim=2)
        raise ValueError(f"proprio history must have 3 or 4 dims, got {tuple(proprio_history.shape)}")

    @staticmethod
    def _get_depth_group_and_shape(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[str, tuple[int, int, int]]:
        active_obs_groups = obs_groups[obs_set]
        if len(active_obs_groups) != 1:
            raise ValueError(f"'{obs_set}' must contain exactly one depth observation group, got {active_obs_groups}")
        obs_group = active_obs_groups[0]
        if len(obs[obs_group].shape) != 4:
            raise ValueError(
                f"Depth observation '{obs_group}' must have shape [batch, C, H, W], got {obs[obs_group].shape}"
            )
        return obs_group, tuple(obs[obs_group].shape[1:])


class _DepthCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channels: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (channels, height, width), got {input_shape}")
        if any(dim <= 0 for dim in input_shape):
            raise ValueError(f"input_shape dimensions must be positive, got {input_shape}")
        if not channels:
            raise ValueError("depth_channels must contain at least one output channel")

        activation_cls = _resolve_activation(activation)
        in_channels = input_shape[0]
        layers: list[nn.Module] = []
        for idx, out_channels in enumerate(channels):
            kernel_size = 5 if idx == 0 else 3
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2))
            layers.append(activation_cls())
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = int(self.cnn(dummy).flatten(start_dim=1).shape[1])

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, output_dim),
            activation_cls(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.projection(self.cnn(depth))


class _TorchDepthRepresentationVelocityActorCritic(nn.Module):
    """TorchScript wrapper for stateful student depth policy inference."""

    def __init__(self, model: DepthRepresentationVelocityActorCritic) -> None:
        super().__init__()
        self.proprio_history_obs_normalizer = copy.deepcopy(model.proprio_history_obs_normalizer)
        self.current_proprio_obs_normalizer = copy.deepcopy(model.current_proprio_obs_normalizer)
        self.command_obs_normalizer = copy.deepcopy(model.command_obs_normalizer)
        self.lin_vel_normalizer = copy.deepcopy(model.lin_vel_normalizer)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.depth_gru = copy.deepcopy(model.depth_gru)
        self.student_latent_head = copy.deepcopy(model.student_latent_head)
        self.lin_vel_head = copy.deepcopy(model.lin_vel_head)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(1, model.depth_gru_hidden_dim))

    def forward(
        self,
        proprio_history: torch.Tensor,
        actor_command: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proprio_obs = self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=1))
        depth_latent = self.depth_gru(self.depth_encoder(depth), self.hidden_state)
        self.hidden_state[:] = depth_latent.detach()
        return self._actor_from_student_inputs(proprio_history, actor_command, proprio_obs, depth_latent)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0

    def _actor_from_student_inputs(
        self,
        proprio_history: torch.Tensor,
        actor_command: torch.Tensor,
        proprio_obs: torch.Tensor,
        depth_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.proprio_encoder(torch.cat((proprio_obs, depth_latent), dim=-1))
        latent = self.student_latent_head(features)
        predicted_lin_vel = self.lin_vel_head(features)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        actor_obs = torch.cat(
            (
                self.lin_vel_normalizer(predicted_lin_vel),
                self.current_proprio_obs_normalizer(proprio_history[:, -1, :]),
                self.command_obs_normalizer(actor_command),
            ),
            dim=-1,
        )
        out = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        return self.deterministic_output(out), predicted_lin_vel


class _OnnxDepthRepresentationVelocityActorCritic(_TorchDepthRepresentationVelocityActorCritic):
    """ONNX wrapper for state-explicit student depth policy inference."""

    is_recurrent: bool = True

    def __init__(self, model: DepthRepresentationVelocityActorCritic, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.use_external_data = False
        self.history_length = model.proprio_history_length
        self.proprio_input_size = model.current_proprio_dim
        self.command_input_size = model.command_dim
        self.depth_input_shape = model.depth_shape
        self.hidden_size = model.depth_gru_hidden_dim

    def forward(
        self,
        proprio_history: torch.Tensor,
        actor_command: torch.Tensor,
        depth: torch.Tensor,
        hidden_state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_obs = self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=1))
        hidden_state_out = self.depth_gru(self.depth_encoder(depth), hidden_state_in)
        actions, predicted_lin_vel = self._actor_from_student_inputs(
            proprio_history,
            actor_command,
            proprio_obs,
            hidden_state_out,
        )
        return actions, predicted_lin_vel, hidden_state_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.history_length, self.proprio_input_size),
            torch.zeros(1, self.command_input_size),
            torch.zeros(1, *self.depth_input_shape),
            torch.zeros(1, self.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["proprio_history", "actor_command", "depth", "hidden_state_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "predicted_lin_vel", "hidden_state_out"]


def _resolve_activation(name: str) -> type[nn.Module]:
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation for depth encoder: {name}")
