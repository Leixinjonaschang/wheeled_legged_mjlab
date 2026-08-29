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


class DepthPreprocessor(nn.Module):
    """Convert finite metric depth into normalized camera-range values."""

    def __init__(self, depth_min_m: float = 0.2, depth_max_m: float = 2.0) -> None:
        """Initialize the metric clipping range."""
        super().__init__()
        if depth_min_m < 0.0:
            raise ValueError(f"depth_min_m must be >= 0, got {depth_min_m}")
        if depth_max_m <= depth_min_m:
            raise ValueError(
                f"depth_max_m must be greater than depth_min_m, got {depth_max_m}"
            )
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m

    def forward(self, depth_m: torch.Tensor) -> torch.Tensor:
        """Treat depths below the near limit as empty, clip, and normalize."""
        depth_m = torch.where(
            depth_m >= self.depth_min_m,
            depth_m,
            torch.full_like(depth_m, self.depth_max_m),
        )
        depth_m = torch.clamp(depth_m, self.depth_min_m, self.depth_max_m)
        return (depth_m - self.depth_min_m) / (
            self.depth_max_m - self.depth_min_m
        )


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
        depth_gru_hidden_dim: int = 128,
        depth_channels: tuple[int, ...] | list[int] = (16, 32, 32),
        depth_conv_strides: tuple[int, ...] | list[int] | None = None,
        depth_min_m: float = 0.2,
        depth_max_m: float = 2.0,
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
        self.wheel_roughness_group = self._get_optional_wheel_roughness_group(obs, obs_groups)
        self.depth_feature_dim = depth_feature_dim
        self.depth_gru_input_dim = depth_feature_dim + self.current_proprio_dim
        self.depth_gru_hidden_dim = depth_gru_hidden_dim
        self.depth_preprocessor = DepthPreprocessor(depth_min_m, depth_max_m)
        if depth_conv_strides is None:
            depth_conv_strides = (*((2,) * (len(depth_channels) - 1)), 1)
        self.depth_encoder = _DepthCNN(
            input_shape=self.depth_shape,
            channels=depth_channels,
            output_dim=depth_feature_dim,
            activation=activation,
            strides=depth_conv_strides,
        )
        self.depth_gru = nn.GRUCell(self.depth_gru_input_dim, depth_gru_hidden_dim)

        encoder_hidden_dims = hidden_dims if encoder_hidden_dims is None else encoder_hidden_dims
        encoder_feature_dim = self._encoder_feature_dim(encoder_hidden_dims)
        self.proprio_depth_encoder_obs_dim = self.proprio_encoder_obs_dim + depth_gru_hidden_dim
        self.proprio_encoder = MLP(
            self.proprio_depth_encoder_obs_dim,
            encoder_feature_dim,
            encoder_hidden_dims,
            activation,
        )
        self.wheel_roughness_dim = 2
        self.wheel_roughness_head = nn.Linear(encoder_feature_dim, self.wheel_roughness_dim)
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
        total_loss, representation_loss, lin_vel_loss, _ = self.compute_student_losses_with_roughness(obs)
        return total_loss, representation_loss, lin_vel_loss

    def compute_student_losses_with_roughness(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return student losses, including wheel roughness when configured."""
        proprio_latent, predicted_lin_vel, predicted_wheel_roughness, _ = self._get_student_outputs_with_roughness(obs)
        with torch.no_grad():
            privileged_latent = self.get_privileged_latent(obs)
        representation_loss = (1.0 - F.cosine_similarity(proprio_latent, privileged_latent, dim=-1)).mean()
        lin_vel_loss = F.mse_loss(predicted_lin_vel, self.get_lin_vel_target(obs))
        roughness_loss = (
            F.smooth_l1_loss(predicted_wheel_roughness, self.get_wheel_roughness(obs))
            if self.wheel_roughness_group is not None
            else representation_loss.new_zeros(())
        )
        return representation_loss + lin_vel_loss + roughness_loss, representation_loss, lin_vel_loss, roughness_loss

    def compute_student_losses_sequence(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        hidden_state: HiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute student losses over a continuous rollout chunk."""
        total_loss, representation_loss, lin_vel_loss, _ = self.compute_student_losses_sequence_with_roughness(
            obs,
            dones,
            hidden_state,
        )
        return total_loss, representation_loss, lin_vel_loss

    def compute_student_losses_sequence_with_roughness(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        hidden_state: HiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all student losses over a recurrent rollout chunk."""
        if len(obs.batch_size) != 2:
            raise ValueError(f"Expected sequence observations with [time, batch], got {obs.batch_size}")
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("Depth velocity representation expects a tensor GRU hidden state")

        time_steps, batch_size = obs.batch_size
        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        proprio_obs = self.proprio_history_obs_normalizer(
            self._flatten_proprio_history(proprio_history)
        )
        current_proprio = self.current_proprio_obs_normalizer(
            proprio_history[:, :, -1, :].flatten(0, 1)
        ).view(time_steps, batch_size, self.current_proprio_dim)
        depth_latent, _ = self._encode_depth_sequence(
            obs[self.depth_obs_group],
            current_proprio,
            dones,
            hidden_state,
        )
        student_input = torch.cat((proprio_obs, depth_latent), dim=-1)
        features = self.proprio_encoder(student_input.flatten(0, 1))
        latent = self.student_latent_head(features).view(time_steps, batch_size, self.latent_dim)
        predicted_lin_vel = self.lin_vel_head(features).view(time_steps, batch_size, self.lin_vel_dim)
        predicted_wheel_roughness = torch.sigmoid(
            self.wheel_roughness_head(features)
        ).view(time_steps, batch_size, self.wheel_roughness_dim)
        latent = self._normalize_latent(latent)

        privileged_obs = self.privileged_obs_normalizer(
            self._cat_obs(obs, self.privileged_encoder_obs_groups)
        )
        with torch.no_grad():
            privileged_latent = self._normalize_latent(
                self.privileged_encoder(privileged_obs.flatten(0, 1))
            ).view(time_steps, batch_size, self.latent_dim)
        representation_loss = (1.0 - F.cosine_similarity(latent, privileged_latent, dim=-1)).mean()
        lin_vel_loss = F.mse_loss(predicted_lin_vel, self.get_lin_vel_target(obs))
        roughness_loss = (
            F.smooth_l1_loss(predicted_wheel_roughness, self.get_wheel_roughness(obs))
            if self.wheel_roughness_group is not None
            else representation_loss.new_zeros(())
        )
        return representation_loss + lin_vel_loss + roughness_loss, representation_loss, lin_vel_loss, roughness_loss

    def student_parameters(self):
        """Yield parameters optimized by student representation learning."""
        yield from self.depth_encoder.parameters()
        yield from self.depth_gru.parameters()
        yield from self.wheel_roughness_head.parameters()
        yield from super().student_parameters()

    def get_proprio_outputs(
        self,
        obs: TensorDict,
        hidden_state: HiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, predicted_lin_vel, _ = self._get_student_outputs(obs, hidden_state)
        return latent, predicted_lin_vel

    def get_predicted_lin_vel(self, obs: TensorDict) -> torch.Tensor:
        """Preview the recurrent velocity estimate without advancing the internal GRU state."""
        _, predicted_lin_vel, _ = self._get_student_outputs(
            obs,
            update_hidden_state=False,
            use_internal_state=True,
        )
        return predicted_lin_vel

    def get_depth_obs(self, obs: TensorDict) -> torch.Tensor:
        if self.depth_obs_group not in obs:
            raise ValueError(f"missing depth observation group '{self.depth_obs_group}'")
        depth = obs[self.depth_obs_group]
        expected_shape = (*obs.batch_size, *self.depth_shape)
        if tuple(depth.shape) != expected_shape:
            raise ValueError(f"expected depth shape {expected_shape}, got {tuple(depth.shape)}")
        return depth

    def get_wheel_roughness(self, obs: TensorDict) -> torch.Tensor:
        if self.wheel_roughness_group is None:
            raise ValueError("wheel_roughness must be configured to train the roughness head.")
        wheel_roughness = obs[self.wheel_roughness_group]
        expected_shape = (*obs.batch_size, self.wheel_roughness_dim)
        if tuple(wheel_roughness.shape) != expected_shape:
            raise ValueError(f"expected wheel roughness shape {expected_shape}, got {tuple(wheel_roughness.shape)}")
        return wheel_roughness

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
        latent, predicted_lin_vel, _, next_hidden_state = self._get_student_outputs_with_roughness(
            obs,
            hidden_state,
            update_hidden_state=update_hidden_state,
            use_internal_state=use_internal_state,
        )
        return latent, predicted_lin_vel, next_hidden_state

    def _get_student_outputs_with_roughness(
        self,
        obs: TensorDict,
        hidden_state: HiddenState = None,
        *,
        update_hidden_state: bool = False,
        use_internal_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("Depth velocity representation expects a tensor GRU hidden state")
        active_hidden_state = (
            self._student_hidden_state
            if use_internal_state and hidden_state is None
            else hidden_state
        )
        normalized_current_proprio = self.current_proprio_obs_normalizer(self.get_current_proprio(obs))
        depth_latent, next_hidden_state = self._encode_depth(
            self.get_depth_obs(obs),
            normalized_current_proprio,
            active_hidden_state,
        )
        if update_hidden_state:
            self._student_hidden_state = next_hidden_state.detach()
        proprio_depth_obs = torch.cat((self.get_proprio_obs(obs), depth_latent), dim=-1)
        features = self.proprio_encoder(proprio_depth_obs)
        latent = self.student_latent_head(features)
        predicted_lin_vel = self.lin_vel_head(features)
        predicted_wheel_roughness = torch.sigmoid(self.wheel_roughness_head(features))
        return self._normalize_latent(latent), predicted_lin_vel, predicted_wheel_roughness, next_hidden_state

    def _encode_depth(
        self,
        depth: torch.Tensor,
        normalized_current_proprio: torch.Tensor,
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
        next_hidden_state = depth_proprio_gru_step(
            self.depth_encoder,
            self.depth_gru,
            self.depth_preprocessor(depth),
            normalized_current_proprio,
            hidden_state,
        )
        return next_hidden_state, next_hidden_state

    def _encode_depth_sequence(
        self,
        depth: torch.Tensor,
        normalized_current_proprio: torch.Tensor,
        dones: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth.ndim != 5:
            raise ValueError(f"depth sequence must have shape [time, batch, C, H, W], got {tuple(depth.shape)}")
        if tuple(depth.shape[2:]) != self.depth_shape:
            raise ValueError(f"expected depth shape {self.depth_shape}, got {tuple(depth.shape[2:])}")
        if tuple(dones.shape[:2]) != tuple(depth.shape[:2]):
            raise ValueError(f"dones must share [time, batch] with depth, got {tuple(dones.shape)}")
        if tuple(normalized_current_proprio.shape[:2]) != tuple(depth.shape[:2]):
            raise ValueError(
                "normalized_current_proprio must share [time, batch] with depth, got "
                f"{tuple(normalized_current_proprio.shape)}"
            )

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
            latent, hidden_state = self._encode_depth(
                depth[step],
                normalized_current_proprio[step],
                hidden_state,
            )
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

    @staticmethod
    def _get_optional_wheel_roughness_group(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
    ) -> str | None:
        groups = obs_groups.get("wheel_roughness")
        if groups is None:
            return None
        if len(groups) != 1:
            raise ValueError("'wheel_roughness' must contain exactly one observation group.")
        group = groups[0]
        if tuple(obs[group].shape[-1:]) != (2,):
            raise ValueError(f"wheel roughness '{group}' must end in dimension 2, got {obs[group].shape}")
        return group


def depth_proprio_gru_step(
    depth_encoder: nn.Module,
    depth_gru: nn.GRUCell,
    depth: torch.Tensor,
    normalized_current_proprio: torch.Tensor,
    hidden_state: torch.Tensor,
) -> torch.Tensor:
    """Advance the depth--proprio belief state."""
    return depth_gru(torch.cat((depth_encoder(depth), normalized_current_proprio), dim=-1), hidden_state)


class _DepthCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channels: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str,
        final_stride: int = 2,
        strides: tuple[int, ...] | list[int] | None = None,
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
        if strides is None:
            if final_stride <= 0:
                raise ValueError(f"final_stride must be positive, got {final_stride}")
            strides = (*((2,) * (len(channels) - 1)), final_stride)
        if len(strides) != len(channels) or any(stride <= 0 for stride in strides):
            raise ValueError(
                "depth CNN strides must be positive and have one entry per channel, "
                f"got {tuple(strides)} for {len(channels)} channels"
            )
        for idx, out_channels in enumerate(channels):
            kernel_size = 5 if idx == 0 else 3
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides[idx]))
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
        self.depth_preprocessor = copy.deepcopy(model.depth_preprocessor)
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
        current_proprio = self.current_proprio_obs_normalizer(proprio_history[:, -1, :])
        depth_latent = self.depth_gru(
            torch.cat(
                (self.depth_encoder(self.depth_preprocessor(depth)), current_proprio),
                dim=-1,
            ),
            self.hidden_state,
        )
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
        student_obs = torch.cat(
            (proprio_obs, depth_latent),
            dim=-1,
        )
        features = self.proprio_encoder(student_obs)
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
        current_proprio = self.current_proprio_obs_normalizer(proprio_history[:, -1, :])
        hidden_state_out = self.depth_gru(
            torch.cat(
                (self.depth_encoder(self.depth_preprocessor(depth)), current_proprio),
                dim=-1,
            ),
            hidden_state_in,
        )
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
