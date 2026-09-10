# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployable recurrent depth actor used by depth-predictor PPO."""

# ruff: file-ignore[lowercase-imported-as-non-lowercase]

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterator
from tensordict import TensorDict

from rsl_rl.models.depth_representation_velocity_actor_critic import (
    DepthPreprocessor,
    _DepthCNN,
)
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


class DepthActor(nn.Module):
    """Depth student actor with no teacher or privileged-observation path."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        latent_dim: int = 64,
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
        """Build the deployable actor and training-only supervision heads."""
        super().__init__()
        (
            self.proprio_history_obs_groups,
            self.proprio_history_length,
            self.current_proprio_dim,
        ) = self._get_history_shape(obs, obs_groups, "proprio_history")
        self.command_obs_groups, self.command_dim = self._get_obs_dim(obs, obs_groups, "actor_command")
        self.lin_vel_target_obs_groups = obs_groups.get("lin_vel_target", [])
        self.lin_vel_dim = 3
        if self.lin_vel_target_obs_groups:
            self.lin_vel_target_obs_groups, lin_vel_target_dim = self._get_obs_dim(obs, obs_groups, "lin_vel_target")
            if lin_vel_target_dim != self.lin_vel_dim:
                raise ValueError(f"lin_vel_target must have dimension {self.lin_vel_dim}, got {lin_vel_target_dim}.")
        self.depth_obs_group, self.depth_shape = self._get_depth_group_and_shape(obs, obs_groups, "depth_encoder")
        self.wheel_roughness_group = self._get_optional_wheel_roughness_group(obs, obs_groups)

        self.proprio_history_obs_dim = self.proprio_history_length * self.current_proprio_dim
        self.actor_obs_dim = self.lin_vel_dim + self.current_proprio_dim + self.command_dim + latent_dim
        self.obs_groups = [
            *self.proprio_history_obs_groups,
            *self.command_obs_groups,
            self.depth_obs_group,
        ]
        self.obs_dim = self.proprio_history_obs_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.normalize_latent = normalize_latent
        self.wheel_roughness_dim = 2

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.proprio_history_obs_normalizer = EmpiricalNormalization(self.proprio_history_obs_dim)
            self.current_proprio_obs_normalizer = EmpiricalNormalization(self.current_proprio_dim)
            self.command_obs_normalizer = EmpiricalNormalization(self.command_dim)
            self.lin_vel_normalizer = EmpiricalNormalization(self.lin_vel_dim)
        else:
            self.proprio_history_obs_normalizer = nn.Identity()
            self.current_proprio_obs_normalizer = nn.Identity()
            self.command_obs_normalizer = nn.Identity()
            self.lin_vel_normalizer = nn.Identity()

        if distribution_cfg is not None:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            distribution_class: type[Distribution] = resolve_callable(  # type: ignore
                distribution_cfg.pop("class_name")
            )
            self.distribution: Distribution | None = distribution_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        self.depth_feature_dim = depth_feature_dim
        self.depth_gru_hidden_dim = depth_gru_hidden_dim
        self.depth_gru_input_dim = depth_feature_dim + self.current_proprio_dim
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
        self.depth_gru = nn.GRUCell(self.depth_gru_input_dim, self.depth_gru_hidden_dim)

        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must contain at least one dimension.")
        encoder_feature_dim = encoder_hidden_dims[-1]
        self.student_encoder = MLP(
            self.proprio_history_obs_dim + self.depth_gru_hidden_dim,
            encoder_feature_dim,
            encoder_hidden_dims,
            activation,
        )
        self.latent_head = nn.Linear(encoder_feature_dim, self.latent_dim)
        self.lin_vel_head = nn.Linear(encoder_feature_dim, self.lin_vel_dim)
        self.roughness_head = nn.Linear(encoder_feature_dim, self.wheel_roughness_dim)
        self.actor_head = MLP(
            self.actor_obs_dim,
            actor_output_dim,
            hidden_dims,
            activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_head)

        self._hidden_state: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run one actor step, advancing internal state only when state is implicit."""
        if masks is not None:
            raise ValueError("DepthActor uses explicit full-rollout replay, not padded masks.")
        if len(obs.batch_size) != 1:
            raise ValueError(f"DepthActor.forward expects [batch] observations, got {obs.batch_size}.")
        latent, predicted_lin_vel, _, next_hidden_state = self.get_student_outputs(
            obs,
            hidden_state=hidden_state,
            use_internal_state=True,
        )
        if hidden_state is None:
            self._hidden_state = next_hidden_state.detach()
        return self._act(obs, latent, predicted_lin_vel, stochastic_output)

    def forward_sequence(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        *,
        stochastic_output: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Replay a continuous ``[time, batch]`` sequence without online-state mutation."""
        latent, predicted_lin_vel, predicted_roughness, final_hidden_state = self.get_student_outputs_sequence(
            obs, dones, hidden_state
        )
        actions = self._act(obs, latent, predicted_lin_vel, stochastic_output)
        return (
            actions,
            latent,
            predicted_lin_vel,
            predicted_roughness,
            final_hidden_state,
        )

    def get_student_outputs(
        self,
        obs: TensorDict,
        hidden_state: torch.Tensor | None = None,
        *,
        use_internal_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one observation and return latent, supervision heads, and state."""
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("DepthActor expects a tensor GRU hidden state.")
        active_hidden_state = self._hidden_state if use_internal_state and hidden_state is None else hidden_state
        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        current_proprio = self.current_proprio_obs_normalizer(proprio_history[..., -1, :])
        next_hidden_state = self._encode_depth_step(self.get_depth_obs(obs), current_proprio, active_hidden_state)
        proprio_input = self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=-2))
        return self._heads(
            torch.cat((proprio_input, next_hidden_state), dim=-1),
            next_hidden_state,
        )

    def get_student_outputs_sequence(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay the student encoder with resets after terminal transitions."""
        if len(obs.batch_size) != 2:
            raise ValueError(f"Expected sequence observations with [time, batch], got {obs.batch_size}.")
        time_steps, batch_size = obs.batch_size
        if tuple(dones.shape[:2]) != (time_steps, batch_size):
            raise ValueError(f"dones must share [time, batch] with observations, got {tuple(dones.shape)}.")
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size,
                self.depth_gru_hidden_dim,
                device=self.get_depth_obs(obs).device,
                dtype=self.get_depth_obs(obs).dtype,
            )
        else:
            hidden_state = hidden_state.detach()
        expected_hidden_shape = (batch_size, self.depth_gru_hidden_dim)
        if tuple(hidden_state.shape) != expected_hidden_shape:
            raise ValueError(f"expected hidden_state shape {expected_hidden_shape}, got {tuple(hidden_state.shape)}.")

        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        current_proprio = self.current_proprio_obs_normalizer(proprio_history[..., -1, :])
        depth = self.get_depth_obs(obs)
        recurrent_features = []
        for step in range(time_steps):
            hidden_state = self._encode_depth_step(depth[step], current_proprio[step], hidden_state)
            recurrent_features.append(hidden_state)
            done_mask = dones[step].to(dtype=torch.bool, device=hidden_state.device)
            hidden_state = torch.where(done_mask.view(-1, 1), torch.zeros_like(hidden_state), hidden_state)

        proprio_input = self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=-2))
        student_input = torch.cat((proprio_input, torch.stack(recurrent_features)), dim=-1)
        return self._heads(student_input, hidden_state)

    def compute_supervised_losses_sequence(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return velocity and roughness losses plus replayed student latents."""
        latent, predicted_lin_vel, predicted_roughness, _ = self.get_student_outputs_sequence(obs, dones, hidden_state)
        lin_vel_loss = F.mse_loss(predicted_lin_vel, self.get_lin_vel_target(obs))
        roughness_loss = (
            F.smooth_l1_loss(predicted_roughness, self.get_wheel_roughness(obs))
            if self.wheel_roughness_group is not None
            else lin_vel_loss.new_zeros(())
        )
        return lin_vel_loss, roughness_loss, latent

    def get_predicted_lin_vel(self, obs: TensorDict) -> torch.Tensor:
        """Preview velocity without advancing the online GRU state."""
        _, predicted_lin_vel, _, _ = self.get_student_outputs(obs, use_internal_state=True)
        return predicted_lin_vel

    def get_normalized_lin_vel_target(self, obs: TensorDict) -> torch.Tensor:
        """Return true linear velocity in the predictor's normalized coordinates."""
        return self.lin_vel_normalizer(self.get_lin_vel_target(obs))

    def get_lin_vel_target(self, obs: TensorDict) -> torch.Tensor:
        """Extract the unnormalized true linear-velocity target."""
        if not self.lin_vel_target_obs_groups:
            raise ValueError("lin_vel_target is unavailable in this inference-only actor.")
        return self._cat_obs(obs, self.lin_vel_target_obs_groups)

    def get_wheel_roughness(self, obs: TensorDict) -> torch.Tensor:
        """Extract the two wheel-roughness supervision targets."""
        if self.wheel_roughness_group is None:
            raise ValueError("wheel_roughness is not configured for this actor.")
        roughness = obs[self.wheel_roughness_group]
        if roughness.shape[-1] != self.wheel_roughness_dim:
            raise ValueError(f"wheel_roughness must end in dimension 2, got {tuple(roughness.shape)}.")
        return roughness

    def get_depth_obs(self, obs: TensorDict) -> torch.Tensor:
        """Validate and return metric depth observations."""
        if self.depth_obs_group not in obs:
            raise ValueError(f"missing depth observation group '{self.depth_obs_group}'")
        depth = obs[self.depth_obs_group]
        expected_shape = (*obs.batch_size, *self.depth_shape)
        if tuple(depth.shape) != expected_shape:
            raise ValueError(f"expected depth shape {expected_shape}, got {tuple(depth.shape)}")
        return depth

    def student_backbone_parameters(self) -> Iterator[nn.Parameter]:
        """Yield CNN, GRU, and student-encoder parameters."""
        yield from self.depth_encoder.parameters()
        yield from self.depth_gru.parameters()
        yield from self.student_encoder.parameters()

    def policy_parameters(self) -> Iterator[nn.Parameter]:
        """Yield exactly the actor parameters reached by policy and entropy losses."""
        yield from self.student_backbone_parameters()
        yield from self.latent_head.parameters()
        yield from self.lin_vel_head.parameters()
        yield from self.actor_head.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        """Reset all or selected online actor states."""
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("DepthActor expects a tensor GRU hidden state.")
        if dones is None:
            self._hidden_state = None if hidden_state is None else hidden_state.detach().clone()
        elif hidden_state is not None:
            raise ValueError("Cannot combine per-environment reset and a custom state.")
        elif self._hidden_state is not None:
            done_mask = dones.to(device=self._hidden_state.device, dtype=torch.bool).view(-1)
            self._hidden_state[done_mask] = 0.0

    def get_hidden_state(self) -> HiddenState:
        """Return the online GRU state saved before the next actor step."""
        return self._hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach all or selected online recurrent state."""
        if self._hidden_state is None:
            return
        if dones is None:
            self._hidden_state = self._hidden_state.detach()
        else:
            done_mask = dones.to(device=self._hidden_state.device, dtype=torch.bool).view(-1)
            self._hidden_state[done_mask] = self._hidden_state[done_mask].detach()

    @property
    def output_mean(self) -> torch.Tensor:
        """Mean of the active action distribution."""
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        """Standard deviation of the active action distribution."""
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        """Entropy of the active action distribution."""
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Parameters of the active action distribution."""
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Evaluate sampled actions under the active distribution."""
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return KL divergence from old to new distribution parameters."""
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def update_normalization(self, obs: TensorDict) -> None:
        """Update all actor statistics from one fixed-rollout batch."""
        if not self.obs_normalization:
            return
        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        self.proprio_history_obs_normalizer.update(  # type: ignore[union-attr]
            proprio_history.reshape(-1, self.proprio_history_obs_dim)
        )
        self.current_proprio_obs_normalizer.update(  # type: ignore[union-attr]
            proprio_history[..., -1, :].reshape(-1, self.current_proprio_dim)
        )
        self.command_obs_normalizer.update(  # type: ignore[union-attr]
            self._cat_obs(obs, self.command_obs_groups).reshape(-1, self.command_dim)
        )
        if self.lin_vel_target_obs_groups:
            self.lin_vel_normalizer.update(  # type: ignore[union-attr]
                self.get_lin_vel_target(obs).reshape(-1, self.lin_vel_dim)
            )

    def as_jit(self) -> nn.Module:
        """Return the stateful TorchScript deployment wrapper."""
        return _TorchDepthActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return the state-explicit ONNX deployment wrapper."""
        return _OnnxDepthActor(self, verbose)

    def _heads(
        self, student_input: torch.Tensor, final_hidden_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.student_encoder(student_input)
        latent = self.latent_head(features)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        predicted_lin_vel = self.lin_vel_head(features)
        predicted_roughness = torch.sigmoid(self.roughness_head(features))
        return latent, predicted_lin_vel, predicted_roughness, final_hidden_state

    def _encode_depth_step(
        self,
        depth: torch.Tensor,
        normalized_current_proprio: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = depth.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size,
                self.depth_gru_hidden_dim,
                device=depth.device,
                dtype=depth.dtype,
            )
        expected_shape = (batch_size, self.depth_gru_hidden_dim)
        if tuple(hidden_state.shape) != expected_shape:
            raise ValueError(f"expected hidden_state shape {expected_shape}, got {tuple(hidden_state.shape)}.")
        return self.depth_gru(
            torch.cat(
                (
                    self.depth_encoder(self.depth_preprocessor(depth)),
                    normalized_current_proprio,
                ),
                dim=-1,
            ),
            hidden_state,
        )

    def _act(
        self,
        obs: TensorDict,
        latent: torch.Tensor,
        predicted_lin_vel: torch.Tensor,
        stochastic_output: bool,
    ) -> torch.Tensor:
        proprio_history = self._cat_obs(obs, self.proprio_history_obs_groups)
        actor_input = torch.cat(
            (
                latent,
                self.lin_vel_normalizer(predicted_lin_vel),
                self.current_proprio_obs_normalizer(proprio_history[..., -1, :]),
                self.command_obs_normalizer(self._cat_obs(obs, self.command_obs_groups)),
            ),
            dim=-1,
        )
        mlp_output = self.actor_head(actor_input)
        if self.distribution is None:
            return mlp_output
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_output)

    @staticmethod
    def _cat_obs(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[group] for group in groups], dim=-1)

    @staticmethod
    def _get_obs_dim(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        groups = obs_groups[obs_set]
        dimension = 0
        for group in groups:
            if len(obs[group].shape) != 2:
                raise ValueError(f"'{obs_set}' only supports 1D observations, got {obs[group].shape} for '{group}'.")
            dimension += obs[group].shape[-1]
        return groups, dimension

    @staticmethod
    def _get_history_shape(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int, int]:
        groups = obs_groups[obs_set]
        history_length: int | None = None
        frame_dimension = 0
        for group in groups:
            group_obs = obs[group]
            if len(group_obs.shape) != 3:
                raise ValueError(
                    f"Proprio history must have [batch, history, features], got {group_obs.shape} for '{group}'."
                )
            if history_length is None:
                history_length = group_obs.shape[-2]
            elif group_obs.shape[-2] != history_length:
                raise ValueError("All proprio history groups must have equal length.")
            frame_dimension += group_obs.shape[-1]
        if history_length is None:
            raise ValueError("At least one proprio history group is required.")
        return groups, history_length, frame_dimension

    @staticmethod
    def _get_depth_group_and_shape(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[str, tuple[int, int, int]]:
        groups = obs_groups[obs_set]
        if len(groups) != 1:
            raise ValueError(f"'{obs_set}' must contain exactly one group.")
        group = groups[0]
        if len(obs[group].shape) != 4:
            raise ValueError(f"Depth observation must have [batch, C, H, W], got {obs[group].shape}.")
        return group, tuple(obs[group].shape[1:])

    @staticmethod
    def _get_optional_wheel_roughness_group(obs: TensorDict, obs_groups: dict[str, list[str]]) -> str | None:
        groups = obs_groups.get("wheel_roughness")
        if groups is None:
            return None
        if len(groups) != 1:
            raise ValueError("'wheel_roughness' must contain exactly one group.")
        group = groups[0]
        if obs[group].shape[-1] != 2:
            raise ValueError("wheel_roughness must end in dimension 2.")
        return group


class _TorchDepthActor(nn.Module):
    """TorchScript wrapper for stateful single-robot deployment."""

    def __init__(self, model: DepthActor) -> None:
        super().__init__()
        self.proprio_history_obs_normalizer = copy.deepcopy(model.proprio_history_obs_normalizer)
        self.current_proprio_obs_normalizer = copy.deepcopy(model.current_proprio_obs_normalizer)
        self.command_obs_normalizer = copy.deepcopy(model.command_obs_normalizer)
        self.lin_vel_normalizer = copy.deepcopy(model.lin_vel_normalizer)
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.depth_gru = copy.deepcopy(model.depth_gru)
        self.student_encoder = copy.deepcopy(model.student_encoder)
        self.latent_head = copy.deepcopy(model.latent_head)
        self.lin_vel_head = copy.deepcopy(model.lin_vel_head)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.register_buffer("hidden_state", torch.zeros(1, model.depth_gru_hidden_dim))

    def forward(
        self,
        proprio_history: torch.Tensor,
        actor_command: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state_out = self.depth_gru(
            torch.cat(
                (
                    self.depth_encoder(depth),
                    self.current_proprio_obs_normalizer(proprio_history[:, -1, :]),
                ),
                dim=-1,
            ),
            self.hidden_state,
        )
        self.hidden_state[:] = hidden_state_out.detach()
        return self._actor_outputs(proprio_history, actor_command, hidden_state_out)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0

    def _actor_outputs(
        self,
        proprio_history: torch.Tensor,
        actor_command: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student_input = torch.cat(
            (
                self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=1)),
                hidden_state,
            ),
            dim=-1,
        )
        features = self.student_encoder(student_input)
        latent = self.latent_head(features)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        predicted_lin_vel = self.lin_vel_head(features)
        actor_input = torch.cat(
            (
                latent,
                self.lin_vel_normalizer(predicted_lin_vel),
                self.current_proprio_obs_normalizer(proprio_history[:, -1, :]),
                self.command_obs_normalizer(actor_command),
            ),
            dim=-1,
        )
        return (
            self.deterministic_output(self.actor_head(actor_input)),
            predicted_lin_vel,
        )


class _OnnxDepthActor(_TorchDepthActor):
    """ONNX wrapper with explicit recurrent state input and output."""

    is_recurrent: bool = True

    def __init__(self, model: DepthActor, verbose: bool) -> None:
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
        hidden_state_out = self.depth_gru(
            torch.cat(
                (
                    self.depth_encoder(depth),
                    self.current_proprio_obs_normalizer(proprio_history[:, -1, :]),
                ),
                dim=-1,
            ),
            hidden_state_in,
        )
        actions, predicted_lin_vel = self._actor_outputs(proprio_history, actor_command, hidden_state_out)
        return actions, predicted_lin_vel, hidden_state_out

    def get_dummy_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
