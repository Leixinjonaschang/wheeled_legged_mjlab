# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.depth_representation_velocity_actor_critic import (
    DepthRepresentationVelocityActorCritic,
)
from rsl_rl.modules import MLP


class DepthRepresentationVelocityPredictorActorCritic(DepthRepresentationVelocityActorCritic):
    """Depth velocity representation model with training-only latent dynamics predictors."""

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
        latent_dynamics_hidden_dims: tuple[int, ...] | list[int] = (128, 256, 256, 128),
        latent_dynamics_horizons: tuple[int, ...] | list[int] = (1,),
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
            depth_feature_dim=depth_feature_dim,
            depth_gru_hidden_dim=depth_gru_hidden_dim,
            depth_channels=depth_channels,
        )
        self.latent_dynamics_horizons = tuple(int(horizon) for horizon in latent_dynamics_horizons)
        if not self.latent_dynamics_horizons:
            raise ValueError("latent_dynamics_horizons must not be empty.")
        if any(horizon <= 0 for horizon in self.latent_dynamics_horizons):
            raise ValueError("latent_dynamics_horizons must contain only positive integers.")
        if len(set(self.latent_dynamics_horizons)) != len(self.latent_dynamics_horizons):
            raise ValueError("latent_dynamics_horizons must not contain duplicates.")
        self.latent_dynamics_action_dim = output_dim
        self.latent_dynamics_predictors = torch.nn.ModuleDict(
            {
                str(horizon): MLP(
                    latent_dim + horizon * output_dim,
                    latent_dim,
                    latent_dynamics_hidden_dims,
                    activation,
                )
                for horizon in self.latent_dynamics_horizons
            }
        )

    def ppo_parameters(self):
        """Yield teacher optimizer parameters, including the training-only predictors."""
        yield from super().ppo_parameters()
        yield from self.latent_dynamics_predictors.parameters()

    def latent_dynamics_parameters(self):
        """Yield parameters optimized by the direct multi-horizon dynamics loss."""
        yield from self.privileged_encoder.parameters()
        yield from self.latent_dynamics_predictors.parameters()

    def predict_privileged_latent(
        self,
        latent: torch.Tensor,
        applied_action_block: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        predictor_key = str(horizon)
        if predictor_key not in self.latent_dynamics_predictors:
            raise ValueError(
                f"No latent dynamics predictor configured for horizon {horizon}; "
                f"available horizons are {self.latent_dynamics_horizons}."
            )
        expected_action_dim = horizon * self.latent_dynamics_action_dim
        if applied_action_block.shape[-1] != expected_action_dim:
            raise ValueError(
                f"Horizon {horizon} expects an action block with {expected_action_dim} features, "
                f"got shape {tuple(applied_action_block.shape)}."
            )
        predictor_input = torch.cat((latent, applied_action_block), dim=-1)
        prediction = self.latent_dynamics_predictors[predictor_key](predictor_input)
        return F.normalize(prediction, p=2.0, dim=-1)

    def compute_latent_dynamics_loss(
        self,
        obs_t: TensorDict,
        applied_action_block: torch.Tensor,
        obs_future: TensorDict,
        horizon: int = 1,
        detach_source: bool = False,
    ) -> torch.Tensor:
        latent_t = self.get_privileged_latent(obs_t)
        if detach_source:
            latent_t = latent_t.detach()
        with torch.no_grad():
            latent_future = self.get_privileged_latent(obs_future)
        predicted_future = self.predict_privileged_latent(
            latent_t,
            applied_action_block,
            horizon,
        )
        return F.mse_loss(predicted_future, latent_future)

    def rollout_privileged_latent(
        self,
        latent: torch.Tensor,
        applied_action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressively compose the one-step predictor without truncating gradients."""
        if "1" not in self.latent_dynamics_predictors:
            raise ValueError("Autoregressive latent rollout requires a horizon-1 predictor.")
        if applied_action_sequence.shape[-1] != self.latent_dynamics_action_dim:
            raise ValueError(
                "Each latent rollout action must have "
                f"{self.latent_dynamics_action_dim} features, got shape "
                f"{tuple(applied_action_sequence.shape)}."
            )
        predictions = []
        current_latent = latent
        for applied_action in applied_action_sequence.unbind(dim=0):
            current_latent = self.predict_privileged_latent(
                current_latent,
                applied_action,
                horizon=1,
            )
            predictions.append(current_latent)
        if not predictions:
            raise ValueError("Latent rollout action sequence must contain at least one step.")
        return torch.stack(predictions, dim=0)
