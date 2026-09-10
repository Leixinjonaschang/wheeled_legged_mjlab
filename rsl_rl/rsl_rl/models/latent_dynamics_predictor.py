# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action-conditioned student-latent dynamics prediction."""

# ruff: file-ignore[lowercase-imported-as-non-lowercase]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import MLP


class LatentDynamicsPredictor(nn.Module):
    """Direct multi-horizon and autoregressive latent-state predictor."""

    def __init__(
        self,
        latent_dim: int,
        lin_vel_dim: int,
        action_dim: int,
        horizons: tuple[int, ...] | list[int] = (1, 5, 10),
        hidden_dims: tuple[int, ...] | list[int] = (128, 256, 256, 128),
        activation: str = "elu",
        normalize_latent: bool = True,
    ) -> None:
        """Create one direct predictor for each configured horizon."""
        super().__init__()
        self.latent_dim = latent_dim
        self.lin_vel_dim = lin_vel_dim
        self.action_dim = action_dim
        self.state_dim = latent_dim + lin_vel_dim
        self.horizons = tuple(int(horizon) for horizon in horizons)
        if not self.horizons:
            raise ValueError("horizons must not be empty.")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain only positive integers.")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must not contain duplicates.")
        self.normalize_latent = normalize_latent
        self.predictors = nn.ModuleDict({
            str(horizon): MLP(
                self.state_dim + horizon * action_dim,
                self.state_dim,
                hidden_dims,
                activation,
            )
            for horizon in self.horizons
        })

    def forward(
        self,
        latent: torch.Tensor,
        normalized_lin_vel: torch.Tensor,
        applied_actions: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the state at one configured direct horizon."""
        key = str(horizon)
        if key not in self.predictors:
            raise ValueError(f"No predictor configured for horizon {horizon}; available: {self.horizons}.")
        expected_action_dim = horizon * self.action_dim
        if applied_actions.ndim == latent.ndim + 1:
            if applied_actions.shape[-2] != horizon:
                raise ValueError(
                    f"Horizon {horizon} requires {horizon} ordered actions, got shape {tuple(applied_actions.shape)}."
                )
            applied_actions = applied_actions.flatten(start_dim=-2)
        if applied_actions.shape[-1] != expected_action_dim:
            raise ValueError(
                f"Horizon {horizon} expects {expected_action_dim} action features, got {tuple(applied_actions.shape)}."
            )
        prediction = self.predictors[key](torch.cat((latent, normalized_lin_vel, applied_actions), dim=-1))
        predicted_latent, predicted_lin_vel = prediction.split((self.latent_dim, self.lin_vel_dim), dim=-1)
        if self.normalize_latent:
            predicted_latent = F.normalize(predicted_latent, p=2.0, dim=-1)
        return predicted_latent, predicted_lin_vel

    def rollout(
        self,
        latent: torch.Tensor,
        normalized_lin_vel: torch.Tensor,
        applied_action_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose the horizon-one predictor without detaching intermediate states."""
        if "1" not in self.predictors:
            raise ValueError("Autoregressive rollout requires a horizon-one predictor.")
        if applied_action_sequence.ndim < 2 or (applied_action_sequence.shape[-1] != self.action_dim):
            raise ValueError(
                f"applied_action_sequence must be [time, ..., action_dim], got {tuple(applied_action_sequence.shape)}."
            )
        latent_predictions = []
        velocity_predictions = []
        current_latent = latent
        current_velocity = normalized_lin_vel
        for action in applied_action_sequence.unbind(dim=0):
            current_latent, current_velocity = self(current_latent, current_velocity, action, horizon=1)
            latent_predictions.append(current_latent)
            velocity_predictions.append(current_velocity)
        if not latent_predictions:
            raise ValueError("Autoregressive rollout requires at least one action.")
        return torch.stack(latent_predictions), torch.stack(velocity_predictions)
