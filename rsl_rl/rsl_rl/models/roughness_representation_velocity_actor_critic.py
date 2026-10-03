# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: ANN201, D102, D107, N812

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.representation_velocity_actor_critic import (
    RepresentationVelocityActorCritic,
)


class RoughnessRepresentationVelocityActorCritic(RepresentationVelocityActorCritic):
    """Blind velocity model whose student also regresses the wheel roughness gate."""

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
        self.wheel_roughness_obs_groups, self.wheel_roughness_dim = self._get_obs_dim(
            obs, obs_groups, "wheel_roughness"
        )
        if self.wheel_roughness_dim != 2:
            raise ValueError(f"wheel_roughness must have dimension 2, got {self.wheel_roughness_dim}.")
        self.wheel_roughness_head = nn.Linear(self.student_latent_head.in_features, self.wheel_roughness_dim)

    def compute_student_losses(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_loss, representation_loss, lin_vel_loss, _ = self.compute_student_losses_with_roughness(obs)
        return total_loss, representation_loss, lin_vel_loss

    def compute_student_losses_with_roughness(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return student losses, including wheel roughness regression."""
        proprio_latent, predicted_lin_vel, predicted_wheel_roughness = self._get_student_outputs_with_roughness(obs)
        with torch.no_grad():
            privileged_latent = self.get_privileged_latent(obs)
        representation_loss = (1.0 - F.cosine_similarity(proprio_latent, privileged_latent, dim=-1)).mean()
        lin_vel_loss = F.mse_loss(predicted_lin_vel, self.get_lin_vel_target(obs))
        roughness_loss = F.smooth_l1_loss(predicted_wheel_roughness, self.get_wheel_roughness(obs))
        return representation_loss + lin_vel_loss + roughness_loss, representation_loss, lin_vel_loss, roughness_loss

    def student_parameters(self):
        """Yield parameters optimized by student representation learning."""
        yield from super().student_parameters()
        yield from self.wheel_roughness_head.parameters()

    def get_proprio_outputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        latent, predicted_lin_vel, _ = self._get_student_outputs_with_roughness(obs)
        return latent, predicted_lin_vel

    def get_wheel_roughness(self, obs: TensorDict) -> torch.Tensor:
        return self._cat_obs(obs, self.wheel_roughness_obs_groups)

    def _get_student_outputs_with_roughness(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Share one student forward pass between the policy and the student losses."""
        features = self.proprio_encoder(self.get_proprio_obs(obs))
        latent = self._normalize_latent(self.student_latent_head(features))
        predicted_lin_vel = self.lin_vel_head(features)
        predicted_wheel_roughness = torch.sigmoid(self.wheel_roughness_head(features))
        return latent, predicted_lin_vel, predicted_wheel_roughness
