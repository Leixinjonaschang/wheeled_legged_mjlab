"""Reusable ray patterns for wheeled-legged sensors."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.sensor import GridPatternCfg


@dataclass
class OffsetGridPatternCfg(GridPatternCfg):
    """Grid ray pattern whose center can be translated in the aligned frame."""

    center: tuple[float, float] = (0.0, 0.0)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Number of grid samples as ``(rows, columns)``."""
        offsets, _ = self.generate_rays(None, "cpu")
        rows = int(offsets[:, 1].unique().numel())
        columns = int(offsets[:, 0].unique().numel())
        return rows, columns

    def generate_rays(
        self,
        mj_model: mujoco.MjModel | None,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets, directions = super().generate_rays(mj_model, device)
        offsets[:, 0] += self.center[0]
        offsets[:, 1] += self.center[1]
        return offsets, directions
