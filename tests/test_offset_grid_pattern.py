from __future__ import annotations

import math

import torch

from wheeled_legged_mjlab.sensors import OffsetGridPatternCfg
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    TERRAIN_SCAN_CENTER,
    TERRAIN_SCAN_GRID_SHAPE,
    TERRAIN_SCAN_RESOLUTION,
    TERRAIN_SCAN_SIZE,
    make_sensors,
)


def test_offset_grid_pattern_translates_grid_center() -> None:
    pattern = OffsetGridPatternCfg(
        size=TERRAIN_SCAN_SIZE,
        resolution=TERRAIN_SCAN_RESOLUTION,
        center=TERRAIN_SCAN_CENTER,
    )

    offsets, directions = pattern.generate_rays(None, "cpu")

    rows, cols = TERRAIN_SCAN_GRID_SHAPE
    size_x, size_y = TERRAIN_SCAN_SIZE
    center_x, center_y = TERRAIN_SCAN_CENTER

    assert pattern.grid_shape == TERRAIN_SCAN_GRID_SHAPE
    assert offsets.shape == (rows * cols, 3)
    assert directions.shape == offsets.shape
    assert offsets[:, 0].unique().numel() == cols
    assert offsets[:, 1].unique().numel() == rows
    torch.testing.assert_close(
        offsets[:, 0].amin(), torch.tensor(center_x - size_x / 2)
    )
    torch.testing.assert_close(
        offsets[:, 0].amax(), torch.tensor(center_x + size_x / 2)
    )
    torch.testing.assert_close(
        offsets[:, 1].amin(), torch.tensor(center_y - size_y / 2)
    )
    torch.testing.assert_close(
        offsets[:, 1].amax(), torch.tensor(center_y + size_y / 2)
    )
    torch.testing.assert_close(
        offsets[:, :2].mean(dim=0), torch.tensor(TERRAIN_SCAN_CENTER)
    )
    torch.testing.assert_close(
        directions, torch.tensor([0.0, 0.0, -1.0]).expand_as(directions)
    )


def test_rough_sensor_uses_offset_grid_pattern() -> None:
    terrain_scan = next(
        sensor for sensor in make_sensors(rough=True) if sensor.name == "terrain_scan"
    )

    assert isinstance(terrain_scan.pattern, OffsetGridPatternCfg)
    assert terrain_scan.pattern.size == TERRAIN_SCAN_SIZE
    assert terrain_scan.pattern.resolution == TERRAIN_SCAN_RESOLUTION
    assert terrain_scan.pattern.center == TERRAIN_SCAN_CENTER
    assert terrain_scan.pattern.grid_shape == TERRAIN_SCAN_GRID_SHAPE
    offsets, _ = terrain_scan.pattern.generate_rays(None, "cpu")
    assert offsets.shape[0] == math.prod(TERRAIN_SCAN_GRID_SHAPE)
    assert not any(
        sensor.name == "terrain_scan" for sensor in make_sensors(rough=False)
    )
