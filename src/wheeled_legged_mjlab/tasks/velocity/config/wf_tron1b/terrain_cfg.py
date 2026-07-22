from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains import (
    HfDiscreteObstaclesTerrainCfg,
    BoxRandomSpreadTerrainCfg,
    BoxRandomStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxSteppingStonesTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxTiltedGridTerrainCfg,
)

TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=50,
    num_cols=21,
    sub_terrains={
        # Alternate flat and non-flat columns so world-y straight-line commands
        # encounter terrain transitions at a fixed difficulty level.
        "flat__0": BoxFlatTerrainCfg(
            proportion=0.01,
        ),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_width_range=(0.3, 1.5),
            obstacle_height_range=(0.01, 0.15),
            num_obstacles=200,
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "flat__1": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
            horizontal_scale=0.15,  # Increase resolution spacing to reduce collision points
            vertical_scale=0.005,
        ),
        "flat__2": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,  # Increase resolution spacing to reduce collision points
            vertical_scale=0.005,
        ),
        "flat__3": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "hf_pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,  # Increase resolution spacing to reduce collision points
            vertical_scale=0.005,
            inverted=True,
        ),
        "flat__4": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "pyramid_stair_inv": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            border_width =0.5,
            platform_width=2.0,
        ),
        "flat__5": BoxFlatTerrainCfg(
            proportion=0.02,
        ),
        "pyramid_stair": BoxPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            border_width =0.5,
            platform_width=2.0,
        ),
        "flat__6": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "random_stairs": BoxRandomStairsTerrainCfg(
            proportion=0.05,
            step_width=0.5,
            step_height_range=(0.015, 0.075),
            border_width=0.5,
            platform_width=2.0,
        ),
        "flat__7": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "tilted_grid": BoxTiltedGridTerrainCfg(
            proportion=0.05,
            grid_width=0.8,
            tilt_range_deg=6.0,
            height_range=0.04,
            border_width=0.5,
            platform_width=2.0,
            floor_depth=0.8,
        ),
        "flat__8": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "random_spread": BoxRandomSpreadTerrainCfg(
            proportion=0.05,
            num_boxes=16,
            box_width_range=(0.25, 0.45),
            box_length_range=(0.25, 0.55),
            box_height_range=(0.03, 0.20),
            box_yaw_range=(-20.0, 20.0),
            border_width=0.5,
            platform_width=2.0,
        ),
        "flat__9": BoxFlatTerrainCfg(
            proportion=0.03,
        ),
        "stepping_stones": BoxSteppingStonesTerrainCfg(
            proportion=0.07,
            # Level zero is a contiguous, uniform tiled surface. Difficulty then
            # ramps across 50 rows into smaller, separated, irregular stones.
            stone_size_range=(0.50, 0.75),
            stone_distance_range=(0.0, 0.15),
            stone_height=0.0,
            stone_height_variation=0.075,
            stone_size_variation=0.075,
            displacement_range=0.075,
            border_width=0.5,
            platform_width=2.0,
            floor_depth=0.8,
        ),
        "flat__10": BoxFlatTerrainCfg(
            proportion=0.01,
        ),
    }
)

TERRAINS_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TERRAINS_CFG,
    max_init_terrain_level=8,
    env_spacing=2.5,
)

PLANE_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="plane"
)
