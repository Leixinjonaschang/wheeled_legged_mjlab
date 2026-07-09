from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains import (
    HfDiscreteObstaclesTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxSteppingStonesTerrainCfg,
    BoxPyramidStairsTerrainCfg,
)

TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=35.0,
    num_rows=50,
    num_cols=10,
    sub_terrains={
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_width_range=(0.3, 1.5),
            obstacle_height_range=(0.01, 0.15),
            num_obstacles=200,
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
            horizontal_scale=0.15,  # Increase resolution spacing to reduce collision points
            vertical_scale=0.005,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,  # Increase resolution spacing to reduce collision points
            vertical_scale=0.005,
        ),
        "pyramid_stair_inv": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            border_width =0.5,
            platform_width=2.0,
        ),
        "pyramid_stair": BoxPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            border_width =0.5,
            platform_width=2.0,
        ),
        # "stepping_stones": BoxSteppingStonesTerrainCfg(
        #     proportion=0.2,
        #     stone_distance_range=(0.1, 0.2),
        #     border_width = 0.5,
        #     platform_width=2.0,
        # ),
        "flat": BoxFlatTerrainCfg(
            proportion=0.3,
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
