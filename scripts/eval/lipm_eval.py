"""LIPM-style terrain evaluation of the depth ablation checkpoints.

Protocol reference: Su et al., "LIPM-Guided Reinforcement Learning for Stable and
Perceptive Locomotion in Bipedal Robots", arXiv:2509.09106v2, Sec. V-B, Fig. 4 and
Table IV. Terrains, difficulty and metric formulas follow this project's own
configuration and the conventions documented in ``PROTOCOL`` below.

Subcommands:
  audit      Build every evaluation terrain at its maximum training difficulty and
             measure the generated geometry by ray casting.
  rollout    Evaluate one checkpoint on one terrain class (one fresh process).
  aggregate  Build per-run and per-group CSV / Markdown tables.
  run        audit + every terrain x checkpoint rollout in isolated subprocesses
             + aggregate.

Full evaluation:
  uv run python scripts/eval/lipm_eval.py run --out logs/lipm_eval/full
Small trial:
  uv run python scripts/eval/lipm_eval.py run --out logs/lipm_eval/trial \
    --num-envs 64 --num-steps 100 --terrains flat pyramid_stair --ckpts Ours
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth"


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    """Unique run name."""
    group: str
    """Experiment group; runs of one group with different training seeds are pooled."""
    path: str
    task_id: str
    """Inferred from the checkpoint: latent-dynamics predictor weights present (Ours,
    RGGP) or absent (LPGP); Ours and RGGP share one architecture and differ only in
    training rewards, so they are told apart by file name."""
    train_seed: str
    """Training seed, or "unknown" when the checkpoint does not record it."""


CHECKPOINTS = (
    CheckpointSpec(
        "Ours", "Ours", "logs/exp_ckpt/model_29999_Ours.pt",
        PREFIX + "-Predict-OursGP", "unknown",
    ),
    CheckpointSpec(
        "RGGP", "RGGP", "logs/exp_ckpt/model_29999_RGGP.pt",
        PREFIX + "-Predict-RGGP", "unknown",
    ),
    CheckpointSpec(
        "LPGP", "LPGP", "logs/exp_ckpt/model_29999_LPGP.pt",
        PREFIX + "-LPGP", "unknown",
    ),
)

# Evaluation terrain classes: every sub-terrain type of the training generator.
# The ten flat columns (flat__0..flat__9) are identical BoxFlatTerrainCfg entries
# and form a single "flat" class.
TERRAINS = (
    "flat",
    "discrete_obstacles",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "pyramid_stair",
    "pyramid_stair_inv",
    "random_stairs",
    "random_spread",
    "stepping_stones",
    "tilted_grid",
)

EVAL_SEED = 20260928
DIFFICULTY = 1.0
GRID_PATCHES = 12
"""Patches per side of the evaluation terrain grid (all patches: one terrain class)."""
SPAWN_PATCHES = 8
"""Inner spawn patches per side; the outer ring of padding patches keeps robots on
the target terrain for the full 10 s (max command speed sqrt(2) m/s)."""
PAD_PATCHES = (GRID_PATCHES - SPAWN_PATCHES) // 2
NUM_ENVS = 1024
NUM_STEPS = 500
LOCAL_FLAT_RELIEF_M = 0.02
PLATFORM_HALF_WIDTH_M = 1.0
FAILURE_TERMS = ("non_finite_physics", "fell_over", "illegal_contact")

PROTOCOL = {
    "reference": "arXiv:2509.09106v2 Sec. V-B, Fig. 4, Table IV",
    "author_code": "No public code or metric implementation found (searched 2026-09-28).",
    "trajectories": "One trajectory per environment: from the evaluation reset until the "
    "first failure or 500 policy steps (10 s). No auto-reset during evaluation; "
    "states after the first failure are never used.",
    "success_rate": "survived 500 steps / valid started trajectories x 100. Failure = any "
    "of the task play configuration's non-timeout terminations: non_finite_physics, "
    "fell_over (|tilt| > 85 deg), illegal_contact (non-wheel collision geom vs terrain, "
    "force > 10 N within the last 4 physics substeps).",
    "orientation_error": "||g_b - [0,0,-1]||_2, g_b = unit gravity vector in the base frame "
    "(dimensionless).",
    "angular_velocity_error": "||(w_x, w_y)||_2 of the base angular velocity in the base "
    "frame, i.e. error w.r.t. zero roll/pitch rate (rad/s).",
    "velocity_tracking_error": "||c_xy - v_xy||_2 (m/s): c = policy-facing command "
    "(world-frame command rotated by the robot yaw, UniformVelocityCommand.command), "
    "v = base linear velocity in the base frame (IsaacLab error_vel_xy convention).",
    "rec_error": "N/A: no checkpoint has a height-map reconstruction output.",
    "sampling": "Post-step states s_1..s_500 (t = 0.02..10 s). A failed trajectory "
    "contributes s_1..s_(T-1); the state that triggered the failure is excluded.",
    "aggregation": "Per-trajectory time average, then equal-weight mean over trajectories; "
    "across training seeds: mean +- sample std (n=1 here: std not defined).",
}


# --------------------------------------------------------------------------------------
# Generic helpers.


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(np.asarray(array))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    diff = git("diff", "HEAD")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked_files": git("status", "--porcelain", "--untracked-files=no"),
        "untracked_eval_files": git("status", "--porcelain", "scripts/eval"),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def _checkpoint(name: str) -> CheckpointSpec:
    for spec in CHECKPOINTS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown checkpoint {name!r}; choices: {[c.name for c in CHECKPOINTS]}")


def _sub_terrain_key(terrain: str) -> str:
    return "flat__0" if terrain == "flat" else terrain


# --------------------------------------------------------------------------------------
# Terrain configuration.


def eval_terrain_generator_cfg(play_cfg, train_cfg, terrain: str, seed: int):
    """Tile one training sub-terrain over the whole grid at the maximum difficulty."""
    from mjlab.terrains import TerrainGeneratorCfg

    key = _sub_terrain_key(terrain)
    train_gen = train_cfg.scene.terrain.terrain_generator
    play_gen = play_cfg.scene.terrain.terrain_generator
    sub = copy.deepcopy(train_gen.sub_terrains[key])
    if sub != play_gen.sub_terrains[key]:
        raise RuntimeError(f"Play and training configs disagree on sub-terrain {key!r}.")
    if terrain == "flat":
        flats = [v for k, v in train_gen.sub_terrains.items() if k.startswith("flat__")]
        if any(type(v) is not type(sub) for v in flats):
            raise RuntimeError("Flat columns are not all BoxFlatTerrainCfg.")
    # Curriculum rows r = 0..num_rows-1 use difficulty lower + (upper-lower) * r/(rows-1);
    # the maximum reachable level is num_rows-1, i.e. the upper end of the range.
    if tuple(train_gen.difficulty_range) != (0.0, 1.0) or not train_gen.curriculum:
        raise RuntimeError("Unexpected training difficulty mapping.")
    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=False,  # Random mode: every patch is an independent instance.
        size=train_gen.size,
        border_width=play_gen.border_width,
        border_height=play_gen.border_height,
        num_rows=GRID_PATCHES,
        num_cols=GRID_PATCHES,
        color_scheme=play_gen.color_scheme,
        sub_terrains={key: sub},
        difficulty_range=(DIFFICULTY, DIFFICULTY),
        add_lights=play_gen.add_lights,
    )


def make_eval_env_cfg(task_id: str, terrain: str, num_envs: int, seed: int):
    """The task's registered play config plus the frozen evaluation overrides."""
    from mjlab.tasks.registry import load_env_cfg

    play_cfg = load_env_cfg(task_id, play=True)
    train_cfg = load_env_cfg(task_id)
    play_cfg.scene.terrain.terrain_generator = eval_terrain_generator_cfg(
        play_cfg, train_cfg, terrain, seed
    )
    play_cfg.scene.terrain.max_init_terrain_level = None
    play_cfg.scene.num_envs = num_envs
    play_cfg.seed = seed
    # Terminated robots are never reset during the evaluation window (see rollout()).
    play_cfg.auto_reset = False
    # Play mode samples a random (row, column) on every reset; spawn patches are
    # assigned explicitly instead.
    play_cfg.events.pop("randomize_terrain")
    if play_cfg.curriculum:
        raise RuntimeError("Play configuration unexpectedly keeps a curriculum.")
    return play_cfg, train_cfg


def spawn_layout(num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column of the spawn patch of every environment (interleaved)."""
    patch = np.arange(num_envs) % (SPAWN_PATCHES * SPAWN_PATCHES)
    return PAD_PATCHES + patch // SPAWN_PATCHES, PAD_PATCHES + patch % SPAWN_PATCHES


def grid_membership(pos_xy: np.ndarray, patch_size: float) -> tuple[np.ndarray, np.ndarray]:
    """In-grid flag and on-central-platform flag for world xy positions."""
    half = 0.5 * GRID_PATCHES * patch_size
    rel = pos_xy + half
    in_grid = np.all((rel >= 0.0) & (rel < 2.0 * half), axis=-1)
    center = (np.floor(rel / patch_size) + 0.5) * patch_size
    on_platform = np.all(np.abs(rel - center) <= PLATFORM_HALF_WIDTH_M, axis=-1)
    return in_grid, on_platform & in_grid


def formula_parameters(sub, difficulty: float) -> dict:
    """Geometry parameters implied by mjlab's generator code at ``difficulty``."""
    d = float(difficulty)
    name = type(sub).__name__
    size = sub.size
    if name == "BoxFlatTerrainCfg":
        return {"note": "flat box; geometry does not depend on difficulty"}
    if name == "HfDiscreteObstaclesTerrainCfg":
        lo, hi = sub.obstacle_height_range
        h_units = int((lo + d * (hi - lo)) / sub.vertical_scale)
        w_min = int(sub.obstacle_width_range[0] / sub.horizontal_scale)
        w_max = int(sub.obstacle_width_range[1] / sub.horizontal_scale)
        widths = np.arange(w_min, w_max + 1, 4) * sub.horizontal_scale
        return {
            "obstacle_height_m": h_units * sub.vertical_scale,
            "obstacle_height_choices_m": [
                -h_units * sub.vertical_scale, -(h_units // 2) * sub.vertical_scale,
                (h_units // 2) * sub.vertical_scale, h_units * sub.vertical_scale,
            ],
            "obstacle_side_choices_m": [round(float(w), 3) for w in widths],
            "num_obstacles": sub.num_obstacles,
            "platform_width_m": sub.platform_width,
        }
    if name == "HfRandomUniformTerrainCfg":
        scale = d if sub.scale_with_difficulty else 1.0
        return {
            "noise_range_m": [sub.noise_range[0] * scale, sub.noise_range[1] * scale],
            "noise_step_m": sub.noise_step,
            "horizontal_scale_m": sub.horizontal_scale,
            "scale_with_difficulty": sub.scale_with_difficulty,
            "note": "difficulty is ignored (scale_with_difficulty=False): every level "
            "uses the full noise range" if not sub.scale_with_difficulty else "",
        }
    if name == "HfPyramidSlopedTerrainCfg":
        slope = sub.slope_range[0] + d * (sub.slope_range[1] - sub.slope_range[0])
        return {
            "slope_rise_over_run": slope,
            "slope_deg": math.degrees(math.atan(slope)),
            "inverted": sub.inverted,
            "platform_width_m": sub.platform_width,
        }
    if name in ("BoxPyramidStairsTerrainCfg", "BoxInvertedPyramidStairsTerrainCfg"):
        step = sub.step_height_range[0] + d * (sub.step_height_range[1] - sub.step_height_range[0])
        steps = int(min(
            (size[0] - 2 * sub.border_width - sub.platform_width) / (2 * sub.step_width),
            (size[1] - 2 * sub.border_width - sub.platform_width) / (2 * sub.step_width),
        ))
        return {
            "step_height_m": step,
            "step_depth_m": sub.step_width,
            "num_steps_per_side": steps,
            "platform_to_border_height_m": (steps + 1) * step,
            "direction": "descending pit (spawn at bottom)"
            if name.startswith("BoxInverted") else "pyramid (spawn on top)",
        }
    if name == "BoxRandomStairsTerrainCfg":
        factor = 0.5 + 0.5 * d
        steps = int(min(
            (size[0] - 2 * sub.border_width - sub.platform_width) / (2 * sub.step_width),
            (size[1] - 2 * sub.border_width - sub.platform_width) / (2 * sub.step_width),
        ))
        return {
            "step_height_range_m": [sub.step_height_range[0] * factor, sub.step_height_range[1] * factor],
            "step_depth_m": sub.step_width,
            "num_steps_per_side": steps,
        }
    if name == "BoxRandomSpreadTerrainCfg":
        factor = 0.2 + 0.8 * d
        return {
            "num_boxes_sampled": int(sub.num_boxes * (0.5 + 0.5 * d)),
            "box_height_range_m": [sub.box_height_range[0] * factor, sub.box_height_range[1] * factor],
            "box_width_range_m": list(sub.box_width_range),
            "box_length_range_m": list(sub.box_length_range),
            "box_yaw_range_deg": list(sub.box_yaw_range),
        }
    if name == "BoxSteppingStonesTerrainCfg":
        s_min, s_max = sub.stone_size_range
        avg = s_max - d * (s_max - s_min)
        inner = size[0] - 2 * sub.border_width
        num = max(2, int(np.floor(inner / (s_max + sub.stone_distance_range[0]))) + 1)
        pitch = inner / (num - 1)
        return {
            "stone_size_m": avg,
            "stone_size_variation_m": sub.stone_size_variation * d,
            "stone_displacement_m": sub.displacement_range * d,
            "stone_top_height_variation_m": sub.stone_height_variation * d,
            "grid_pitch_m": pitch,
            "nominal_gap_m": max(0.0, pitch - avg),
            "pit_depth_m": sub.floor_depth,
        }
    if name == "BoxTiltedGridTerrainCfg":
        return {
            "max_tile_tilt_per_axis_deg": sub.tilt_range_deg * d,
            "tile_height_offset_range_m": [-sub.height_range * d / 2, sub.height_range * d / 2],
            "tile_width_m": sub.grid_width,
            "pit_depth_m": sub.floor_depth,
        }
    return {"note": f"no formula for {name}"}


def terrain_fingerprint(model) -> str:
    """Hash of terrain-body geometry, comparable between standalone and env models."""
    import mujoco

    terrain_bodies = [
        b for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("terrain")
    ]
    geoms = np.flatnonzero(np.isin(model.geom_bodyid, terrain_bodies))
    parts = [
        model.geom_type[geoms], model.geom_size[geoms].round(9),
        model.geom_pos[geoms].round(9), model.geom_quat[geoms].round(9),
    ]
    for g in geoms:
        data_id = model.geom_dataid[g]
        if data_id < 0:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            start = model.hfield_adr[data_id]
            count = model.hfield_nrow[data_id] * model.hfield_ncol[data_id]
            parts.append(model.hfield_data[start:start + count].round(6))
            parts.append(model.hfield_size[data_id].round(9))
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            start = model.mesh_vertadr[data_id]
            count = model.mesh_vertnum[data_id]
            parts.append(model.mesh_vert[start:start + count].round(6))
    return _array_sha(*parts)


# --------------------------------------------------------------------------------------
# Audit: build each terrain alone and ray-cast its surface.


def _heights(model, xs: np.ndarray, ys: np.ndarray, z_top: float = 6.0):
    """Surface height and normal-z of the terrain under a grid of vertical rays."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wp_model = mjw.put_model(model)
    wp_data = mjw.put_data(model, data, nworld=1)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_top)], -1)
    heights = np.empty(points.shape[0])
    normal_z = np.empty(points.shape[0])
    chunk = 1 << 18
    for start in range(0, points.shape[0], chunk):
        pnt = points[start:start + chunk]
        count = pnt.shape[0]
        vec = np.tile([0.0, 0.0, -1.0], (count, 1))
        dist = wp.zeros((1, count), dtype=float)
        normal = wp.zeros((1, count), dtype=wp.vec3)
        mjw.rays(
            wp_model, wp_data,
            wp.array(pnt[None].astype(np.float32), dtype=wp.vec3),
            wp.array(vec[None].astype(np.float32), dtype=wp.vec3),
            wp.types.vector(length=6, dtype=float)(-1, -1, -1, -1, -1, -1), True,
            wp.full(count, -1, dtype=int),
            dist, wp.zeros((1, count), dtype=int), normal,
        )
        dist = dist.numpy()[0]
        heights[start:start + chunk] = np.where(dist >= 0, z_top - dist, np.nan)
        normal_z[start:start + chunk] = normal.numpy()[0][:, 2]
    heights = heights.reshape(grid_x.shape)
    normal_z = normal_z.reshape(grid_x.shape)

    # The GPU kernel occasionally misses a heightfield surface that CPU mj_ray hits.
    # Re-cast every locally anomalous sample (and every miss) on the CPU, which is
    # treated as ground truth, and keep count of the corrections.
    from scipy.ndimage import median_filter

    median = median_filter(np.nan_to_num(heights, nan=-10.0), size=3, mode="nearest")
    suspects = np.argwhere(~np.isfinite(heights) | (np.abs(heights - median) > 0.03))
    corrected = 0
    geom_id = np.zeros(1, np.int32)
    normal_out = np.zeros(3)
    for i, j in suspects:
        dist = mujoco.mj_ray(
            model, data, np.array([xs[i], ys[j], z_top]), np.array([0.0, 0.0, -1.0]),
            None, 1, -1, geom_id, normal_out,
        )
        height = z_top - dist if dist >= 0 else np.nan
        if not np.isclose(height, heights[i, j], atol=1e-3, equal_nan=True):
            corrected += 1
        heights[i, j] = height
        normal_z[i, j] = normal_out[2] if dist >= 0 else np.nan
    return heights, normal_z, {"cpu_rechecked": int(len(suspects)), "cpu_corrected": corrected}


def _surface_stats(heights: np.ndarray, normal_z: np.ndarray, res: float) -> dict:
    jumps, grads = [], []
    for axis in (0, 1):
        diff = np.abs(np.diff(heights, axis=axis)).ravel()
        diff = diff[np.isfinite(diff)]
        jumps.append(diff[diff > 0.02])
        grads.append(diff[diff <= 0.02] / res)
    jumps = np.concatenate(jumps)
    grads = np.concatenate(grads)
    hit = np.isfinite(heights)
    tilt = np.degrees(np.arccos(np.clip(normal_z[hit], -1.0, 1.0)))
    finite = heights[hit]
    levels, counts = np.unique(np.round(finite / 0.005) * 0.005, return_counts=True)
    top = np.argsort(-counts)[:12]

    def q(values, p):
        return float(np.quantile(values, p)) if values.size else None

    pit = finite < -0.3
    return {
        "height_min_m": float(finite.min()),
        "height_max_m": float(finite.max()),
        "height_std_m": float(finite.std()),
        "vertical_jumps_gt_2cm": {
            "count": int(jumps.size),
            "p10_m": q(jumps, 0.10), "p50_m": q(jumps, 0.5), "p90_m": q(jumps, 0.9),
            "max_m": q(jumps, 1.0),
        },
        "smooth_gradient": {
            "p50": q(grads, 0.5), "p90": q(grads, 0.9), "p99": q(grads, 0.99),
            "max": q(grads, 1.0),
        },
        "surface_normal_tilt_deg": {
            "p50": q(tilt, 0.5), "p90": q(tilt, 0.9), "p99": q(tilt, 0.99),
        },
        "pit_area_fraction_below_-0.3m": float(pit.mean()),
        "most_common_height_levels_m": [
            [round(float(levels[i]), 3), round(float(counts[i]) / finite.size, 4)] for i in top
        ],
    }


def audit(args: argparse.Namespace) -> None:
    import mujoco
    import mjlab.tasks  # noqa: F401
    import wheeled_legged_mjlab  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.terrains import TerrainGenerator

    out = Path(args.out)
    image_dir = out / "terrain_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    reference_task = CHECKPOINTS[0].task_id
    play_cfg = load_env_cfg(reference_task, play=True)
    train_cfg = load_env_cfg(reference_task)
    for spec in CHECKPOINTS[1:]:
        other = load_env_cfg(spec.task_id)
        if other.scene.terrain != train_cfg.scene.terrain:
            raise RuntimeError(f"{spec.task_id} trains on a different terrain configuration.")
    train_gen = train_cfg.scene.terrain.terrain_generator
    report = {
        "training_generator": {
            "num_rows_difficulty_levels": train_gen.num_rows,
            "difficulty_range": list(train_gen.difficulty_range),
            "difficulty_of_row_r": "difficulty_range[0] + (hi - lo) * r / (num_rows - 1)",
            "max_row": train_gen.num_rows - 1,
            "max_difficulty": train_gen.difficulty_range[1],
            "max_init_terrain_level": train_cfg.scene.terrain.max_init_terrain_level,
            "patch_size_m": list(train_gen.size),
            "note": "terrain_levels_vel moves an env up one row after walking > size/2; "
            "levels >= num_rows are re-sampled uniformly (mjlab update_env_origins), "
            "so row num_rows-1 (difficulty 1.0) is the highest reachable level.",
        },
        "evaluation_grid": {
            "patches_per_side": GRID_PATCHES,
            "spawn_patches_per_side": SPAWN_PATCHES,
            "padding_patches": PAD_PATCHES,
            "difficulty": DIFFICULTY,
            "terrain_seed": args.seed,
            "outer_border_width_m": play_cfg.scene.terrain.terrain_generator.border_width,
        },
        "terrains": {},
    }
    for terrain in args.terrains:
        started = time.time()
        gen_cfg = eval_terrain_generator_cfg(play_cfg, train_cfg, terrain, args.seed)
        sub = next(iter(gen_cfg.sub_terrains.values()))
        spec = mujoco.MjSpec()
        TerrainGenerator(gen_cfg, device="cpu").compile(spec)
        model = spec.compile()
        patch = gen_cfg.size[0]
        half = 0.5 * GRID_PATCHES * patch
        lo = -half + PAD_PATCHES * patch
        hi = lo + SPAWN_PATCHES * patch
        # Irrational-ish offset: keeps samples off box edges that lie on 0.1 m multiples.
        xs = np.arange(lo + 0.371 * args.res, hi, args.res)
        heights, normal_z, raycast_check = _heights(model, xs, xs)
        per_patch = []
        n = int(round(patch / args.res))
        for i in range(SPAWN_PATCHES):
            for j in range(SPAWN_PATCHES):
                block = heights[i * n:(i + 1) * n, j * n:(j + 1) * n]
                per_patch.append(float(np.nanmax(block) - np.nanmin(block)))
        report["terrains"][terrain] = {
            "generator_class": type(sub).__name__,
            "training_config": {
                k: v for k, v in asdict(sub).items() if k not in ("size", "flat_patch_sampling")
            },
            "difficulty": DIFFICULTY,
            "formula_parameters_at_eval_difficulty": formula_parameters(sub, DIFFICULTY),
            "formula_parameters_at_difficulty_0": formula_parameters(sub, 0.0),
            "measured_spawn_region": _surface_stats(heights, normal_z, args.res),
            "measured_patch_height_range_m": {
                "min": min(per_patch), "median": float(np.median(per_patch)),
                "max": max(per_patch),
            },
            "raycast_resolution_m": args.res,
            "raycast_gpu_cpu_check": raycast_check,
            "num_terrain_geoms": int(model.ngeom),
            "terrain_fingerprint": terrain_fingerprint(model),
            "audit_seconds": round(time.time() - started, 1),
        }
        _save_height_image(heights, lo, hi, patch, terrain, image_dir / f"{terrain}.png")
        print(f"[audit] {terrain}: {report['terrains'][terrain]['formula_parameters_at_eval_difficulty']}",
              flush=True)
    _write_json(out / "terrain_audit.json", report)


def _save_height_image(heights, lo, hi, patch, terrain, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    extent = (lo, hi, lo, hi)
    image = axes[0].imshow(heights.T, origin="lower", extent=extent, cmap="viridis")
    for tick in np.arange(lo, hi + 1e-6, patch):
        axes[0].axvline(tick, color="w", lw=0.3)
        axes[0].axhline(tick, color="w", lw=0.3)
    axes[0].set_title(f"{terrain}: 8x8 spawn patches (d={DIFFICULTY})")
    fig.colorbar(image, ax=axes[0], label="height [m]")
    n = int(round(patch / ((hi - lo) / heights.shape[0])))
    image = axes[1].imshow(
        heights[:n, :n].T, origin="lower", extent=(lo, lo + patch, lo, lo + patch), cmap="viridis"
    )
    axes[1].set_title("first spawn patch")
    fig.colorbar(image, ax=axes[1], label="height [m]")
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Rollout: one checkpoint on one terrain class.


def _model_state_sha(module) -> str:
    return _array_sha(*[
        value.detach().float().cpu().numpy() for _, value in sorted(module.state_dict().items())
    ])


def rollout(args: argparse.Namespace) -> None:
    import torch
    import mjlab.tasks  # noqa: F401
    import wheeled_legged_mjlab  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from tensordict import TensorDict

    from wheeled_legged_mjlab.tasks.velocity import mdp

    spec = _checkpoint(args.ckpt)
    run_name = spec.name + ("__zero_actions" if args.zero_actions else "")
    run_dir = Path(args.out) / "raw" / args.terrain / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    configure_torch_backends(allow_tf32=False, deterministic=True)
    device = args.device
    num_envs, num_steps = args.num_envs, args.num_steps
    ckpt_path = ROOT / spec.path
    ckpt_md5 = _file_md5(ckpt_path)

    env_cfg, train_env_cfg = make_eval_env_cfg(spec.task_id, args.terrain, num_envs, args.seed)
    agent_cfg = load_rl_cfg(spec.task_id)
    dump_yaml(run_dir / "env_cfg.yaml", asdict(env_cfg))
    dump_yaml(run_dir / "agent_cfg.yaml", asdict(agent_cfg))
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    step_dt = env.step_dt
    terrain = env.scene.terrain
    rows, cols = spawn_layout(num_envs)
    rows_t = torch.as_tensor(rows, device=device, dtype=torch.long)
    cols_t = torch.as_tensor(cols, device=device, dtype=torch.long)
    terrain.terrain_levels[:] = rows_t
    terrain.terrain_types[:] = cols_t
    terrain.env_origins[:] = terrain.terrain_origins[rows_t, cols_t]
    env_origins = terrain.env_origins.cpu().numpy().copy()

    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(spec.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapper, asdict(agent_cfg), device=device)
    runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    policy_class = type(policy).__name__
    policy_sha_before = _model_state_sha(policy)

    # Everything above may consume RNG differently per checkpoint (model init). The
    # evaluation episode starts from a clean counter and a fresh seed so all
    # checkpoints see identical initial states and command sequences.
    env.common_step_counter = 0
    obs_dict, _ = env.reset(seed=args.seed)
    obs = TensorDict(obs_dict, batch_size=[num_envs])
    policy.reset()

    robot = env.scene["robot"]
    command = env.command_manager.get_term("twist")
    scan = env.scene["terrain_scan"]
    manager = env.termination_manager
    failure_terms = [n for n in manager.active_terms if not manager.get_term_cfg(n).time_out]
    if tuple(failure_terms) != FAILURE_TERMS:
        raise RuntimeError(f"Unexpected failure terms: {failure_terms}")
    initial = {
        "root_pos_w": robot.data.root_link_pos_w, "root_quat_w": robot.data.root_link_quat_w,
        "root_lin_vel_w": robot.data.root_link_lin_vel_w,
        "root_ang_vel_w": robot.data.root_link_ang_vel_w,
        "joint_pos": robot.data.joint_pos, "joint_vel": robot.data.joint_vel,
        "command_w": command.command_w, "heading_target": command.heading_target,
        "is_standing": command.is_standing_env, "is_forward": command.is_forward_env,
        "command_time_left": command.time_left,
    }
    initial = {k: v.detach().cpu().numpy().copy() for k, v in initial.items()}
    model = env.sim.model
    dr_sha = _array_sha(*[
        getattr(model, name).cpu().numpy() for name in (
            "geom_friction", "body_mass", "body_ipos", "body_inertia",
            "actuator_gainprm", "actuator_biasprm",
        )
    ], robot.data.encoder_bias.cpu().numpy())
    invalid_start = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for name in failure_terms:
        term_cfg = manager.get_term_cfg(name)
        invalid_start |= term_cfg.func(env, **term_cfg.params)

    # Diagnostic only (not a failure criterion here): the training-time
    # world-command tracking termination, evaluated in shadow from step 0.
    track_cfg = copy.deepcopy(train_env_cfg.terminations["world_command_tracking_failure"])
    track_params = dict(track_cfg.params, activation_step=0)
    track_term = mdp.world_command_tracking_failure(track_cfg, env)

    def buffer(*shape, dtype=torch.float32):
        return torch.zeros((num_steps, num_envs, *shape), dtype=dtype, device=device)

    series = {
        "pos_w": buffer(3), "projected_gravity_b": buffer(3), "ang_vel_b": buffer(3),
        "lin_vel_b": buffer(3), "lin_vel_w": buffer(3), "command_b": buffer(3),
        "command_w": buffer(3), "heading_target": buffer(), "heading_w": buffer(),
        "command_counter": buffer(dtype=torch.int16), "is_standing": buffer(dtype=torch.bool),
        "terrain_relief": buffer(), "valid": buffer(dtype=torch.bool),
        "failure_flags": buffer(len(failure_terms), dtype=torch.bool),
        "tracking_failure_shadow": buffer(dtype=torch.bool),
        "actions": buffer(env.action_manager.total_action_dim),
    }
    alive = ~invalid_start
    fail_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    fail_flags = torch.zeros((num_envs, len(failure_terms)), dtype=torch.bool, device=device)

    clip = agent_cfg.clip_actions
    rollout_started = time.time()
    for k in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        if args.zero_actions:  # Debug path for checking failure handling.
            actions = torch.zeros_like(actions)
        actions = torch.clamp(actions, -clip, clip) if clip is not None else actions
        series["actions"][k] = actions
        # auto_reset=False: done robots are deliberately left un-reset, so drop the
        # pending-reset guard instead of calling reset(); their data is masked below.
        env._manual_reset_pending.zero_()
        obs, _, _, _ = wrapper.step(actions)
        if manager.time_outs.any():
            raise RuntimeError("Unexpected time-out inside the evaluation window.")
        terminated = manager.terminated.clone()
        flags = torch.stack([manager.get_term(n) for n in failure_terms], dim=-1)
        hit = scan.data.distances >= 0.0
        hit_z = scan.data.hit_pos_w[..., 2]
        relief = (
            torch.where(hit, hit_z, -torch.inf).amax(-1)
            - torch.where(hit, hit_z, torch.inf).amin(-1)
        )
        series["pos_w"][k] = robot.data.root_link_pos_w
        series["projected_gravity_b"][k] = robot.data.projected_gravity_b
        series["ang_vel_b"][k] = robot.data.root_link_ang_vel_b
        series["lin_vel_b"][k] = robot.data.root_link_lin_vel_b
        series["lin_vel_w"][k] = robot.data.root_link_lin_vel_w
        series["command_b"][k] = command.command
        series["command_w"][k] = command.command_w
        series["heading_target"][k] = command.heading_target
        series["heading_w"][k] = robot.data.heading_w
        series["command_counter"][k] = command.command_counter.to(torch.int16)
        series["is_standing"][k] = command.is_standing_env
        series["terrain_relief"][k] = torch.where(hit.any(-1), relief, torch.nan)
        series["tracking_failure_shadow"][k] = track_term(env, **track_params)
        newly_failed = alive & terminated
        fail_step[newly_failed] = k + 1
        fail_flags[newly_failed] = flags[newly_failed]
        series["valid"][k] = alive & ~terminated
        series["failure_flags"][k] = flags
        alive &= ~terminated
    rollout_seconds = time.time() - rollout_started
    policy_sha_after = _model_state_sha(policy)
    if policy_sha_after != policy_sha_before:
        raise RuntimeError("Policy parameters changed during evaluation.")

    data = {k: v.cpu().numpy() for k, v in series.items()}
    data.update(
        invalid_start=invalid_start.cpu().numpy(), fail_step=fail_step.cpu().numpy(),
        fail_flags=fail_flags.cpu().numpy(), spawn_row=rows, spawn_col=cols,
        env_origins=env_origins, step_dt=np.float64(step_dt),
        terrain_origins=terrain.terrain_origins.cpu().numpy(),
    )
    np.savez_compressed(run_dir / "timeseries.npz", **data)
    np.savez_compressed(run_dir / "initial_state.npz", **initial)
    patch_size = float(env_cfg.scene.terrain.terrain_generator.size[0])
    trajectories, summary = compute_metrics(data, initial, patch_size)
    command_sha = _array_sha(
        data["command_w"][..., :2], data["heading_target"], data["command_counter"],
        data["is_standing"],
    )
    env_model_fingerprint = terrain_fingerprint(env.sim.mj_model)
    group = spec.group + ("__zero_actions" if args.zero_actions else "")
    for row in trajectories:
        row.update(terrain=args.terrain, run=run_name, group=group, train_seed=spec.train_seed)
    _write_csv(run_dir / "trajectories.csv", trajectories)
    meta = {
        "terrain": args.terrain,
        "run": run_name,
        "group": group,
        "task_id": spec.task_id,
        "train_seed": spec.train_seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": spec.path,
        "checkpoint_md5_before": ckpt_md5,
        "checkpoint_md5_after": _file_md5(ckpt_path),
        "policy_class": policy_class,
        "policy_state_sha256": policy_sha_before,
        "eval_seed": args.seed,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "step_dt_s": step_dt,
        "physics_dt_s": env.physics_dt,
        "decimation": env_cfg.decimation,
        "clip_actions": agent_cfg.clip_actions,
        "failure_terms": failure_terms,
        "terrain_fingerprint": env_model_fingerprint,
        "startup_dr_sha256": dr_sha,
        "initial_state_sha256": _array_sha(*[initial[k] for k in sorted(initial)]),
        "command_sequence_sha256": command_sha,
        "wall_seconds_total": round(time.time() - started, 1),
        "wall_seconds_rollout": round(rollout_seconds, 1),
        "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else device,
        "summary": summary,
    }
    _write_json(run_dir / "summary.json", meta)
    env.close()
    print(json.dumps(summary, indent=2), flush=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------------------
# Metrics (shared by rollout and the offline recomputation check).


def per_step_errors(data: dict) -> dict[str, np.ndarray]:
    gravity = data["projected_gravity_b"].astype(np.float64)
    ang = data["ang_vel_b"].astype(np.float64)
    cmd_b = data["command_b"].astype(np.float64)
    cmd_w = data["command_w"].astype(np.float64)
    return {
        "orientation_error": np.linalg.norm(gravity - np.array([0.0, 0.0, -1.0]), axis=-1),
        "ang_vel_error": np.linalg.norm(ang[..., :2], axis=-1),
        "vel_track_error": np.linalg.norm(cmd_b[..., :2] - data["lin_vel_b"][..., :2], axis=-1),
        "vel_track_error_world": np.linalg.norm(
            cmd_w[..., :2] - data["lin_vel_w"][..., :2], axis=-1
        ),
        "yaw_rate_error": np.abs(cmd_b[..., 2] - ang[..., 2]),
    }


def compute_metrics(data: dict, initial: dict, patch_size: float) -> tuple[list[dict], dict]:
    valid = data["valid"]
    num_steps, num_envs = valid.shape
    dt = float(data["step_dt"])
    errors = per_step_errors(data)
    n_valid = valid.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # np.where, not multiplication: post-failure states may be non-finite.
        means = {
            k: np.where(n_valid > 0, np.where(valid, v, 0.0).sum(0) / n_valid, np.nan)
            for k, v in errors.items()
        }
    in_grid, on_platform = grid_membership(data["pos_w"][..., :2], patch_size)
    left_grid = (~in_grid & valid).any(0)
    left_platform = (~on_platform & valid).any(0)
    relief = data["terrain_relief"]
    local_flat = (relief < LOCAL_FLAT_RELIEF_M) & np.isfinite(relief)
    displacement = np.linalg.norm(
        data["pos_w"][..., :2] - initial["root_pos_w"][None, :, :2], axis=-1
    )
    shadow = data["tracking_failure_shadow"] & valid
    shadow_step = np.where(shadow.any(0), shadow.argmax(0) + 1, -1)
    fail_step = data["fail_step"]
    valid_start = ~data["invalid_start"]
    evaluated = valid_start & ~left_grid
    success = evaluated & (fail_step < 0)
    counters = data["command_counter"]
    resample_step = np.where(
        (counters[-1] > counters[0]), (counters != counters[0]).argmax(0) + 1, -1
    )
    rows = []
    for i in range(num_envs):
        nv = int(n_valid[i])
        rows.append({
            "env_id": i,
            "spawn_row": int(data["spawn_row"][i]),
            "spawn_col": int(data["spawn_col"][i]),
            "valid_start": bool(valid_start[i]),
            "left_terrain_grid": bool(left_grid[i]),
            "evaluated": bool(evaluated[i]),
            "success": bool(success[i]),
            "fail_step": int(fail_step[i]),
            "survival_time_s": round(
                (fail_step[i] if fail_step[i] > 0 else num_steps) * dt, 4
            ),
            **{
                f"fail_{name}": bool(data["fail_flags"][i, j])
                for j, name in enumerate(FAILURE_TERMS)
            },
            "n_valid_samples": nv,
            **{k: float(v[i]) for k, v in means.items()},
            "tracking_failure_shadow_step": int(shadow_step[i]),
            "left_center_platform": bool(left_platform[i]),
            "max_displacement_m": float(displacement[:, i][valid[:, i]].max()) if nv else 0.0,
            "frac_on_center_platform": float(on_platform[:, i][valid[:, i]].mean()) if nv else math.nan,
            "frac_local_flat": float(local_flat[:, i][valid[:, i]].mean()) if nv else math.nan,
            "cmd0_w_x": float(initial["command_w"][i, 0]),
            "cmd0_w_y": float(initial["command_w"][i, 1]),
            "cmd0_heading_target": float(initial["heading_target"][i]),
            "cmd0_standing": bool(initial["is_standing"][i]),
            "cmd0_forward": bool(initial["is_forward"][i]),
            "cmd_resample_step": int(resample_step[i]),
        })
    return rows, summarize_rows(rows, num_steps, dt)


def _mean_ci(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "n": 0}
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size) if values.size > 1 else math.nan
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "n": int(values.size)}


def _wilson(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.96
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (center - half), 100 * (center + half)


def _subset_summary(rows: list[dict]) -> dict:
    with_samples = [r for r in rows if r["n_valid_samples"] > 0]
    return {
        "n": len(rows),
        "success_rate_pct": 100.0 * sum(r["success"] for r in rows) / len(rows) if rows else math.nan,
        **{
            key: float(np.mean([r[key] for r in with_samples])) if with_samples else math.nan
            for key in ("orientation_error", "ang_vel_error", "vel_track_error")
        },
    }


def summarize_rows(rows: list[dict], num_steps: int, dt: float) -> dict:
    def col(name, subset=None):
        return np.array([r[name] for r in rows if subset is None or subset(r)], dtype=float)

    evaluated = [r for r in rows if r["evaluated"]]
    n_eval = len(evaluated)
    n_success = sum(r["success"] for r in evaluated)
    failed = [r for r in evaluated if not r["success"]]
    shadow_ok = sum(r["success"] and r["tracking_failure_shadow_step"] < 0 for r in evaluated)
    has_samples = lambda r: r["evaluated"] and r["n_valid_samples"] > 0  # noqa: E731
    low, high = _wilson(n_success, n_eval)
    summary = {
        "n_started": len(rows),
        "n_invalid_start": sum(not r["valid_start"] for r in rows),
        "n_left_terrain_grid": sum(r["left_terrain_grid"] for r in rows),
        "n_evaluated": n_eval,
        "n_success": n_success,
        "success_rate_pct": 100.0 * n_success / n_eval if n_eval else math.nan,
        "success_rate_wilson95_pct": [low, high],
        "n_with_error_samples": sum(has_samples(r) for r in rows),
        "orientation_error": _mean_ci(col("orientation_error", has_samples)),
        "ang_vel_error_rad_s": _mean_ci(col("ang_vel_error", has_samples)),
        "vel_track_error_m_s": _mean_ci(col("vel_track_error", has_samples)),
        "vel_track_error_world_m_s": _mean_ci(col("vel_track_error_world", has_samples)),
        "yaw_rate_error_rad_s": _mean_ci(col("yaw_rate_error", has_samples)),
        "rec_error_cm": None,
        "survival_time_s_all": _mean_ci(col("survival_time_s", lambda r: r["evaluated"])),
        "survival_time_s_failed": _mean_ci(np.array([r["survival_time_s"] for r in failed])),
        "failure_reasons": {
            name: sum(r[f"fail_{name}"] for r in failed) for name in FAILURE_TERMS
        },
        "success_rate_with_tracking_shadow_pct": 100.0 * shadow_ok / n_eval if n_eval else math.nan,
        # Supplementary subset: trajectories whose base left the flat central spawn
        # platform at least once (standing / very slow commands never do).
        "exposed_subset": _subset_summary(
            [r for r in evaluated if r["left_center_platform"]]
        ),
        "frac_on_center_platform": _mean_ci(col("frac_on_center_platform", has_samples)),
        "frac_local_flat": _mean_ci(col("frac_local_flat", has_samples)),
        "max_displacement_m": _mean_ci(col("max_displacement_m", has_samples)),
        "max_displacement_m_max": float(col("max_displacement_m").max()),
        "episode_s": num_steps * dt,
    }
    return summary


# --------------------------------------------------------------------------------------
# Aggregation.

METRIC_COLUMNS = (
    ("success_rate_pct", "Success Rate ↑ (%)", 2),
    ("orientation_error", "Orientation Error ↓ (–)", 3),
    ("ang_vel_error_rad_s", "Angular Velocity Error ↓ (rad/s)", 3),
    ("vel_track_error_m_s", "Velocity Tracking Error ↓ (m/s)", 3),
)


def _metric_value(summary: dict, key: str) -> float:
    value = summary[key]
    return float(value["mean"] if isinstance(value, dict) else value)


def aggregate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    runs = []
    for path in sorted((out / "raw").glob("*/*/summary.json")):
        runs.append(json.loads(path.read_text()))
    if not runs:
        raise SystemExit(f"No runs under {out / 'raw'}")
    terrain_order = {t: i for i, t in enumerate(TERRAINS)}
    group_order = {c.group: i for i, c in enumerate(CHECKPOINTS)}
    runs.sort(key=lambda r: (terrain_order.get(r["terrain"], 99), group_order.get(r["group"], 99)))
    results = out / "results"
    results.mkdir(exist_ok=True)

    # Per run (terrain x checkpoint).
    per_run = []
    for r in runs:
        s = r["summary"]
        per_run.append({
            "terrain": r["terrain"], "group": r["group"], "run": r["run"],
            "train_seed": r["train_seed"], "task_id": r["task_id"],
            "n_started": s["n_started"], "n_evaluated": s["n_evaluated"],
            "n_left_terrain_grid": s["n_left_terrain_grid"],
            "n_invalid_start": s["n_invalid_start"],
            "success_rate_pct": s["success_rate_pct"],
            "success_wilson95_low": s["success_rate_wilson95_pct"][0],
            "success_wilson95_high": s["success_rate_wilson95_pct"][1],
            **{
                f"{key}{suffix}": s[key][field]
                for key in (
                    "orientation_error", "ang_vel_error_rad_s", "vel_track_error_m_s",
                    "vel_track_error_world_m_s", "yaw_rate_error_rad_s",
                    "survival_time_s_all", "survival_time_s_failed",
                    "frac_on_center_platform", "frac_local_flat", "max_displacement_m",
                )
                for suffix, field in (("", "mean"), ("_ci95_low", "ci95_low"),
                                      ("_ci95_high", "ci95_high"), ("_n", "n"))
            },
            "rec_error_cm": "N/A",
            **{f"fail_{k}": v for k, v in s["failure_reasons"].items()},
            "success_rate_with_tracking_shadow_pct": s["success_rate_with_tracking_shadow_pct"],
            **{f"exposed_{k}": v for k, v in s["exposed_subset"].items()},
            "max_displacement_m_max": s["max_displacement_m_max"],
            "eval_seed": r["eval_seed"], "num_envs": r["num_envs"], "num_steps": r["num_steps"],
            "step_dt_s": r["step_dt_s"], "checkpoint": r["checkpoint"],
            "checkpoint_md5": r["checkpoint_md5_before"],
        })
    _write_csv(results / "per_run.csv", per_run)

    # Per group: mean +- sample std over training seeds (runs of the group).
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        grouped.setdefault((r["terrain"], r["group"]), []).append(r)
    by_group = []
    for (terrain, group), members in grouped.items():
        row = {"terrain": terrain, "group": group, "n_train_seeds": len(members),
               "train_seeds": ";".join(m["train_seed"] for m in members),
               "n_trajectories_total": sum(m["summary"]["n_evaluated"] for m in members)}
        for key, _, _ in METRIC_COLUMNS:
            values = np.array([_metric_value(m["summary"], key) for m in members])
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        row["rec_error_cm"] = "N/A"
        by_group.append(row)
    _write_csv(results / "by_group.csv", by_group)

    # Identity checks across checkpoints on each terrain.
    checks = {}
    for terrain in sorted({r["terrain"] for r in runs}, key=lambda t: terrain_order.get(t, 99)):
        members = [r for r in runs if r["terrain"] == terrain]
        checks[terrain] = {
            key: len({m[key] for m in members}) == 1
            for key in (
                "terrain_fingerprint", "startup_dr_sha256", "initial_state_sha256",
                "command_sequence_sha256", "eval_seed", "num_envs", "num_steps",
                "script_sha256",
            )
        }
        checks[terrain]["runs"] = [m["run"] for m in members]
        checks[terrain]["checkpoints_unchanged"] = all(
            m["checkpoint_md5_before"] == m["checkpoint_md5_after"] for m in members
        )
    audit_path = out / "terrain_audit.json"
    if audit_path.exists():
        audit_report = json.loads(audit_path.read_text())
        for terrain, check in checks.items():
            audited = audit_report["terrains"].get(terrain, {}).get("terrain_fingerprint")
            check["matches_audited_terrain"] = all(
                r["terrain_fingerprint"] == audited for r in runs if r["terrain"] == terrain
            )
    _write_json(results / "identity_checks.json", checks)
    diagnostics = [
        {"terrain": r["terrain"], "run": r["run"], "group": r["group"],
         **_timeseries_diagnostics(out / "raw" / r["terrain"] / r["run"])}
        for r in runs
    ]
    _write_csv(results / "diagnostics_by_run.csv", diagnostics)
    _write_markdown(results / "table.md", by_group, per_run, audit_path, checks, diagnostics)
    if audit_path.exists():
        _write_terrain_markdown(results / "terrains.md", json.loads(audit_path.read_text()))
    print((results / "table.md").read_text())


COMMAND_CLASSES = ("forward", "backward", "lateral", "standing")


def _timeseries_diagnostics(run_dir: Path) -> dict:
    """Supplementary statistics recomputed from the saved time series.

    * Success by the direction of the first command in the target-heading frame
      (the frame the robot tracks once its heading has converged).
    * Fall-only survival: failure = fell_over or non_finite_physics at any step,
      ignoring illegal_contact. Valid because terminated robots are never reset,
      so their continued motion is the exact counterfactual without that criterion.
    """
    data = np.load(run_dir / "timeseries.npz")
    initial = np.load(run_dir / "initial_state.npz")
    with (run_dir / "trajectories.csv").open() as stream:
        evaluated = np.array([row["evaluated"] == "True" for row in csv.DictReader(stream)])
    success = evaluated & (data["fail_step"] < 0)
    flags = data["failure_flags"]
    fell = (
        flags[..., FAILURE_TERMS.index("fell_over")]
        | flags[..., FAILURE_TERMS.index("non_finite_physics")]
    ).any(0)
    cmd = initial["command_w"][:, :2]
    heading = initial["heading_target"]
    vx = np.cos(heading) * cmd[:, 0] + np.sin(heading) * cmd[:, 1]
    vy = -np.sin(heading) * cmd[:, 0] + np.cos(heading) * cmd[:, 1]
    classes = np.where(
        initial["is_standing"], "standing",
        np.where(np.abs(vx) >= np.abs(vy), np.where(vx > 0, "forward", "backward"), "lateral"),
    )
    n_eval = int(evaluated.sum())
    row = {
        "n_evaluated": n_eval,
        "success_rate_pct": 100.0 * success.sum() / n_eval,
        "fall_only_success_rate_pct": 100.0 * (evaluated & ~fell).sum() / n_eval,
    }
    for name in COMMAND_CLASSES:
        mask = evaluated & (classes == name)
        row[f"n_{name}"] = int(mask.sum())
        row[f"success_{name}_pct"] = 100.0 * success[mask].sum() / mask.sum() if mask.any() else math.nan
    return row


def _write_terrain_markdown(path: Path, report: dict) -> None:
    def short(params: dict) -> str:
        keep = {k: v for k, v in params.items() if k != "note" and v != ""}
        text = "; ".join(
            f"{k}={[round(x, 4) for x in v] if isinstance(v, list) else round(v, 4) if isinstance(v, float) else v}"
            for k, v in keep.items()
        )
        return "; ".join(part for part in (text, params.get("note")) if part).replace("|", "/")

    gen = report["training_generator"]
    lines = [
        "# Evaluation terrains (maximum training difficulty)",
        "",
        f"Training generator: {gen['num_rows_difficulty_levels']} curriculum rows, difficulty of row r = "
        f"{gen['difficulty_of_row_r']}; highest reachable row {gen['max_row']} -> difficulty "
        f"{gen['max_difficulty']}. {gen['note']}",
        "",
        "Evaluation grid: every patch of a 12 x 12 grid of 8 m patches is an independent random "
        "instance of one terrain class at d = 1.0 (terrain seed "
        f"{report['evaluation_grid']['terrain_seed']}); robots spawn on the inner 8 x 8 patches.",
        "",
        "| Terrain | Generator | Parameters at d = 1.0 (from generator code) | Parameters at d = 0 | "
        "Measured height min / max (m) | Measured vertical jumps > 2 cm p50 / max (m) | "
        "Measured smooth gradient p99 / max | Surface tilt p90 / p99 (deg) | Pit area (< -0.3 m) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, t in report["terrains"].items():
        m = t["measured_spawn_region"]
        jumps = m["vertical_jumps_gt_2cm"]
        grad = m["smooth_gradient"]
        tilt = m["surface_normal_tilt_deg"]

        def f(x, digits=3):
            return "–" if x is None else f"{x:.{digits}f}"

        lines.append(
            f"| {name} | {t['generator_class']} | {short(t['formula_parameters_at_eval_difficulty'])} | "
            f"{short(t['formula_parameters_at_difficulty_0'])} | "
            f"{f(m['height_min_m'])} / {f(m['height_max_m'])} | {f(jumps['p50_m'])} / {f(jumps['max_m'])} | "
            f"{f(grad['p99'])} / {f(grad['max'])} | {f(tilt['p90'], 1)} / {f(tilt['p99'], 1)} | "
            f"{m['pit_area_fraction_below_-0.3m']:.3f} |"
        )
    lines += [
        "",
        "Measurements: vertical rays on a 4 cm grid over all 64 spawn patches "
        "(MuJoCo Warp; samples that disagree with their 3 x 3 neighbourhood are re-cast with CPU "
        "mj_ray). Heightfield steps are linear ramps across one 0.15 m heightfield cell, so "
        "discrete-obstacle edges appear as steep gradients / tilts rather than vertical jumps.",
        "",
    ]
    path.write_text("\n".join(lines))


def _fmt(mean: float, std: float, digits: int, n: int) -> str:
    if not np.isfinite(mean):
        return "–"
    if n > 1:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def _terrain_label(terrain: str, audit_report: dict | None) -> str:
    if not audit_report or terrain not in audit_report.get("terrains", {}):
        return terrain
    p = audit_report["terrains"][terrain]["formula_parameters_at_eval_difficulty"]
    parts = {
        "flat": "",
        "discrete_obstacles": f"±{p.get('obstacle_height_m', 0) * 100:.1f} cm",
        "random_rough": "noise {:.0f}–{:.0f} cm".format(*(100 * v for v in p.get("noise_range_m", [0, 0]))),
        "hf_pyramid_slope": f"slope {p.get('slope_deg', 0):.1f}° up",
        "hf_pyramid_slope_inv": f"slope {p.get('slope_deg', 0):.1f}° pit",
        "pyramid_stair": f"step {p.get('step_height_m', 0) * 100:.0f} cm",
        "pyramid_stair_inv": f"step {p.get('step_height_m', 0) * 100:.0f} cm",
        "random_stairs": "step {:.0f}–{:.0f} cm".format(*(100 * v for v in p.get("step_height_range_m", [0, 0]))),
        "random_spread": "boxes {:.0f}–{:.0f} cm".format(*(100 * v for v in p.get("box_height_range_m", [0, 0]))),
        "stepping_stones": f"gap {p.get('nominal_gap_m', 0) * 100:.0f} cm",
        "tilted_grid": f"tilt ≤{p.get('max_tile_tilt_per_axis_deg', 0):.0f}°",
    }.get(terrain, "")
    return f"{terrain} ({parts})" if parts else terrain


def _write_markdown(
    path: Path, by_group, per_run, audit_path: Path, checks: dict, diagnostics: list[dict]
) -> None:
    audit_report = json.loads(audit_path.read_text()) if audit_path.exists() else None
    lines = [
        "# LIPM-style terrain evaluation (Table IV format)",
        "",
        "Maximum training difficulty (d = 1.0) per terrain class; 10 s per trajectory "
        "(500 policy steps x 0.02 s); deterministic policy. With one training seed per "
        "group (n = 1) values are single-run estimates, not mean ± std.",
        "",
        "| Terrain (d=1.0) | Method | " + " | ".join(label for _, label, _ in METRIC_COLUMNS)
        + " | Rec Error ↓ (cm) | Trajectories | Training seeds |",
        "|---|---|" + "---:|" * (len(METRIC_COLUMNS) + 3),
    ]
    for row in by_group:
        n = row["n_train_seeds"]
        cells = [
            _fmt(row[f"{key}_mean"], row[f"{key}_std"], digits, n)
            for key, _, digits in METRIC_COLUMNS
        ]
        lines.append(
            f"| {_terrain_label(row['terrain'], audit_report)} | {row['group']} | "
            + " | ".join(cells)
            + f" | N/A | {row['n_trajectories_total']} | {n} |"
        )
    lines += [
        "",
        "## Supplementary (per run; trajectory-level 95% CI reflects evaluation sampling only, "
        "not training-seed variability)",
        "",
        "| Terrain | Run | Success % [Wilson 95%] | Vel. err. world (m/s) | Yaw-rate err. (rad/s) | "
        "Mean survival (s) | Failures: fell_over / illegal_contact / non_finite | "
        "Success % incl. tracking-failure shadow | Frac. locally flat | Left grid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in per_run:
        lines.append(
            f"| {r['terrain']} | {r['run']} | {r['success_rate_pct']:.2f} "
            f"[{r['success_wilson95_low']:.1f}, {r['success_wilson95_high']:.1f}] | "
            f"{r['vel_track_error_world_m_s']:.3f} | {r['yaw_rate_error_rad_s']:.3f} | "
            f"{r['survival_time_s_all']:.2f} | {r['fail_fell_over']} / "
            f"{r['fail_illegal_contact']} / {r['fail_non_finite_physics']} | "
            f"{r['success_rate_with_tracking_shadow_pct']:.2f} | "
            f"{r['frac_local_flat']:.3f} | {r['n_left_terrain_grid']} |"
        )
    lines += [
        "",
        "## Supplementary: exposed subset (trajectories whose base left the flat central "
        "2 m x 2 m spawn platform; excludes standing / very slow commands)",
        "",
        "| Terrain | Run | N exposed | Success ↑ (%) | Orientation Error ↓ | "
        "Angular Velocity Error ↓ (rad/s) | Velocity Tracking Error ↓ (m/s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in per_run:
        lines.append(
            f"| {r['terrain']} | {r['run']} | {r['exposed_n']} | "
            f"{r['exposed_success_rate_pct']:.2f} | {r['exposed_orientation_error']:.3f} | "
            f"{r['exposed_ang_vel_error']:.3f} | {r['exposed_vel_track_error']:.3f} |"
        )
    lines += [
        "",
        "## Supplementary: success by first-command direction (target-heading frame) and "
        "fall-only survival (illegal_contact not counted as failure)",
        "",
        "| Terrain | Run | Success (%) | Fall-only survival (%) | "
        + " | ".join(f"{c} n / success (%)" for c in COMMAND_CLASSES) + " |",
        "|---|---|---:|---:|" + "---:|" * len(COMMAND_CLASSES),
    ]
    for d in diagnostics:
        lines.append(
            f"| {d['terrain']} | {d['run']} | {d['success_rate_pct']:.2f} | "
            f"{d['fall_only_success_rate_pct']:.2f} | "
            + " | ".join(f"{d[f'n_{c}']} / {d[f'success_{c}_pct']:.1f}" for c in COMMAND_CLASSES)
            + " |"
        )
    all_ok = all(
        all(v for k, v in c.items() if isinstance(v, bool)) for c in checks.values()
    )
    lines += [
        "",
        f"Identity checks across checkpoints (terrain, startup DR, initial states, command "
        f"sequences, unchanged checkpoints): {'all passed' if all_ok else 'FAILED, see identity_checks.json'}.",
        "",
    ]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
# Orchestration.


def run(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    import torch

    config = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "protocol": PROTOCOL,
        "eval_seed": args.seed,
        "terrain_seed": args.seed,
        "difficulty": DIFFICULTY,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "grid": {"patches_per_side": GRID_PATCHES, "spawn_patches_per_side": SPAWN_PATCHES,
                 "envs_per_spawn_patch": args.num_envs / SPAWN_PATCHES ** 2},
        "terrains": args.terrains,
        "checkpoints": [
            {**asdict(_checkpoint(name)), "md5": _file_md5(ROOT / _checkpoint(name).path)}
            for name in args.ckpts
        ],
        "git": _git_state(),
        "script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "snapshot": "lipm_eval.py.snapshot",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            **{
                pkg: _pkg_version(pkg)
                for pkg in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")
            },
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _write_json(out / "config.json", config)
    (out / "lipm_eval.py.snapshot").write_bytes(Path(__file__).read_bytes())
    common = ["--out", str(out), "--seed", str(args.seed)]
    jobs = []
    if args.force or not (out / "terrain_audit.json").exists():
        jobs.append(("audit", [*common, "--terrains", *args.terrains]))
    for terrain in args.terrains:
        for ckpt in args.ckpts:
            if not args.force and (out / "raw" / terrain / ckpt / "summary.json").exists():
                continue
            jobs.append((f"{terrain}__{ckpt}", [
                *common, "--terrain", terrain, "--ckpt", ckpt,
                "--num-envs", str(args.num_envs), "--num-steps", str(args.num_steps),
            ]))
    for index, (name, job_args) in enumerate(jobs, 1):
        sub = "audit" if name == "audit" else "rollout"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), sub, *job_args]
        log_path = out / "logs" / f"{name}.log"
        print(f"[{index}/{len(jobs)}] {name} -> {log_path}", flush=True)
        started = time.time()
        with log_path.open("w") as log:
            log.write(" ".join(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"    exit={result.returncode} {time.time() - started:.0f}s", flush=True)
        if result.returncode != 0:
            raise SystemExit(f"{name} failed; see {log_path}")
    aggregate(args)


def _pkg_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name, func):
        sub = subparsers.add_parser(name)
        sub.add_argument("--out", required=True)
        sub.add_argument("--seed", type=int, default=EVAL_SEED)
        sub.set_defaults(func=func)
        return sub

    sub = add("audit", audit)
    sub.add_argument("--terrains", nargs="+", default=list(TERRAINS), choices=TERRAINS)
    sub.add_argument("--res", type=float, default=0.04)
    sub = add("rollout", rollout)
    sub.add_argument("--terrain", required=True, choices=TERRAINS)
    sub.add_argument("--ckpt", required=True, choices=[c.name for c in CHECKPOINTS])
    sub.add_argument("--num-envs", type=int, default=NUM_ENVS)
    sub.add_argument("--num-steps", type=int, default=NUM_STEPS)
    sub.add_argument("--device", default="cuda:0")
    sub.add_argument("--zero-actions", action="store_true",
                     help="Debug only: send zero actions to exercise failure handling.")
    add("aggregate", aggregate)
    sub = add("run", run)
    sub.add_argument("--terrains", nargs="+", default=list(TERRAINS), choices=TERRAINS)
    sub.add_argument("--ckpts", nargs="+", default=[c.name for c in CHECKPOINTS],
                     choices=[c.name for c in CHECKPOINTS])
    sub.add_argument("--num-envs", type=int, default=NUM_ENVS)
    sub.add_argument("--num-steps", type=int, default=NUM_STEPS)
    sub.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
