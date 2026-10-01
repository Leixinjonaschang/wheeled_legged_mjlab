"""CPU terrain audit of the mode_switch_v2 flat-rough-flat courses (no GPU, no policy).

For every rough type the exact rollout terrain is built on the CPU (make_env_cfg(..., levels=4),
eval seed; 4 difficulty rows x 16 lanes) and probed with CPU mj_ray. Per difficulty level:
  flat     max |z| of the flat segments on 48 lines (16 lanes x lane centre and +-0.3 m), the
           cross-width lines at x = 0.5, 2, 3.5, 9, 11 m and the full-width lane-0 map, over
           (a) all of [0, 4.0) and (7.6, 11.6], (b) the judged windows [1.8, 3.2) and [9.0, 11.6),
           (c) all flat but the 0.3 m aprons (3.7, 4.0) and (7.6, 7.9). random_spread boxes are
           yawed and overhang the aprons (as they overhang the training border): for it (a) may
           be > 0 (overhang height and extent reported); every other terrain needs (a) = 0.
  seams    height jump at x = 4.0 and 7.6 m (z(x + 0.5 mm) - z(x - 0.5 mm) minus the local slope, so
           heightfield ramps starting at the seam count as continuous) every 1 cm across the full
           width of all lanes, versus the joint the training geometry has at its own border.
  holes    ray misses and narrow dips (< 3 cm wide, > 1 cm deep) in 1 mm profiles within +-5 cm of
           both seams.
  section  min / max height in x in [4.0, 7.6] m.
  feature  key feature size measured on the lane centre lines / the compiled model versus the
           training formula at difficulty d; for the random terrains also a height-distribution
           comparison with a training patch (training cfg, 8 m, same d, platform excluded).
           random_spread: boxes per lane, raised fraction on the robots' path band (|y - 1.8| <=
           0.46 m, x in [4.2, 7.4]) and the centre-pocket check (raised fraction at |x - 5.8| <=
           0.2 m vs x in [4.4, 5.4] u [6.2, 7.2], pooled over the 4 levels: >= 0.8).
           stepping_stones: stones per lane (5 x 5) and no level (z = 0) stone at the centre.
Writes terrain_audit.json and one PNG per terrain x level (top-down height map of lane 0 and
the lane-0 side profiles) into terrain_audit/.

Run (repo root):
  .venv/bin/python scripts/eval/mode_switch/tools/terrain_audit.py [--rough ...] [--jobs 5]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "logs/lipm_eval/mode_switch_v2/terrain_audit"
sys.path.insert(0, str(ROOT / "scripts/eval"))
Z_TOP = 3.0
LEVELS = 4
OFFSETS = (-0.3, 0.0, 0.3)  # Probe lines relative to the lane centre (spawn y range).
TOL = 0.002


class Prober:
    """Surface height under vertical CPU mj_ray probes (NaN = miss)."""

    def __init__(self, model):
        import mujoco

        self.mujoco, self.model = mujoco, model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        self.gid = np.zeros(1, np.int32)
        self.vec = np.array([0.0, 0.0, -1.0])

    def __call__(self, xs, ys) -> np.ndarray:
        xs, ys = np.broadcast_arrays(np.asarray(xs, float), np.asarray(ys, float))
        out = np.empty(xs.shape)
        pnt = np.array([0.0, 0.0, Z_TOP])
        for idx in np.ndindex(xs.shape):
            pnt[0], pnt[1] = xs[idx], ys[idx]
            dist = self.mujoco.mj_ray(self.model, self.data, pnt, self.vec, None, 1, -1, self.gid)
            out[idx] = Z_TOP - dist if dist >= 0 else np.nan
        return out


def q(values, ps=(0.0, 0.01, 0.5, 0.99, 1.0)) -> list:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return [round(float(np.quantile(values, p)), 4) for p in ps] if values.size else []


def plateaus(xs, zs, min_len=0.05):
    """(height, x_start, x_end) of flat runs (|dz| < 1 mm) at least min_len long."""
    runs, start = [], 0
    for i in range(1, len(zs) + 1):
        if i == len(zs) or not abs(zs[i] - zs[start]) < 1e-3:
            if xs[i - 1] - xs[start] >= min_len - 1e-9:
                runs.append((float(np.median(zs[start:i])), float(xs[start]), float(xs[i - 1])))
            start = i
    return runs


def build_course(rough: str, seed: int):
    import mujoco
    import mjlab.tasks  # noqa: F401
    import wheeled_legged_mjlab  # noqa: F401
    import lipm_eval as L
    import mode_switch_eval as M
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.terrains import TerrainGenerator

    task = L._checkpoint("Ours").task_id
    train_gen = load_env_cfg(task).scene.terrain.terrain_generator
    for spec in L.CHECKPOINTS[1:]:
        if load_env_cfg(spec.task_id).scene.terrain != load_env_cfg(task).scene.terrain:
            raise RuntimeError(f"{spec.task_id} trains on a different terrain configuration.")
    cfg = M.make_env_cfg(task, rough, "forward_wide", M.NUM_ENVS, seed, LEVELS)
    gen_cfg = cfg.scene.terrain.terrain_generator
    spec = mujoco.MjSpec()
    gen = TerrainGenerator(gen_cfg, device="cpu")
    gen.compile(spec)
    section, z_shift = M.rough_section(train_gen, rough)
    return spec.compile(), gen, train_gen, section, z_shift


def training_patch(train_gen, rough: str, d: float, seed: int):
    """One training patch (training cfg unchanged) at difficulty d, centred at the origin."""
    import copy

    import mujoco
    from mjlab.terrains import TerrainGenerator, TerrainGeneratorCfg

    sub = copy.deepcopy(train_gen.sub_terrains[rough])
    cfg = TerrainGeneratorCfg(seed=seed, size=train_gen.size, num_rows=1, num_cols=1,
                              sub_terrains={rough: sub}, difficulty_range=(d, d))
    spec = mujoco.MjSpec()
    TerrainGenerator(cfg, device="cpu").compile(spec)
    return spec.compile()


def course_geoms(model, gen, row: int, lane: int):
    """Box geoms (half sizes, world pos) whose centre lies in the rough section of (row, lane)."""
    import mode_switch_eval as M

    x0, y0 = gen._get_sub_terrain_position(row, lane)[:2]
    pos = model.geom_pos
    sel = (
        (model.geom_type == 6) & (pos[:, 0] > x0 + M.FLAT_IN) & (pos[:, 0] < x0 + M.FLAT_IN + M.ROUGH_LEN)
        & (pos[:, 1] > y0) & (pos[:, 1] < y0 + M.WIDTH)
    )
    return model.geom_size[sel], pos[sel]


def hfield_vertex_heights(model, gen, row: int) -> np.ndarray:
    """Physical vertex heights of all heightfields in terrain row ``row`` (all lanes)."""
    import mode_switch_eval as M

    x0 = gen._get_sub_terrain_position(row, 0)[0]
    out = []
    for g in np.flatnonzero(model.geom_type == 1):  # mjGEOM_HFIELD
        if not x0 < model.geom_pos[g, 0] < x0 + M.COURSE_END:
            continue
        h = model.geom_dataid[g]
        start, count = model.hfield_adr[h], model.hfield_nrow[h] * model.hfield_ncol[h]
        out.append(model.geom_pos[g, 2] + model.hfield_data[start:start + count] * model.hfield_size[h, 2])
    return np.concatenate(out)


def feature_check(rough, d, section, z_shift, xs, lines, model, gen, row, train_gen, prober) -> dict:
    """Key feature measured on the course vs the training formula at difficulty d."""
    import mode_switch_eval as M

    sec = (xs >= M.FLAT_IN) & (xs <= M.FLAT_IN + M.ROUGH_LEN)
    centre = lines[:, 1][:, sec]  # [lanes, x] on the lane centre lines
    xsec = xs[sec]
    res: dict = {}
    if rough in ("pyramid_stair", "pyramid_stair_inv"):
        lo, hi = section.step_height_range
        h = lo + d * (hi - lo)
        n = int((M.ROUGH_LEN - 2 * section.border_width - section.platform_width) / (2 * section.step_width))
        sign = -1 if rough.endswith("inv") else 1
        levels = [sorted({round(p[0], 4) for p in plateaus(xsec, z)}, key=abs) for z in centre]
        rises = np.concatenate([np.diff([0.0, *lv]) for lv in levels])
        tops = [[p for p in plateaus(xsec, z) if abs(p[0] - sign * (n + 1) * h) < TOL] for z in centre]
        res = {
            "formula": f"step height = {lo} + d*({hi}-{lo}) = {h:.4f} m; {n} steps + platform, platform "
                       f"at {sign * (n + 1) * h:+.4f} m",
            "measured_step_heights_min_max": [float(np.abs(rises).min()), float(np.abs(rises).max())],
            "measured_levels_lane0": levels[0],
            "platform_length_on_centre_line_m": [round(t[0][2] - t[0][1] + 0.01, 3) for t in tops[:1] if t],
            "pass": bool(np.all(np.abs(np.abs(rises) - h) < TOL) and all(len(lv) == n + 1 for lv in levels)),
        }
    elif rough == "random_stairs":
        f = 0.5 + 0.5 * d
        lo, hi = section.step_height_range[0] * f, section.step_height_range[1] * f
        n = int((M.ROUGH_LEN - 2 * section.border_width - section.platform_width) / (2 * section.step_width))
        levels = [sorted({round(p[0], 4) for p in plateaus(xsec, z)}) for z in centre]
        rises = np.concatenate([np.diff([0.0, *lv]) for lv in levels])
        top = [p for p in plateaus(xsec, centre[0]) if abs(p[0] - levels[0][-1]) < TOL]
        res = {
            "formula": f"rise ~ U({section.step_height_range[0]}, {section.step_height_range[1]}) x (0.5+0.5d) "
                       f"= U({lo:.4f}, {hi:.4f}) m; {n} rings of {section.step_width} m + platform box "
                       f"{M.ROUGH_LEN - 2 * n * section.step_width:.2f} m at the height of the last ring "
                       f"(generator code), i.e. {n} rises and a top landing of "
                       f"{M.ROUGH_LEN - 2 * (n - 1) * section.step_width:.2f} m",
            "measured_rise_min_max": [float(rises.min()), float(rises.max())],
            "measured_top_height_min_max": [min(lv[-1] for lv in levels), max(lv[-1] for lv in levels)],
            "top_landing_length_on_centre_line_m": round(top[0][2] - top[0][1] + 0.01, 3) if top else None,
            "pass": bool(rises.min() >= lo - TOL and rises.max() <= hi + TOL and all(len(lv) == n for lv in levels)),
        }
    elif rough.startswith("hf_pyramid_slope"):
        slope = section.slope_range[0] + d * (section.slope_range[1] - section.slope_range[0])
        slopes, apex, plat = [], [], []
        for z in centre:
            peak = z[np.nanargmax(np.abs(z))]
            ramp = (xsec < M.FLAT_IN + M.ROUGH_LEN / 2) & (np.abs(z) < 0.9 * abs(peak)) & (xsec > M.FLAT_IN + 0.02)
            slopes.append(abs(np.polyfit(xsec[ramp], z[ramp], 1)[0]))
            apex.append(peak)
            top = np.abs(z - peak) < 1e-3
            plat.append(float(xsec[top].max() - xsec[top].min()))
        res = {
            "formula": f"slope = {section.slope_range[0]} + d*({section.slope_range[1]}-{section.slope_range[0]}) = {slope:.4f}",
            "measured_slope_on_centre_line_min_max": [round(min(slopes), 4), round(max(slopes), 4)],
            "measured_apex_m": round(float(np.mean(apex)), 4),
            "platform_length_on_centre_line_m": round(float(np.mean(plat)), 3),
            "pass": bool(abs(np.mean(slopes) - slope) < 0.01),
        }
    elif rough == "discrete_obstacles":
        lo, hi = section.obstacle_height_range
        units = int((lo + d * (hi - lo)) / section.vertical_scale)
        # Generator: rng.choice([-h, -h // 2, h // 2, h]) in vertical units (floor division).
        allowed = np.array([-units, -units // 2, 0, units // 2, units]) * section.vertical_scale
        values = centre[np.isfinite(centre)]
        counts = {round(float(a), 3): float(np.mean(np.abs(values - a) < 1e-3)) for a in allowed}
        vertices = hfield_vertex_heights(model, gen, row)
        off = np.min(np.abs(vertices[:, None] - allowed[None]), 1) > 1e-4
        res = {
            "formula": f"obstacle heights in {{-h, -h//2, 0, h//2, h}} (units), h = int(({lo} + d*({hi}-{lo}))/"
                       f"{section.vertical_scale}) = {units} -> {units * section.vertical_scale:.3f} m "
                       f"(unquantized {lo + d * (hi - lo):.4f} m)",
            "fraction_of_centre_line_at_each_level": counts,
            "heightfield_vertices_not_at_an_allowed_level": f"{int(off.sum())} of {len(vertices)}",
            "measured_min_max": [float(values.min()), float(values.max())],
            "num_obstacles_per_lane": section.num_obstacles,
            "pass": bool(values.min() >= allowed[0] - 1e-3 and values.max() <= allowed[-1] + 1e-3 and not off.any()),
        }
    elif rough == "random_rough":
        lo, hi = section.noise_range
        inner = (xsec > M.FLAT_IN + 0.16) & (xsec < M.FLAT_IN + M.ROUGH_LEN - 0.16)
        values = centre[:, inner]
        allowed = np.array([0.0, *np.arange(lo, hi + 1e-9, section.noise_step)])  # 0: border ring
        vertices = hfield_vertex_heights(model, gen, row)
        off = np.min(np.abs(vertices[:, None] - allowed[None]), 1) > 1e-4
        res = {
            "formula": f"vertex heights in {{{lo}, ..., {hi}}} step {section.noise_step} m (border ring 0), "
                       f"scale_with_difficulty={section.scale_with_difficulty} (difficulty ignored)",
            "measured_quantiles_0_1_50_99_100": q(values),
            "heightfield_vertices_not_at_an_allowed_level": f"{int(off.sum())} of {len(vertices)}",
            "vertex_height_histogram": {f"{a:.2f}": round(float(np.mean(np.abs(vertices - a) < 1e-4)), 4) for a in allowed},
            "pass": bool(np.nanmin(values) >= lo - TOL and np.nanmax(values) <= hi + TOL and not off.any()),
        }
    elif rough == "random_spread":
        f = 0.2 + 0.8 * d
        sampled = int(section.num_boxes * (0.5 + 0.5 * d))
        counts, heights = [], []
        for lane in range(M.LANES):
            size, _ = course_geoms(model, gen, row, lane)
            boxes = size[size[:, 0] < 1.0]  # Exclude the 3.6 m section floor.
            counts.append(len(boxes))
            heights.extend(2 * boxes[:, 2])
        # Robots' path band (lane centre +- 0.3 m spawn +- 0.156 m wheel), 4 cm x 1 cm samples.
        bx, by = np.arange(4.205, 7.4, 0.01), np.arange(-0.46, 0.4601, 0.04)
        raised = np.concatenate([
            prober(x0 + bx[:, None], y0 + M.WIDTH / 2 + by[None, :]) > 0.005
            for x0, y0 in (gen._get_sub_terrain_position(row, lane)[:2] for lane in range(M.LANES))
        ], axis=1)  # [x, lanes x lines]
        pocket, rest = np.abs(bx - 5.8) <= 0.2, ((bx >= 4.4) & (bx <= 5.4)) | ((bx >= 6.2) & (bx <= 7.2))
        res = {
            "formula": f"{section.num_boxes} boxes x (0.5+0.5d) = {sampled} sampled (no centre skip), "
                       f"height U({section.box_height_range[0]}, {section.box_height_range[1]}) x (0.2+0.8d) = "
                       f"U({section.box_height_range[0] * f:.4f}, {section.box_height_range[1] * f:.4f}) m",
            "boxes_per_lane_min_max": [min(counts), max(counts)],
            "box_height_min_max": [round(min(heights), 4), round(max(heights), 4)],
            "measured_section_max_height": float(np.nanmax(centre)),
            "mean_boxes_per_lane": round(float(np.mean(counts)), 2),
            "path_band_raised_fraction": round(float(raised.mean()), 4),
            "centre_pocket_raised_fraction": round(float(raised[pocket].mean()), 4),
            "rest_of_band_raised_fraction": round(float(raised[rest].mean()), 4),
            "raised_samples_pocket_rest": [int(raised[pocket].sum()), int(raised[pocket].size), int(raised[rest].sum()), int(raised[rest].size)],
            "pass": bool(min(counts) == max(counts) == sampled
                         and min(heights) >= section.box_height_range[0] * f - 1e-6
                         and max(heights) <= section.box_height_range[1] * f + 1e-6),
        }
    elif rough == "stepping_stones":
        s_min, s_max = section.stone_size_range
        avg = s_max - d * (s_max - s_min)
        inner = M.ROUGH_LEN - 2 * section.border_width
        num = max(2, int(np.floor(inner / (s_max + section.stone_distance_range[0]))) + 1)
        pitch = inner / (num - 1)
        t_inner = train_gen.size[0] - 2 * train_gen.sub_terrains[rough].border_width
        t_num = max(2, int(np.floor(t_inner / (s_max + section.stone_distance_range[0]))) + 1)
        gaps, tops = [], []
        for z in centre:
            pit = np.concatenate([[False], z < -0.3, [False]]).astype(np.int8)
            edges = np.diff(pit)
            gaps.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * 0.01)
            tops.extend(z[z > -0.3])
        var = section.stone_height_variation * d
        stones, level_centre = [], 0
        for lane in range(M.LANES):
            size, pos = course_geoms(model, gen, row, lane)
            stone = np.max(size[:, :2], axis=1) < 0.5  # Not the floor / border boxes.
            stones.append(int(stone.sum()))
            x0, y0 = gen._get_sub_terrain_position(row, lane)[:2]
            k = np.argmin(np.hypot(pos[stone, 0] - x0 - 5.8, pos[stone, 1] - y0 - M.WIDTH / 2))
            level_centre += int(abs(pos[stone][k, 2] + size[stone][k, 2]) < 1e-9)  # Top exactly at 0.
        res = {
            "formula": f"pitch = inner/(stones-1): training {t_inner:.3f}/{t_num - 1} = {t_inner / (t_num - 1):.4f} m, "
                       f"course {inner:.4f}/{num - 1} = {pitch:.4f} m; stone {s_max} - d*({s_max}-{s_min}) = {avg:.4f} m "
                       f"+- {section.stone_size_variation * d:.4f}; nominal gap {max(0.0, pitch - avg):.4f} m; "
                       f"top height +- {var:.4f} m",
            "measured_gap_on_centre_lines_quantiles": q(gaps) if gaps else "no gaps",
            "measured_top_height_min_max": [float(np.min(tops)), float(np.max(tops))],
            "stones_per_lane_min_max": [min(stones), max(stones)],
            "lanes_with_level_centre_stone": level_centre,
            "pass": bool(abs(pitch - t_inner / (t_num - 1)) < 1e-9 and np.min(tops) >= -var - TOL
                         and min(stones) == max(stones) == num * num and level_centre == 0
                         and np.max(tops) <= var + TOL
                         and (not gaps or abs(np.median(gaps) - max(0.0, pitch - avg)) < 0.06)),
        }
    elif rough == "tilted_grid":
        tilt = math.radians(section.tilt_range_deg * d)
        bound = section.height_range * d / 2 + tilt * section.grid_width
        values = centre[np.isfinite(centre)]
        res = {
            "formula": f"tile top = 0.2 + U(+-{section.height_range * d / 2:.4f}) + slope_x dx + slope_y dy, "
                       f"|slope| <= {tilt:.4f} (tilt {section.tilt_range_deg * d:.2f} deg used as slope), "
                       f"tile {section.grid_width} m; corner bound |z - 0.2| <= {bound:.4f} m; z shift {z_shift}",
            "measured_min_max": [float(values.min()), float(values.max())],
            "pass": bool(np.abs(values).max() <= bound + TOL and np.abs(values).max() > 0.5 * bound),
        }
    return res


def distribution_vs_training(prober, gen, row, rough, d, section, train_gen, z_shift, seed) -> dict | None:
    """Height distribution of the course section (lanes 0-3) vs one training patch at d, both
    sampled 0.2 m inside their border (training: also outside the central platform)."""
    import mode_switch_eval as M

    if rough not in ("discrete_obstacles", "random_rough", "random_spread", "stepping_stones", "tilted_grid"):
        return None
    step = 0.02
    inset = getattr(section, "border_width", 0.0) + 0.2
    course = []
    for lane in range(4):
        x0, y0 = gen._get_sub_terrain_position(row, lane)[:2]
        xs = x0 + M.FLAT_IN + np.arange(inset, M.ROUGH_LEN - inset, step)
        ys = y0 + np.arange(inset, M.WIDTH - inset, step)
        course.append(prober(xs[:, None], ys[None, :]).ravel())
    course = np.concatenate(course)
    sub = train_gen.sub_terrains[rough]
    model = training_patch(train_gen, rough, d, seed)
    half = train_gen.size[0] / 2 - getattr(sub, "border_width", 0.0) - 0.2
    grid = np.arange(-half, half, step)
    train = Prober(model)(grid[:, None], grid[None, :])
    platform = getattr(sub, "platform_width", 0.0) / 2 + 0.3
    keep = ~((np.abs(grid)[:, None] < platform) & (np.abs(grid)[None, :] < platform))
    train = train[keep] + z_shift  # Training border level -> floor level.

    def stats(z):
        z = z[np.isfinite(z)]
        return {
            "quantiles_1_10_50_90_99": q(z, (0.01, 0.1, 0.5, 0.9, 0.99)),
            "raised_fraction_gt_1cm": round(float(np.mean(z > 0.01)), 4),
            "sunk_fraction_lt_-1cm": round(float(np.mean(z < -0.01)), 4),
            "pit_fraction_lt_-0.3m": round(float(np.mean(z < -0.3)), 4),
            "misses": int(np.sum(~np.isfinite(z))),
        }

    return {"course_section": stats(course), "training_patch_excl_platform": stats(train)}


def expected_seam(rough, d, section, z_shift):
    """(description, predicate on steps entering / leaving) of the training border joint."""
    if rough in ("pyramid_stair", "pyramid_stair_inv"):
        lo, hi = section.step_height_range
        h = (lo + d * (hi - lo)) * (-1 if rough.endswith("inv") else 1)
        return (f"border -> first step: {h:+.4f} m in, {-h:+.4f} m out",
                lambda s_in, s_out: np.all(np.abs(s_in - h) < TOL) and np.all(np.abs(s_out + h) < TOL))
    if rough == "random_stairs":
        f = 0.5 + 0.5 * d
        lo, hi = section.step_height_range[0] * f, section.step_height_range[1] * f
        return (f"border -> first ring: +U({lo:.4f}, {hi:.4f}) in, the same down out",
                lambda s_in, s_out: np.all((s_in > lo - TOL) & (s_in < hi + TOL)) and np.all((-s_out > lo - TOL) & (-s_out < hi + TOL)))
    if rough == "tilted_grid":
        bound = section.height_range * d / 2 + math.radians(section.tilt_range_deg * d) * section.grid_width
        return (f"border (tile base level) -> tile edge: |step| <= {bound:.4f} m",
                lambda s_in, s_out: np.all(np.abs(s_in) <= bound + TOL) and np.all(np.abs(s_out) <= bound + TOL))
    if rough == "random_spread":
        top = section.box_height_range[1] * (0.2 + 0.8 * d)
        return (f"0 (floor), or a box standing across the seam (<= {top:.4f} m), as on the training border",
                lambda s_in, s_out: np.all(np.abs(s_in) <= top + TOL) and np.all(np.abs(s_out) <= top + TOL))
    if rough.startswith("hf_pyramid_slope"):
        return ("0 (zero-height pyramid edge; ramp starts at the seam)",
                lambda s_in, s_out: np.all(np.abs(s_in) < 0.005) and np.all(np.abs(s_out) < 0.005))
    return ("0 (flat border ring / flat border at floor level)",
            lambda s_in, s_out: np.all(np.abs(s_in) < TOL) and np.all(np.abs(s_out) < TOL))


def seam_jump(prober, x, ys, eps=0.0005):
    """Height discontinuity at x: z(x+e) - z(x-e) minus the linear-ramp part on both sides."""
    z = {k: prober(x + k * eps, ys) for k in (-3, -1, 1, 3)}
    return z[1] - z[-1] - ((z[3] - z[1]) + (z[-1] - z[-3])) / 2


def narrow_dips(prober, xs, y, z: np.ndarray, half: int = 15, depth: float = 0.01) -> tuple[int, int]:
    """(real, ray-artefact) samples lower than both neighbours 1.5 cm away by > 1 cm. A dip that
    vanishes when the probe moves 0.5 mm in x or y is a CPU mj_ray artefact (rays exactly on a
    heightfield triangle edge), not a hole."""
    real = artefact = 0
    for i in range(half, len(z) - half):
        ref = min(z[i - half], z[i + half]) - depth
        if not z[i] < ref:
            continue
        again = prober(np.array([xs[i] - 5e-4, xs[i] + 5e-4, xs[i], xs[i]]),
                       np.array([y, y, y - 5e-4, y + 5e-4]))
        if np.all(again >= ref):
            artefact += 1
        else:
            real += 1
    return real, artefact


def save_png(path, rough, d, xs_map, ys_map, hmap, xs, lines0, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    import mode_switch_eval as M

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    finite = hmap[np.isfinite(hmap)]
    vmin, vmax = min(-0.02, max(float(finite.min()), -0.35)), max(0.02, float(finite.max()))
    image = axes[0].imshow(hmap.T, origin="lower", extent=(xs_map[0], xs_map[-1], ys_map[0], ys_map[-1]),
                           cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
                           aspect="auto", interpolation="nearest")
    for dy in OFFSETS:
        axes[0].axhline(M.WIDTH / 2 + dy, color="#52514e", lw=0.6, ls=":")
    fig.colorbar(image, ax=axes[0], label="height [m] (floor = 0, pits < -0.35 saturate)", pad=0.01)
    axes[0].set_ylabel("y in lane [m]")
    axes[0].set_title(f"{rough}, d = {d:g}: top-down height map of lane 0 (CPU mj_ray, 2 cm; y axis stretched; "
                      f"dotted: lane centre and +-0.3 m)", fontsize=10, loc="left")
    colors = ("#2a78d6", "#eb6834", "#1baf7a")
    for k, dy in enumerate(OFFSETS):
        axes[1].plot(xs, lines0[k], lw=1.2 if dy == 0 else 0.9, color=colors[k],
                     label=f"y = centre {dy:+.1f} m")
    for x in (M.FLAT_IN, M.FLAT_IN + M.ROUGH_LEN):
        for ax in axes:
            ax.axvline(x, color="#0b0b0b", lw=0.8, ls="--")
    axes[1].set_ylim(max(np.nanmin(lines0) - 0.05, -0.9), np.nanmax(lines0) + 0.08)
    axes[1].set_xlabel("course x [m] (seams at 4.0 and 7.6 m)")
    axes[1].set_ylabel("height [m]")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].grid(color="#e0e0e0", lw=0.5)
    axes[1].set_title(summary, fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def audit_terrain(rough: str, seed: int) -> dict:
    import lipm_eval as L
    import mode_switch_eval as M

    started = time.time()
    model, gen, train_gen, section, z_shift = build_course(rough, seed)
    prober = Prober(model)
    xs = np.arange(0.005, M.COURSE_END, 0.01)
    ys = np.arange(0.005, M.WIDTH, 0.01)
    flat_all = (xs < M.FLAT_IN) | (xs > M.FLAT_IN + M.ROUGH_LEN)
    flat = (xs <= 3.7) | (xs >= 7.9)  # Flat segments without the 0.3 m aprons.
    apron = flat_all & ~flat
    judged = np.zeros(xs.shape, bool)
    for k in ("flat_in", "flat_out"):
        judged |= (xs >= M.WINDOWS[k][0]) & (xs < M.WINDOWS[k][1])
    sec = (xs >= M.FLAT_IN) & (xs <= M.FLAT_IN + M.ROUGH_LEN)
    embedding = {
        k: getattr(section, k) for k in (
            "border_width", "platform_width", "horizontal_scale", "num_obstacles", "num_boxes",
        ) if hasattr(section, k)
    }
    report = {
        "generator_class": type(section).__name__, "embedding": {**embedding, "z_shift": z_shift},
        "terrain_fingerprint_4x16": L.terrain_fingerprint(model), "num_geoms": int(model.ngeom), "levels": {},
    }
    for row, d in enumerate(M.level_difficulties(LEVELS)):
        lines = np.empty((M.LANES, len(OFFSETS), len(xs)))
        cross, seam_in, seam_out, fine_min, dips, artefacts, misses = [], [], [], [], 0, 0, 0
        fine = {x: np.arange(x - 0.05, x + 0.0505, 0.001) for x in (M.FLAT_IN, M.FLAT_IN + M.ROUGH_LEN)}
        for lane in range(M.LANES):
            x0, y0 = gen._get_sub_terrain_position(row, lane)[:2]
            for k, dy in enumerate(OFFSETS):
                lines[lane, k] = prober(x0 + xs, y0 + M.WIDTH / 2 + dy)
                for xc, fx in fine.items():
                    z = prober(x0 + fx, y0 + M.WIDTH / 2 + dy)
                    misses += int(np.sum(~np.isfinite(z)))
                    real, fake = narrow_dips(prober, x0 + fx, y0 + M.WIDTH / 2 + dy, z)
                    dips, artefacts = dips + real, artefacts + fake
                    fine_min.append(np.nanmin(z))
            cross.append(prober(x0 + np.array([0.5, 2.0, 3.5, 9.0, 11.0])[:, None], y0 + ys[None, :]))
            seam_in.append(seam_jump(prober, x0 + M.FLAT_IN, y0 + ys))
            seam_out.append(seam_jump(prober, x0 + M.FLAT_IN + M.ROUGH_LEN, y0 + ys))
        cross, seam_in, seam_out = np.array(cross), np.concatenate(seam_in), np.concatenate(seam_out)
        misses += int(np.sum(~np.isfinite(lines[:, :, ~sec]))) + int(np.sum(~np.isfinite(cross)))
        desc, seam_ok = expected_seam(rough, d, section, z_shift)
        feature = feature_check(rough, d, section, z_shift, xs, lines, model, gen, row, train_gen, prober)
        # Top-down map of lane 0 (2 cm) for the figure.
        x0, y0 = gen._get_sub_terrain_position(row, 0)[:2]
        # 0.7 mm offset: no ray exactly on a shared box edge / heightfield grid line (mj_ray misses those).
        xs_map, ys_map = np.arange(0.0107, M.COURSE_END, 0.02), np.arange(0.0107, M.WIDTH, 0.02)
        hmap = prober(x0 + xs_map[:, None], y0 + ys_map[None, :])
        map_all = (xs_map < M.FLAT_IN) | (xs_map > M.FLAT_IN + M.ROUGH_LEN)
        map_judged = np.zeros(xs_map.shape, bool)
        for k in ("flat_in", "flat_out"):
            map_judged |= (xs_map >= M.WINDOWS[k][0]) & (xs_map < M.WINDOWS[k][1])
        cross_judged = cross[:, [1, 3, 4]]  # x = 2, 9, 11 m
        flat_max_all = float(max(np.nanmax(np.abs(lines[:, :, flat_all])), np.nanmax(np.abs(cross)),
                                 np.nanmax(np.abs(hmap[map_all]))))
        flat_max_judged = float(max(np.nanmax(np.abs(lines[:, :, judged])), np.nanmax(np.abs(cross_judged)),
                                    np.nanmax(np.abs(hmap[map_judged]))))
        flat_max = float(max(np.nanmax(np.abs(lines[:, :, flat])), np.nanmax(np.abs(cross)),
                             np.nanmax(np.abs(hmap[(xs_map <= 3.7) | (xs_map >= 7.9)]))))
        # Overhang onto the floor next to the seams: height and distance from the seam.
        up = np.abs(lines) > 1e-6
        reach = [M.FLAT_IN - xs[k] for k in np.flatnonzero((xs < M.FLAT_IN) & up.any((0, 1)))]
        reach += [xs[k] - M.FLAT_IN - M.ROUGH_LEN for k in np.flatnonzero((xs > M.FLAT_IN + M.ROUGH_LEN) & up.any((0, 1)))]
        entry = {
            "difficulty": d,
            "flat_max_abs_z_all_m": flat_max_all,
            "flat_max_abs_z_judged_windows_m": flat_max_judged,
            "flat_max_abs_z_excl_aprons_m": flat_max,
            "apron_max_z_m": float(np.nanmax(lines[:, :, apron])),
            "apron_overhang_max_reach_m": round(float(max(reach)), 3) if reach else 0.0,
            "seam_in_jump_quantiles_0_1_50_99_100": q(seam_in),
            "seam_out_jump_quantiles_0_1_50_99_100": q(seam_out),
            "seam_expected_training_joint": desc,
            "seam_ok": bool(seam_ok(seam_in, seam_out)),
            "seam_band_min_z_m": float(np.min(fine_min)),
            "seam_band_ray_misses": misses,
            "seam_band_narrow_dips": dips,
            "seam_band_ray_artefacts_discarded": artefacts,
            "section_min_max_z_m": [float(np.nanmin(lines[:, :, sec])), float(np.nanmax(lines[:, :, sec]))],
            "section_map_lane0_min_max_z_m": [
                float(np.nanmin(hmap[(xs_map >= 4.0) & (xs_map <= 7.6)])),
                float(np.nanmax(hmap[(xs_map >= 4.0) & (xs_map <= 7.6)])),
            ],
            "feature": feature,
            "distribution_vs_training": distribution_vs_training(prober, gen, row, rough, d, section, train_gen, z_shift, seed),
        }
        # Flat everywhere; random_spread: overhanging boxes allowed in the aprons (as on the training border).
        overhang_ok = rough == "random_spread" and flat_max < 1e-6 and entry["apron_max_z_m"] <= (
            section.box_height_range[1] * (0.2 + 0.8 * d) + TOL)
        entry["pass"] = bool(
            (flat_max_all < 1e-6 or overhang_ok) and flat_max_judged < 1e-6 and entry["seam_ok"] and misses == 0
            and dips == 0 and feature.get("pass", False)
        )
        report["levels"][f"{d:g}"] = entry
        summary = (f"flat max|z| all {flat_max_all:.3f} / judged {flat_max_judged:.1e} m | seam in {q(seam_in)[0]:+.3f}..{q(seam_in)[-1]:+.3f}, out "
                   f"{q(seam_out)[0]:+.3f}..{q(seam_out)[-1]:+.3f} m | section {entry['section_min_max_z_m'][0]:+.3f}.."
                   f"{entry['section_min_max_z_m'][1]:+.3f} m | holes {misses + dips} | "
                   f"{'PASS' if entry['pass'] else 'FAIL'}")
        save_png(OUT / f"{rough}__d{d:.2f}.png", rough, d, xs_map, ys_map, hmap, xs, lines[0], summary)
        print(f"[{rough} d={d:g}] {summary}", flush=True)
    if rough == "random_spread":
        # Centre pocket (generator centre skip) pooled over the 64 lane instances of the 4 levels.
        counts = np.sum([e["feature"]["raised_samples_pocket_rest"] for e in report["levels"].values()], axis=0)
        ratio = (counts[0] / counts[1]) / (counts[2] / counts[3])
        report["centre_pocket_check"] = {
            "raised_fraction_pocket_over_rest": round(float(ratio), 3), "pass": bool(ratio >= 0.8),
            "note": "|x - 5.8| <= 0.2 m vs x in [4.4, 5.4] u [6.2, 7.2], path band, all levels; "
                    "Monte Carlo: about 1.0 without the centre skip, 0.6 with it",
        }
        print(f"[{rough}] centre pocket raised fraction / rest = {ratio:.3f}", flush=True)
        for e in report["levels"].values():
            e["pass"] = bool(e["pass"] and ratio >= 0.8)
    formulas = {json.dumps(L.formula_parameters(train_gen.sub_terrains[rough], d), sort_keys=True)
                for d in M.level_difficulties(LEVELS)}
    report["levels_are_replicates_by_formula"] = len(formulas) == 1
    report["audit_seconds"] = round(time.time() - started, 1)
    return report


def main():
    import mode_switch_eval as M
    import lipm_eval as L

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rough", nargs="+", default=list(M.ROUGH_TYPES))
    parser.add_argument("--seed", type=int, default=L.EVAL_SEED)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs, mp_context=get_context("spawn")) as pool:
        reports = dict(zip(args.rough, pool.map(audit_terrain, args.rough, [args.seed] * len(args.rough))))
    path = OUT / "terrain_audit.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(reports)
    path.write_text(json.dumps(previous, indent=1, default=float) + "\n")
    for rough, rep in reports.items():
        verdict = {d: e["pass"] for d, e in rep["levels"].items()}
        print(rough, verdict, "replicates" if rep["levels_are_replicates_by_formula"] else "")


if __name__ == "__main__":
    main()
