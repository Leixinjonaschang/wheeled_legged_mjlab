"""Flat -> rough -> flat mode-switching evaluation (wheeled rolling <-> legged stepping).

Course (one lane, along +x): 4.0 m flat, 3.6 m rough section, 4.0 m flat, 3.6 m wide.
(Rollouts saved before 2026-09-30 used a 4.4 m exit flat; see course_geometry().)
The rough section is one of the ten non-flat training sub-terrains (default: tilted_grid,
as in the paper's flat-rough-flat figure), embedded by rough_section():
  - deep copy of the training cfg, x in [4.0, 7.6] m over the full width, box/obstacle
    counts at the training density by area (random_spread: per area of the region the
    generator samples box centres from, 14 boxes; see rough_section());
  - border 0 and no central platform (random_spread / stepping_stones: platform_width < 0
    disables the generator's centre skip / centre platform stone, which would otherwise
    sit on every lane centre line); pyramid-like sections (pyramid_stair(_inv),
    hf_pyramid_slope(_inv), random_stairs) are crossed through their centre and keep a
    central landing (training: 2.0 m platform): pyramid_stair(_inv) 0.6 m platform,
    hf_pyramid_slope(_inv) 0.6 m platform (clipped flat top about 0.9 m on the centre
    line), random_stairs 5 rings with a 0.8 m top (the generator's top is the last ring;
    platform_width 0.6 would give 4 rings and a 1.5 m top). Landings are narrower than
    wheel track (0.31 m) + lateral spawn spread (+-0.3 m): robots crossing the apex more
    than about 0.14 m off the lane centre have the outer wheel one ring lower, as when
    walking a training pyramid off-centre; aggregate stores the lateral offset there
    (dy_centre) to stratify by it;
  - the section rim is flush with the floor (tilted_grid: z shift -0.2 m), and seams have
    no lips, gaps or holes that the training geometry does not have. Exceptions to border 0
    that keep the training joint: heightfield discrete_obstacles / random_rough keep their
    1-pixel flat border ring (border 0 leaves vertical walls where obstacles are cut by the
    seam); stepping_stones keeps the training stone pitch (7/9 m) and pads the 4-pitch
    stone field (3.11 m) with a flat border at floor level. Box slivers < 2 cm wide
    (zero-width border boxes, the 1 cm tilted_grid platform pin, the 1e-6 m
    stepping_stones platform) are dropped. The flat segments are at z = 0 everywhere except
    that yawed random_spread boxes overhang up to about 0.29 m onto the floor next to the
    seams (as they overhang the training border); the judged flat windows are unaffected.
  - discrete_obstacles: the generator truncates the obstacle height to 5 mm units, so the
    levels have obstacle heights 0.045 / 0.075 / 0.110 / 0.150 m (effective difficulty
    0.25 / 0.46 / 0.71 / 1.0); levels.json lists the geometry parameters of every level.
Default (--levels 0): one row of 16 lanes at the maximum training difficulty (1.0), each
lane an independent random instance. --levels N: N rows (curriculum mode) at difficulties
(k+1)/N x 1.0, k = 0..N-1, each row 16 independent lanes; env i runs on lane i % 16 of
level (i // 16) % N (1024 envs, N = 4: 16 robots per level and lane).
The rest of the setup is the frozen play-config protocol of lipm_eval.py (startup DR,
reset perturbations, failure criteria), except that commands are fixed per run:
  forward       heading 0,    heading-frame v_x ~ U(0.6, 1.0) m/s
  forward_wide  heading 0,    heading-frame v_x ~ U(0.5, 2.0) m/s
  lateral       heading pi/2, heading-frame v_y ~ -U(0.6, 1.0) m/s (moves along +x)
  diagonal      heading pi/4, v_x ~ U(0.45, 0.7), v_y ~ -U(0.45, 0.7) m/s (moves along +x)
Each trajectory runs up to --time-limit (default 25 s) and ends at the course end
(x >= 11.6 m) or at the first failure (fell_over, illegal_contact, non_finite_physics).
Rollouts also store per env the difficulty, level and lane, the reset state and command
(identity checks across checkpoints), the terrain / DR fingerprints, the base position
along (course_x) and across (course_y, 0..3.6 m, lane centre 1.8 m) the lane, and the
geometry parameters of every level (level_params).
Rollouts are NOT bit-reproducible run to run: terrain, DR, reset states and commands are
identical across runs and checkpoints, but the GPU physics differs from the first step
(nondeterministic MuJoCo Warp kernels), so repeating a run changes per-cell rates by a few
percentage points. Compare checkpoints with the confidence intervals of stats.json and
against the run-to-run noise floor (scripts/eval/mode_switch/tools/noise_floor.py on
a replicate run), not on point estimates.

Pre-specified switching criterion (forward commands):
  step event = a wheel off the ground for >= 0.06 s that reaches >= 3 cm clearance
  (clearance = wheel-bottom height above the highest terrain point in the 0.4 m x 0.4 m
  wheel height scan, i.e. a conservative clearance on rough ground);
  rolling on flat-in   : no step event starting while the base is at x in [1.8, 3.2] m;
  stepping on rough    : >= 1 step event of EACH wheel while the base is at x in [4.2, 7.4] m;
  rolling on flat-out  : no step event while the base is at x in [9.0, 11.6] m;
  switch success       : all three and the course is completed without failure.

Aggregation first checks that the checkpoints saw identical terrain, DR, initial states and
commands (identity_checks.json; on a mismatch it writes AGGREGATE_FAILED, removes stale
outputs and writes nothing else). It groups by (rough, command, ckpt) and, for rollouts
with per-env difficulty, additionally by difficulty ("rough|command|ckpt|d=0.25"; the key
without d pools the levels) and by commanded speed ("...|v=[0.5,1)", with and without d;
bins SPEED_BINS). time_ratio = completion time / ((course_end - x_start) / v_cmd) is the
speed-normalised time. It flags terrains whose training formula does not depend on
difficulty (levels are then replicates, e.g. random_rough). stats.json holds 95 %
lane-cluster bootstrap intervals for every group (clusters = lanes, resampled within each
difficulty level: the 16 robots of a lane share one terrain instance) and paired
Ours - baseline differences on the same robots with the same resampling.

Usage (repo root):
  .venv/bin/python scripts/eval/mode_switch_eval.py rollout --ckpt Ours --command forward --out DIR
  .venv/bin/python scripts/eval/mode_switch_eval.py rollout --ckpt Ours --rough stepping_stones \
    --command forward_wide --levels 4 --time-limit 40 --out DIR
  uv run --no-project --with numpy --with scipy python scripts/eval/mode_switch_eval.py aggregate --out DIR
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
FLAT_IN, ROUGH_LEN, FLAT_OUT, WIDTH = 4.0, 3.6, 4.0, 3.6
COURSE_END = FLAT_IN + ROUGH_LEN + FLAT_OUT
LANES = 16
NUM_ENVS = 1024
TIME_LIMIT = 25.0  # s (1250 policy steps)
WINDOWS = {"flat_in": (1.8, 3.2), "rough": (4.2, 7.4), "flat_out": (9.0, COURSE_END)}
# Rollouts without stored geometry predate the 4.0 m exit flat (4.4 m, course end 12 m).
LEGACY_COURSE_END = 12.0
LEGACY_WINDOWS = {**WINDOWS, "flat_out": (9.0, LEGACY_COURSE_END)}
STEP_MIN_DURATION = 3  # policy steps (0.06 s)
STEP_MIN_CLEARANCE = 0.03
WHEEL_RADIUS = 0.127
COMMANDS = {
    "forward": {"heading": 0.0, "lin_vel_x": (0.6, 1.0), "lin_vel_y": (0.0, 0.0)},
    "forward_wide": {"heading": 0.0, "lin_vel_x": (0.5, 2.0), "lin_vel_y": (0.0, 0.0)},
    "lateral": {"heading": math.pi / 2, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (-1.0, -0.6)},
    "diagonal": {"heading": math.pi / 4, "lin_vel_x": (0.45, 0.7), "lin_vel_y": (-0.7, -0.45)},
}
ROUGH_TYPES = (
    "tilted_grid", "random_spread", "discrete_obstacles", "random_rough", "hf_pyramid_slope",
    "hf_pyramid_slope_inv", "pyramid_stair", "pyramid_stair_inv", "random_stairs", "stepping_stones",
)
PYRAMID_PLATFORM = 0.6  # Central landing of pyramid-like sections (training: 2.0 m).
SLIVER = 0.02  # Box geoms narrower than this (x or y) are generator artefacts and dropped.
SPEED_BINS = (0.5, 1.0, 1.5, 2.0)  # Commanded-speed strata of the aggregate step (m/s).
BOOTSTRAP = 2000  # Lane-cluster bootstrap replicates (stats.json).
BASELINE = "Ours"  # Paired differences in stats.json: BASELINE - other checkpoint.
IDENTITY_KEYS = (
    "terrain_sha", "dr_sha", "difficulty", "lane", "init_root_state", "init_joint_pos",
    "init_command_b", "cmd_vel_h",
)


def course_cfg_class():
    from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
    import mujoco

    @dataclass(kw_only=True)
    class FlatRoughFlatCourseCfg(SubTerrainCfg):
        rough: SubTerrainCfg
        rough_z_shift: float = 0.0

        def function(self, difficulty, spec, rng):
            body = spec.body("terrain")
            geometries = []
            for x0, x1 in ((0.0, FLAT_IN), (FLAT_IN + ROUGH_LEN, self.size[0])):
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=((x1 - x0) / 2, self.size[1] / 2, 0.05),
                    pos=((x0 + x1) / 2, self.size[1] / 2, -0.05),
                )
                geometries.append(TerrainGeometry(geom=geom, color=(0.55, 0.55, 0.55, 1.0)))
            rough = copy.deepcopy(self.rough)
            rough.size = (ROUGH_LEN, self.size[1])
            output = rough.function(difficulty, spec, rng)
            for item in output.geometries:
                geom = item.geom
                if geom is not None and geom.type == mujoco.mjtGeom.mjGEOM_BOX and min(geom.size[:2]) < SLIVER / 2:
                    spec.delete(geom)
                    continue
                if geom is not None:
                    geom.pos = np.array(geom.pos) + np.array([FLAT_IN, 0.0, self.rough_z_shift])
                geometries.append(item)
            return TerrainOutput(origin=np.array([1.0, self.size[1] / 2, 0.0]), geometries=geometries)

    return FlatRoughFlatCourseCfg


def rough_section(train_gen, rough_type: str):
    """Training sub-terrain cfg resized to the rough section, and its z shift (see docstring)."""
    rough = copy.deepcopy(train_gen.sub_terrains[rough_type])
    patch = train_gen.size[0]
    z_shift = 0.0
    name = type(rough).__name__
    if name == "BoxTiltedGridTerrainCfg":
        rough.border_width, rough.platform_width = 0.0, 0.01  # Tiles cover the section.
        z_shift = -0.2  # Tile surface level (base height 0.2 m) -> flush with the flat floor.
    elif name == "BoxRandomSpreadTerrainCfg":
        # Training density: boxes per m^2 of the area the generator samples box centres from
        # (inner side minus the mean box side, which confines the centres of a 3.6 m section
        # much more than those of the 7 m training area): 64 x 3.15 x 3.05 / (6.55 x 6.45) -> 14.
        # Monte Carlo of the generator (tools/random_spread_density.py): raised area on the
        # robots' path band 0.96-1.05x a training patch outside its platform at d = 0.25..1;
        # the plain area rule (64 x 12.96 / 49 = 17 boxes) gives about 1.2x.
        inner = patch - 2 * rough.border_width
        w, l = sum(rough.box_width_range) / 2, sum(rough.box_length_range) / 2
        rough.num_boxes = int(rough.num_boxes * (ROUGH_LEN - w) * (WIDTH - l) / ((inner - w) * (inner - l)))
        # platform_width < 0 disables the centre skip (the generator drops every box whose
        # footprint contains the section centre: a box-free pocket on the lane centre line; a
        # box would need a side >= 1 m to be skipped). add_floor: no platform geom is built.
        rough.border_width, rough.platform_width = 0.0, -1.0
    elif name == "HfDiscreteObstaclesTerrainCfg":
        # Training border ring (1 pixel at z = 0) kept; density per inner heightfield pixel.
        def inner_pixels(size):
            return int(size / rough.horizontal_scale) - 2 * int(rough.border_width / rough.horizontal_scale)

        rough.num_obstacles = round(
            rough.num_obstacles * inner_pixels(ROUGH_LEN) * inner_pixels(WIDTH) / inner_pixels(patch) ** 2
        )
        rough.platform_width = 0.0
    elif name == "HfRandomUniformTerrainCfg":
        pass  # Training border ring kept: noise 2-10 cm above the floor, as next to a training border.
    elif name == "HfPyramidSlopedTerrainCfg":
        rough.border_width, rough.platform_width = 0.0, PYRAMID_PLATFORM
        # 25 vertices over 3.6 m (0.15 m spacing): an odd count makes the pyramid symmetric with
        # zero-height edges at both seams (24 vertices leave a lip at x = 7.6 m).
        rough.horizontal_scale = ROUGH_LEN / 25.5
    elif name in ("BoxPyramidStairsTerrainCfg", "BoxInvertedPyramidStairsTerrainCfg"):
        rough.border_width, rough.platform_width = 0.0, PYRAMID_PLATFORM
    elif name == "BoxRandomStairsTerrainCfg":
        # The generator's top landing is the last ring (its platform box sits at that ring's
        # height), 3.6 - 2 (n - 1) 0.35 m for n = int((3.6 - platform) / 0.7) rings. platform 0
        # gives 5 rings and a 0.8 m top, the closest to the 0.6 m landing (0.6: 4 rings, 1.5 m).
        rough.border_width, rough.platform_width = 0.0, 0.0
    elif name == "BoxSteppingStonesTerrainCfg":
        # Training stone pitch (inner width / (stones - 1)); the stone field spans whole pitches,
        # the flat border (top at z = 0, as in training) pads it to the section.
        inner = patch - 2 * rough.border_width
        pitch = inner / math.floor(inner / (rough.stone_size_range[1] + rough.stone_distance_range[0]))
        rough.border_width = (ROUGH_LEN - pitch * math.floor(ROUGH_LEN / pitch)) / 2
        # No central platform: with platform_width 0 the grid-snapped platform still replaces
        # the central stone (x = 5.8 m, on the lane centre line) by a nominal, level, undisplaced
        # stone. platform_width -2 makes the snapped half-width negative: no stone is dropped
        # and the 1e-6 m platform box is removed as a sliver (5 x 5 stones per lane).
        rough.platform_width = -2.0
    else:
        raise ValueError(f"Unsupported rough section {name}")
    return rough, z_shift


def level_difficulties(levels: int) -> list[float]:
    """Difficulty of each terrain row: the maximum training difficulty, or (k+1)/N of it."""
    import lipm_eval as L

    if levels == 0:
        return [L.DIFFICULTY]
    lo, hi = L.DIFFICULTY / levels, L.DIFFICULTY
    # mjlab curriculum rows: lo + (hi - lo) * r / (rows - 1); must hit the levels exactly.
    rows = [lo + (hi - lo) * r / max(levels - 1, 1) for r in range(levels)]
    if rows != [L.DIFFICULTY * (k + 1) / levels for k in range(levels)]:
        raise RuntimeError(f"Curriculum rows {rows} miss the intended difficulties.")
    return rows


def spawn_layout(num_envs: int, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Terrain row (difficulty level) and lane of every environment (interleaved)."""
    env = np.arange(num_envs)
    return (env // LANES) % max(levels, 1), env % LANES


def make_env_cfg(task_id: str, rough_type: str, command: str, num_envs: int, seed: int, levels: int = 0):
    import lipm_eval as L
    from mjlab.terrains import TerrainGeneratorCfg

    cfg, train_cfg = L.make_eval_env_cfg(task_id, "flat", num_envs, seed)
    rough, z_shift = rough_section(train_cfg.scene.terrain.terrain_generator, rough_type)
    course = course_cfg_class()(proportion=1.0, rough=rough, rough_z_shift=z_shift)
    if levels:
        # Curriculum mode: one column per sub-terrain (lane), one row per difficulty level.
        difficulties = level_difficulties(levels)
        grid = dict(
            curriculum=True, num_rows=levels, difficulty_range=(difficulties[0], difficulties[-1]),
            sub_terrains={f"lane{j}": copy.deepcopy(course) for j in range(LANES)},
        )
    else:
        grid = dict(
            curriculum=False, num_rows=1, difficulty_range=(L.DIFFICULTY, L.DIFFICULTY),
            sub_terrains={"course": course},
        )
    gen = cfg.scene.terrain.terrain_generator
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed, size=(COURSE_END, WIDTH), border_width=gen.border_width,
        border_height=gen.border_height, num_cols=LANES, color_scheme=gen.color_scheme,
        add_lights=gen.add_lights, **grid,
    )
    spec = COMMANDS[command]
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = spec["lin_vel_x"]
    twist.ranges.lin_vel_y = spec["lin_vel_y"]
    twist.ranges.heading = (spec["heading"], spec["heading"])
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.rel_heading_envs = 1.0
    twist.resampling_time_range = (1.0e4, 1.0e4)
    reset = cfg.events["reset_base"].params["pose_range"]
    reset["x"], reset["y"] = (-0.3, 0.3), (-0.3, 0.3)
    reset["yaw"] = (spec["heading"] - 0.1, spec["heading"] + 0.1)
    return cfg


def rollout(args):
    import torch

    sys.path.insert(0, str(ROOT / "scripts/eval/lipm_diagnostics"))
    import lipm_eval as L
    from behavior_record import build, rollout as record
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False, deterministic=True)
    spec = L._checkpoint(args.ckpt)
    cfg = make_env_cfg(spec.task_id, args.rough, args.command, args.num_envs, args.seed, args.levels)
    rows, lanes = spawn_layout(args.num_envs, args.levels)

    def spawn(env):
        t = env.scene.terrain
        r, c = (torch.as_tensor(a, device=env.device) for a in (rows, lanes))
        t.terrain_levels[:] = r
        t.terrain_types[:] = c
        t.env_origins[:] = t.terrain_origins[r, c]

    env, wrapper, policy, clip = build(args.ckpt, cfg, args.num_envs, spawn=spawn)
    num_steps = args.num_steps or round(args.time_limit / env.step_dt)
    # Course start (x = 0) and lane edge (y = 0; lane centre y = WIDTH / 2).
    origin = env.scene.terrain.env_origins[:, :2] - torch.tensor([1.0, WIDTH / 2], device=env.device)
    origin_x, origin_y = origin.cpu().numpy().T

    # Keep the state right after the evaluation reset (inside record()) for identity checks.
    initial, reset = {}, env.reset

    def reset_and_capture(*a, **k):
        out = reset(*a, **k)
        robot, command = env.scene["robot"], env.command_manager.get_term("twist")
        initial["init_root_state"] = torch.cat([
            robot.data.root_link_pos_w, robot.data.root_link_quat_w,
            robot.data.root_link_lin_vel_w, robot.data.root_link_ang_vel_w,
        ], -1).cpu().numpy()
        initial["init_joint_pos"] = robot.data.joint_pos.cpu().numpy()
        initial["init_command_b"] = command.command.cpu().numpy()
        initial["cmd_vel_h"] = command.vel_command_h.cpu().numpy()  # Heading-frame (v_x, v_y).
        return out

    env.reset = reset_and_capture
    started = time.time()
    data = record(env, wrapper, policy, clip, num_steps)
    print(f"recorded {num_steps} steps x {args.num_envs} envs in {time.time() - started:.1f} s", flush=True)
    model = env.sim.model
    pos = data.pop("pos_w")
    data["course_x"] = pos[..., 0] - origin_x
    data["course_y"] = pos[..., 1] - origin_y  # Lateral position in the lane (centre 1.8 m).
    data["course_end"] = np.float64(COURSE_END)
    data["windows"] = np.array([WINDOWS[k] for k in WINDOWS])
    data["time_limit_s"] = np.float64(num_steps * env.step_dt)
    data["difficulty"] = np.array(level_difficulties(args.levels))[rows]
    data["level"], data["lane"] = rows, lanes
    # Training formula independent of difficulty (random_rough): the levels are replicates.
    train_gen = load_env_cfg(spec.task_id).scene.terrain.terrain_generator
    train_sub = train_gen.sub_terrains[args.rough]
    data["levels_are_replicates"] = np.bool_(args.levels > 1 and len({
        json.dumps(L.formula_parameters(train_sub, d), sort_keys=True) for d in level_difficulties(args.levels)
    }) == 1)
    # Geometry parameters of the embedded section per level (e.g. the truncated obstacle heights).
    section, _ = rough_section(train_gen, args.rough)
    section.size = (ROUGH_LEN, WIDTH)
    data["level_params"] = np.array(json.dumps(
        {f"{d:g}": L.formula_parameters(section, d) for d in level_difficulties(args.levels)}, default=float,
    ))
    data["terrain_sha"] = np.array(L.terrain_fingerprint(env.sim.mj_model))
    data["dr_sha"] = np.array(L._array_sha(*[
        getattr(model, name).cpu().numpy() for name in (
            "geom_friction", "body_mass", "body_ipos", "body_inertia", "actuator_gainprm", "actuator_biasprm",
        )
    ], env.scene["robot"].data.encoder_bias.cpu().numpy()))
    data.update(initial)
    out = Path(args.out) / f"{args.rough}__{args.command}__{args.ckpt}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")  # Complete files only (run scripts skip existing npz).
    with part.open("wb") as stream:
        np.savez_compressed(stream, **data)
    part.replace(out)
    print("saved", out)


# --------------------------------------------------------------------------------------
# Metrics.


def step_events(airborne: np.ndarray, clearance: np.ndarray, valid: np.ndarray):
    """Per wheel: list of (start, end) step indices of qualifying swing phases."""
    events = []
    for w in range(airborne.shape[1]):
        mask = np.concatenate([[False], airborne[:, w] & valid, [False]]).astype(np.int8)
        edges = np.diff(mask)
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        events.append([
            (s, e) for s, e in zip(starts, ends)
            if e - s >= STEP_MIN_DURATION and clearance[s:e, w].max() >= STEP_MIN_CLEARANCE
        ])
    return events


def course_geometry(d) -> tuple[float, dict]:
    """Course end and judging windows the rollout was recorded with."""
    if "course_end" not in d:
        return LEGACY_COURSE_END, LEGACY_WINDOWS
    return float(d["course_end"]), {k: tuple(map(float, b)) for k, b in zip(WINDOWS, d["windows"])}


def trajectory_metrics(d, i: int) -> dict:
    course_end, windows = course_geometry(d)
    valid = d["valid"][:, i]
    x = d["course_x"][:, i]
    dt = float(d["step_dt"])
    reached = np.flatnonzero(valid & (x >= course_end))
    end = int(reached[0]) if len(reached) else int(valid.sum())
    failed = (not len(reached)) and (valid.sum() < len(valid))
    window = np.zeros_like(valid)
    window[:end] = valid[:end]
    steady = window.copy()
    if failed:
        steady[max(0, end - 25):] = False  # Drop the fall itself (last 0.5 s).
    airborne = ~d["wheel_contact"][:, i]
    clearance = d["wheel_clearance"][:, i].astype(float)
    events = step_events(airborne, clearance, window)

    def count(lo, hi, wheel=None):
        wheels = range(2) if wheel is None else [wheel]
        return sum(lo <= x[s] < hi for w in wheels for s, _ in events[w])

    flat_in = count(*windows["flat_in"])
    rough_l, rough_r = count(*windows["rough"], 0), count(*windows["rough"], 1)
    flat_out = count(*windows["flat_out"])
    in_window = {k: window & (x >= lo) & (x < hi) for k, (lo, hi) in windows.items()}
    any_air = airborne.any(-1)
    flat_mask = in_window["flat_in"] | in_window["flat_out"]
    rough_mask = in_window["rough"]
    swings_rough = [(w, s, e) for w in range(2) for s, e in events[w] if windows["rough"][0] <= x[s] < windows["rough"][1]]
    cleared = [clearance[s:e, w].max() >= d["clearance_target"][s:e, i, w].astype(float).max() for w, s, e in swings_rough]
    g = d["gravity_b"][:, i].astype(float)
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
    ang = d["ang_vel_b"][:, i].astype(float)
    power = d["power_leg"][:, i].astype(float) + d["power_wheel"][:, i].astype(float)
    dist = float(np.sum(d["speed"][:, i].astype(float)[window]) * dt)
    completed = bool(len(reached))

    def passed(x_end):
        return bool(np.any(window & (x >= x_end)))

    return {
        "completed": completed,
        "failed": bool(failed),
        "time_s": (end + 1) * dt if completed else float("nan"),
        "flat_in_steps": int(flat_in),
        "rough_steps_left": int(rough_l),
        "rough_steps_right": int(rough_r),
        "flat_out_steps": int(flat_out),
        # Window verdicts only for robots that traversed the whole window alive.
        "roll_flat_in": flat_in == 0 if passed(windows["flat_in"][1]) else None,
        "step_rough": (rough_l >= 1 and rough_r >= 1) if passed(windows["rough"][1]) else None,
        "roll_flat_out": flat_out == 0 if completed else None,
        "switch_success": completed and flat_in == 0 and rough_l >= 1 and rough_r >= 1 and flat_out == 0,
        "flat_lift_rate": float(any_air[flat_mask].mean()) if flat_mask.any() else float("nan"),
        "rough_lift_rate": float(any_air[rough_mask].mean()) if rough_mask.any() else float("nan"),
        "rough_swings": len(swings_rough),
        "rough_cleared_swings": int(np.sum(cleared)),
        "peak_tilt_deg": float(tilt[steady].max()) if steady.any() else float("nan"),
        "mean_tilt_deg": float(tilt[steady].mean()) if steady.any() else float("nan"),
        "roll_pitch_rate": float(np.linalg.norm(ang[steady, :2], axis=-1).mean()) if steady.any() else float("nan"),
        "energy_per_m": float(np.sum(power[window]) * dt / dist) if dist > 1.0 else float("nan"),
        "cot": float(np.sum(power[window]) * dt / dist / (d["mass_kg"][i] * 9.81)) if dist > 1.0 else float("nan"),
    }


def lateral_metrics(d, i: int) -> dict:
    """Lateral offset from the lane centre: at the start, when crossing the section centre
    (x = 5.8 m, the apex of pyramid-like sections) and its maximum until the course end."""
    course_end, _ = course_geometry(d)
    valid = d["valid"][:, i]
    x = d["course_x"][:, i]
    dy = d["course_y"][:, i].astype(float) - WIDTH / 2
    reached = np.flatnonzero(valid & (x >= course_end))
    window = valid.copy()
    window[(int(reached[0]) if len(reached) else len(valid)):] = False
    centre = np.flatnonzero(window & (x >= FLAT_IN + ROUGH_LEN / 2))
    return {
        "dy_start": float(dy[0]),
        "dy_centre": float(dy[centre[0]]) if len(centre) else None,
        "max_abs_dy": float(np.abs(dy[window]).max()) if window.any() else float("nan"),
    }


def summarize(sel: list[dict]) -> dict:
    mean = lambda k: float(np.nanmean([np.nan if r[k] is None else float(r[k]) for r in sel]))  # noqa: E731
    swings = sum(r["rough_swings"] for r in sel)
    extra = {"time_ratio": mean("time_ratio")} if all("time_ratio" in r for r in sel) else {}
    return {
        "n": len(sel),
        "completion_pct": 100 * mean("completed"),
        "switch_success_pct": 100 * mean("switch_success"),
        "roll_flat_in_pct": 100 * mean("roll_flat_in"),
        "step_rough_pct": 100 * mean("step_rough"),
        "roll_flat_out_pct": 100 * mean("roll_flat_out"),
        "flat_lift_rate_pct": 100 * mean("flat_lift_rate"),
        "rough_lift_rate_pct": 100 * mean("rough_lift_rate"),
        "rough_clearance_pct": 100 * sum(r["rough_cleared_swings"] for r in sel) / swings if swings else float("nan"),
        "peak_tilt_deg": mean("peak_tilt_deg"),
        "roll_pitch_rate": mean("roll_pitch_rate"),
        "cot": mean("cot"),
        "time_s": mean("time_s"),
        **extra,
    }


# Summary statistics as ratios of per-trajectory sums (numerator, denominator), so that a
# resampled mean is sum(w * num) / sum(w * den): mean over trajectories with a value
# (None / NaN excluded, as in summarize()) and clearance per rough swing.
STAT_KEYS = {
    "completed": "completion_pct", "failed": "failure_pct", "switch_success": "switch_success_pct",
    "roll_flat_in": "roll_flat_in_pct", "step_rough": "step_rough_pct", "roll_flat_out": "roll_flat_out_pct",
    "flat_lift_rate": "flat_lift_rate_pct", "rough_lift_rate": "rough_lift_rate_pct",
    "peak_tilt_deg": "peak_tilt_deg", "roll_pitch_rate": "roll_pitch_rate", "cot": "cot",
    "time_ratio": "time_ratio",
}
STAT_NAMES = (*STAT_KEYS.values(), "rough_clearance_pct")
STAT_SCALE = np.array([100.0 if name.endswith("_pct") else 1.0 for name in STAT_NAMES])


def cluster_sums(sel: list[dict], clusters: list) -> tuple[np.ndarray, np.ndarray]:
    """Per (difficulty, lane) cluster: sums of the statistic numerators and denominators."""
    values = np.array([[np.nan if r.get(k) is None else float(r[k]) for k in STAT_KEYS] for r in sel])
    num = np.column_stack([np.where(np.isfinite(values), values, 0.0), [r["rough_cleared_swings"] for r in sel]])
    den = np.column_stack([np.isfinite(values), [r["rough_swings"] for r in sel]]).astype(float)
    index = {c: j for j, c in enumerate(clusters)}
    ids = np.array([index[(r["difficulty"], r["lane"])] for r in sel])
    sums = np.zeros((2, len(clusters), num.shape[1]))
    np.add.at(sums[0], ids, num)
    np.add.at(sums[1], ids, den)
    return sums[0], sums[1]


def bootstrap_weights(clusters: list) -> np.ndarray:
    """[BOOTSTRAP, clusters] resampling counts; lanes are resampled within each difficulty."""
    rng = np.random.default_rng(20260928)
    weights = np.zeros((BOOTSTRAP, len(clusters)))
    for level in sorted({c[0] for c in clusters}):
        idx = [j for j, c in enumerate(clusters) if c[0] == level]
        weights[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=BOOTSTRAP)
    return weights


def ratio(weights, num, den) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return STAT_SCALE * (weights @ num) / (weights @ den)


def interval(num, den, weights) -> tuple[np.ndarray, np.ndarray]:
    """(point estimate, bootstrap replicates) of every statistic."""
    return ratio(np.ones((1, len(num))), num, den)[0], ratio(weights, num, den)


def percentiles(point: np.ndarray, boot: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN statistics (e.g. nothing completed).
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {name: [float(p), float(a), float(b)] for name, p, a, b in zip(STAT_NAMES, point, lo, hi)}


def cluster_stats(groups: dict) -> dict:
    """95 % lane-cluster bootstrap intervals per group, and paired BASELINE - ckpt differences
    (same robots, same resampled lanes) for groups with per-env lane data."""
    cells, paired = {}, {}
    for key, sel in groups.items():
        if not all("lane" in r for r in sel):
            continue
        clusters = sorted({(r["difficulty"], r["lane"]) for r in sel})
        weights = bootstrap_weights(clusters)
        point, boot = interval(*cluster_sums(sel, clusters), weights)
        cells["|".join(key)] = {"n": len(sel), "clusters": len(clusters), **percentiles(point, boot)}
        base_key = (*key[:2], BASELINE, *key[3:])
        if key[2] == BASELINE or base_key not in groups:
            continue
        if [r["env"] for r in groups[base_key]] != [r["env"] for r in sel]:
            raise RuntimeError(f"{base_key} and {key} do not hold the same robots.")
        b_point, b_boot = interval(*cluster_sums(groups[base_key], clusters), weights)
        paired["|".join((*key[:2], f"{BASELINE}-{key[2]}", *key[3:]))] = {
            name: [p, lo, hi, "higher" if lo > 0 else "lower" if hi < 0 else "n.s."]
            for name, (p, lo, hi) in percentiles(b_point - point, b_boot - boot).items()
        }
    return {
        "method": f"95 % percentile intervals, {BOOTSTRAP} lane-cluster bootstrap replicates (lanes resampled "
                  "within each difficulty level); values are [estimate, low, high]; paired: "
                  f"{BASELINE} - ckpt on the same robots, verdict 'higher' / 'lower' if the interval excludes 0.",
        "cells": cells, "paired": paired,
    }


def speed_bin(speed: float) -> str:
    k = min(max(int(np.searchsorted(SPEED_BINS, speed, side="right")) - 1, 0), len(SPEED_BINS) - 2)
    return f"v=[{SPEED_BINS[k]:g},{SPEED_BINS[k + 1]:g})"


def identity_checks(runs: dict) -> dict:
    """Per (rough, command): do all checkpoints share terrain, DR, reset states and commands?"""
    checks = {}
    for rough, command in sorted({k[:2] for k in runs}):
        group = {k[2]: v for k, v in runs.items() if k[:2] == (rough, command)}
        ref_ckpt, ref = next(iter(group.items()))
        keys = [k for k in IDENTITY_KEYS if k in ref]
        checks[f"{rough}|{command}"] = {
            "ckpts": list(group), "reference": ref_ckpt,
            "identical": {k: all(k in v and np.array_equal(v[k], ref[k]) for v in group.values()) for k in keys},
        }
    return checks


def aggregate(args):
    out = Path(args.out)
    rows, runs = [], {}
    for path in sorted(out.glob("*__*__*.npz")):
        rough, command, ckpt = path.stem.split("__")
        d = dict(np.load(path))  # NpzFile would re-decompress on every access.
        runs[(rough, command, ckpt)] = {
            k: d[k] for k in (*IDENTITY_KEYS, "levels_are_replicates", "level_params") if k in d
        }
        course_end, _ = course_geometry(d)
        for i in range(d["valid"].shape[1]):
            meta, extra = {}, {}
            if "difficulty" in d:
                meta = {
                    "difficulty": float(d["difficulty"][i]), "lane": int(d["lane"][i]),
                    "cmd_speed": float(np.linalg.norm(d["cmd_vel_h"][i])),
                }
            metrics = trajectory_metrics(d, i)
            if "course_y" in d:
                # Completion time relative to the time the commanded speed needs from the start.
                nominal = (course_end - float(d["course_x"][0, i])) / meta["cmd_speed"]
                extra = {"time_ratio": metrics["time_s"] / nominal, **lateral_metrics(d, i)}
            rows.append({"rough": rough, "command": command, "ckpt": ckpt, **meta, "env": i, **metrics, **extra})

    # Identity first: outputs of checkpoints that did not see identical conditions are not written.
    checks = identity_checks(runs)
    (out / "identity_checks.json").write_text(json.dumps(checks, indent=1))
    failed, verdicts = [], []
    for key, check in checks.items():
        bad = [k for k, ok in check["identical"].items() if not ok]
        failed += bad
        verdict = "n/a (no identity data)" if not check["identical"] else f"MISMATCH {bad}" if bad else "OK"
        verdicts.append(f"identity {key} {check['ckpts']}: {verdict}")
    marker = out / "AGGREGATE_FAILED"
    if failed:
        print("\n".join(verdicts))
        for name in ("trajectories.csv", "summary.json", "levels.json", "stats.json"):
            (out / name).unlink(missing_ok=True)  # Stale outputs of an earlier aggregate would look valid.
        marker.write_text("identity check failed, see identity_checks.json; no summary written\n")
        sys.exit("identity check failed: checkpoints did not see identical terrain / states / commands")
    marker.unlink(missing_ok=True)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for r in rows:
        key = (r["rough"], r["command"], r["ckpt"])
        groups.setdefault(key, []).append(r)
        if "difficulty" in r:  # The key without difficulty pools all levels.
            level, speed = f"d={r['difficulty']:g}", speed_bin(r["cmd_speed"])
            for sub in ((level,), (level, speed), (speed,)):
                groups.setdefault((*key, *sub), []).append(r)
    summary = {"|".join(key): summarize(groups[key]) for key in sorted(groups)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for key, value in summary.items():
        print(key, {k: round(v, 3) for k, v in value.items()})
    stats = cluster_stats(groups)
    if stats["cells"]:
        (out / "stats.json").write_text(json.dumps(stats, indent=1))
        tied = sum(v[3] == "n.s." for cell in stats["paired"].values() for v in cell.values())
        total = sum(len(cell) for cell in stats["paired"].values())
        print(f"stats.json: {len(stats['cells'])} groups with 95 % intervals, {len(stats['paired'])} paired "
              f"{BASELINE} - baseline groups ({tied} of {total} differences not significant)")
    replicates = sorted({k[0] for k, v in runs.items() if v.get("levels_are_replicates", False)})
    if replicates:
        print("difficulty levels are replicates (geometry independent of difficulty):", replicates)
    levels = {"levels_are_replicates": replicates}
    params = {k[0]: json.loads(str(v["level_params"])) for k, v in sorted(runs.items()) if "level_params" in v}
    if params:
        levels["level_parameters"] = params  # Embedded-section geometry per difficulty level.
    (out / "levels.json").write_text(json.dumps(levels, indent=1))
    print("\n".join(verdicts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--command", choices=tuple(COMMANDS), default="forward")
    r.add_argument("--rough", choices=ROUGH_TYPES, default="tilted_grid")
    r.add_argument("--levels", type=int, default=0, help="difficulty rows (k+1)/N x max; 0: max only")
    r.add_argument("--num-envs", type=int, default=NUM_ENVS)
    r.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="seconds per trajectory")
    r.add_argument("--num-steps", type=int, default=None, help="policy steps; overrides --time-limit")
    r.add_argument("--seed", type=int, default=20260928)
    r.add_argument("--out", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--out", required=True)
    args = parser.parse_args()
    {"rollout": rollout, "aggregate": aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
