"""Tidy tables of the mode_switch_v2 rollouts (flat -> rough -> flat, forward_wide, 4 levels).

Data engineering only (no new simulation, nothing outside analysis/ is written):
  per_traj.csv.gz          one row per (terrain, difficulty, lane, env, policy): all trajectories.csv fields,
                           recomputed from the npz files with scripts/eval/mode_switch_eval.py, plus derived
                           fields (speed bin, roughness-gate shares, lift events per window and wheel with and
                           without the 3 cm clearance condition, gate-aware mode success, drift flag).
  per_traj_replicate.csv.gz  the same for the Ours replicate run (replicate/, run-to-run noise).
  cell_modes.csv           per (terrain, difficulty): pooled gate share in the rough window and expected mode.
  cells.csv                per group x metric: policy estimates with 95 % lane-bootstrap intervals, paired
                           Ours - RGGP / Ours - LPGP differences (bootstrap CI, lane sign-flip permutation p,
                           Holm-adjusted p), floor/ceiling flag, noise floor, robust flags.
  validation.json          all validation numbers (also written into TIDY_README.md).
  TIDY_README.md           field descriptions and validation results.

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/tidy.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
RUN = HERE.parent  # logs/lipm_eval/mode_switch_v2 (read only)
ROOT = HERE.parents[3]
EVAL = ROOT / "scripts/eval/mode_switch_eval.py"


def load_module(name: str, clearance: float | None = None):
    """Independent module instance of mode_switch_eval (module globals are not shared)."""
    spec = importlib.util.spec_from_file_location(name, EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if clearance is not None:
        mod.STEP_MIN_CLEARANCE = clearance
    return mod


M = load_module("mode_switch_eval")  # Pre-specified rule: >= 0.06 s airborne and >= 3 cm clearance.
MD = load_module("mode_switch_eval_duration_only", -np.inf)  # Duration only (clearance_sensitivity.py).
assert M.STEP_MIN_CLEARANCE == 0.03 and MD.STEP_MIN_CLEARANCE == -np.inf

COMMAND = "forward_wide"
POLICIES = ("Ours", "RGGP", "LPGP")
LABELS = {"Ours": "Ours", "RGGP": "Ours w/o RG", "LPGP": "Ours w/o LP"}
BASELINES = ("RGGP", "LPGP")
WINDOW_NAMES = ("flat_in", "rough", "flat_out")
SPEED_EDGES = (0.5, 1.0, 1.5, 2.0)
SPEED_LABELS = ("[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0]")
DRIFT_BAND = 1.5  # m, |y - lane centre| (tools/per_env_checks.py LANE_BAND)
GATE_STEP_SHARE = 0.5  # expected mode 'step' if the pooled rough-window gate share >= this
BOOTSTRAP = 2000
BOOT_SEED = 20260930
BOOT_SEED_ALT = 20260931  # second independent seed: Monte-Carlo reference for the stats.json check
NPERM_MC = 10000
EXACT_MAX_LANES = 16  # exact sign-flip enumeration (2^16 = 65536) up to this many lanes
WORKERS = 6

# name, per-trajectory column, kind, tier, better, noise_floor.json metric (None: no matching metric)
METRICS = [
    ("switch_success_pct", "switch_success", "binary", "primary", "higher", "switch_success_pct"),
    ("completion_pct", "completed", "binary", "primary", "higher", "completion_pct"),
    ("cot", "cot", "continuous", "primary", "lower", "cot"),
    ("failure_pct", "failed", "binary", "secondary", "lower", "failure_pct"),
    ("roll_flat_in_pct", "roll_flat_in", "binary", "secondary", "higher", "roll_flat_in_pct"),
    ("step_rough_pct", "step_rough", "binary", "secondary", "higher", "step_rough_pct"),
    ("roll_flat_out_pct", "roll_flat_out", "binary", "secondary", "higher", "roll_flat_out_pct"),
    ("flat_lift_rate_pct", "flat_lift_rate", "step_fraction", "secondary", "lower", None),
    ("rough_lift_rate_pct", "rough_lift_rate", "step_fraction", "secondary", "none", None),
    ("peak_tilt_deg", "peak_tilt_deg", "continuous", "secondary", "lower", "peak_tilt_deg"),
    ("time_ratio", "time_ratio", "continuous", "secondary", "none", None),
    ("mode_success_pct", "mode_success", "binary", "secondary", "higher", None),
    ("switch_success_dur_pct", "switch_success_dur", "binary", "secondary", "higher", None),
]
NAMES = [m[0] for m in METRICS]
SCALE = np.array([100.0 if m[0].endswith("_pct") else 1.0 for m in METRICS])
STATS_JSON_METRICS = [n for n in NAMES if n not in ("mode_success_pct", "switch_success_dur_pct")]


# --------------------------------------------------------------------------------------
# Per-trajectory recomputation.


def process(path_str: str) -> list[dict]:
    path = Path(path_str)
    terrain, command, ckpt = path.stem.split("__")
    d = dict(np.load(path))  # NpzFile would re-decompress on every access.
    course_end, windows = M.course_geometry(d)
    rows = []
    for i in range(d["valid"].shape[1]):
        # --- identical to mode_switch_eval.aggregate() (trajectories.csv fields) ---
        meta = {"difficulty": float(d["difficulty"][i]), "lane": int(d["lane"][i]),
                "cmd_speed": float(np.linalg.norm(d["cmd_vel_h"][i]))}
        metrics = M.trajectory_metrics(d, i)
        nominal = (course_end - float(d["course_x"][0, i])) / meta["cmd_speed"]
        extra = {"time_ratio": metrics["time_s"] / nominal, **M.lateral_metrics(d, i)}
        row = {"rough": terrain, "command": command, "ckpt": ckpt, **meta, "env": i, **metrics, **extra}
        # --- derived fields ---
        valid = d["valid"][:, i]
        x = d["course_x"][:, i]
        reached = np.flatnonzero(valid & (x >= course_end))  # Same judged span as trajectory_metrics().
        end = int(reached[0]) if len(reached) else int(valid.sum())
        window = np.zeros_like(valid)
        window[:end] = valid[:end]
        airborne = ~d["wheel_contact"][:, i]
        clearance = d["wheel_clearance"][:, i].astype(float)
        gate = d["rough"][:, i]
        row["level"] = int(d["level"][i])
        row["x_start"] = float(x[0])
        for name, (lo, hi) in windows.items():
            mask = window & (x >= lo) & (x < hi)
            row[f"n_alive_{name}"] = int(mask.sum())
            row[f"n_gate_{name}"] = int((mask & gate).sum())
        for tag, mod in (("lift", M), ("dlift", MD)):
            events = mod.step_events(airborne, clearance, window)
            for name, (lo, hi) in windows.items():
                for w, side in enumerate("LR"):  # wheel 0 = left (rough_steps_left in trajectory_metrics)
                    row[f"{tag}_{name}_{side}"] = int(sum(lo <= x[s] < hi for s, _ in events[w]))
        dur = MD.trajectory_metrics(d, i)
        for k in ("roll_flat_in", "step_rough", "roll_flat_out", "switch_success", "flat_in_steps",
                  "rough_steps_left", "rough_steps_right", "flat_out_steps"):
            row[f"{k}_dur"] = dur[k]
        rows.append(row)
    return rows


def fmt(value) -> str:
    """Value as csv.DictWriter writes it in aggregate() (None -> '')."""
    return "" if value is None else str(value)


# --------------------------------------------------------------------------------------
# Cluster statistics.


def group_seed(base: int, key: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base, zlib.crc32(key.encode())]))


def lane_weights(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[BOOTSTRAP, clusters] counts; lanes resampled within each (terrain, difficulty) stratum."""
    w = np.zeros((BOOTSTRAP, len(strata)))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=BOOTSTRAP)
    return w


def ratio(w: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return SCALE * (w @ num) / (w @ den)


def pct(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return lo, hi


def signflip_p(nA, dA, nB, dB, rng) -> tuple[np.ndarray, str]:
    """Two-sided lane-level sign-flip (label swap per lane) permutation p of est_A - est_B."""
    L = len(nA)
    if L <= EXACT_MAX_LANES:
        S = ((np.arange(2 ** L)[:, None] >> np.arange(L)) & 1).astype(float)
        method = f"exact({2 ** L})"
    else:
        S = rng.integers(0, 2, size=(NPERM_MC, L)).astype(float)
        method = f"mc({NPERM_MC})"
    sA_n, sA_d, sB_n, sB_d = S @ nA, S @ dA, S @ nB, S @ dB
    tA_n, tA_d, tB_n, tB_d = nA.sum(0), dA.sum(0), nB.sum(0), dB.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # S = 1: lane keeps its labels; S = 0: Ours and baseline swapped in that lane.
        a = (sA_n + tB_n - sB_n) / (sA_d + tB_d - sB_d)
        b = (sB_n + tA_n - sA_n) / (sB_d + tA_d - sA_d)
        T = SCALE * (a - b)
        obs = SCALE * (tA_n / tA_d - tB_n / tB_d)
    finite = np.isfinite(T)
    tol = 1e-9 * np.maximum(1.0, np.abs(obs))
    extreme = (np.abs(T) >= np.abs(obs) - tol) & finite
    n_ok = finite.sum(0)
    if method.startswith("exact"):
        p = extreme.sum(0) / n_ok  # identity included
    else:
        p = (1 + extreme.sum(0)) / (1 + n_ok)
    p = np.where(np.isfinite(obs) & (n_ok > 0), p, np.nan)
    return p, method


def holm(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    m = len(order)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out[order] = adj
    return out


class Data:
    """Per-policy value matrices aligned on the same robots (terrain, env)."""

    def __init__(self, df: pd.DataFrame, policies=POLICIES):
        parts = {p: df[df.ckpt == p].sort_values(["rough", "env"]).reset_index(drop=True) for p in policies}
        ref = parts[policies[0]]
        for p in policies[1:]:
            for k in ("rough", "env", "difficulty", "lane", "cmd_speed"):
                if not np.array_equal(parts[p][k].to_numpy(), ref[k].to_numpy()):
                    raise RuntimeError(f"{p} not aligned with {policies[0]} on {k}")
        self.meta = ref[["rough", "env", "difficulty", "lane", "level", "speed_bin"]].copy()
        self.terrains = sorted(ref.rough.unique())
        self.diffs = sorted(ref.difficulty.unique())
        t_idx = np.searchsorted(self.terrains, ref.rough.to_numpy())
        d_idx = np.searchsorted(self.diffs, ref.difficulty.to_numpy())
        self.stratum = t_idx * len(self.diffs) + d_idx
        self.cluster = self.stratum * M.LANES + ref.lane.to_numpy()
        self.V = {p: np.column_stack([as_float(parts[p][col]) for _, col, *_ in METRICS]) for p in policies}

    def sums(self, idx: np.ndarray, p: str):
        uniq, inv = np.unique(self.cluster[idx], return_inverse=True)
        v = self.V[p][idx]
        num = np.zeros((len(uniq), v.shape[1]))
        den = np.zeros_like(num)
        np.add.at(num, inv, np.where(np.isfinite(v), v, 0.0))
        np.add.at(den, inv, np.isfinite(v).astype(float))
        return uniq, num, den


def as_float(col: pd.Series) -> np.ndarray:
    return np.array([np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in col])


def group_stats(data: Data, idx: np.ndarray, key: str, weights: np.ndarray | None = None,
                seed: int = BOOT_SEED, with_perm: bool = True) -> dict:
    pols = list(data.V)
    a, others = pols[0], pols[1:]  # paired differences a - other (main data: Ours - RGGP, Ours - LPGP)
    sums = {p: data.sums(idx, p) for p in pols}
    uniq = sums[a][0]
    strata = uniq // M.LANES
    w = lane_weights(strata, group_seed(seed, key)) if weights is None else weights
    ones = np.ones((1, len(uniq)))
    out = {"n_robots": len(idx), "n_lanes": len(uniq), "n_cells": len(np.unique(strata)),
           "clusters": uniq, "est": {}, "boot": {}, "n_valid": {}}
    for p, (_, num, den) in sums.items():
        out["est"][p] = ratio(ones, num, den)[0]
        out["boot"][p] = ratio(w, num, den)
        out["n_valid"][p] = den.sum(0)
    out["diff"] = {}
    for b in others:
        diff_boot = out["boot"][a] - out["boot"][b]
        lo, hi = pct(diff_boot)
        entry = {"est": out["est"][a] - out["est"][b], "lo": lo, "hi": hi}
        if with_perm:
            _, nA, dA = sums[a]
            _, nB, dB = sums[b]
            entry["p"], entry["method"] = signflip_p(nA, dA, nB, dB, group_seed(seed + 7, f"{key}|perm|{b}"))
        out["diff"][b] = entry
    return out


def build_table(results: dict, groups: list, pols, mode_of: dict, floors) -> pd.DataFrame:
    """Long table: one row per group x metric; paired differences pols[0] - other."""
    nf_p90, repl_p90_cell, repl_p90_speed = floors
    a, others = pols[0], list(pols[1:])
    rows = []
    for scope, t, dd, sb, _ in groups:
        res = results[gkey(scope, t, dd, sb)]
        for j, (name, col, kind, tier, better, nf_name) in enumerate(METRICS):
            row = {"scope": scope, "terrain": t, "difficulty": dd, "speed_bin": sb, "metric": name,
                   "metric_kind": kind, "tier": tier, "better": better,
                   "expected_mode": mode_of[(t, dd)] if scope in ("cell", "cell_speed") else "",
                   "n_robots": res["n_robots"], "n_lanes": res["n_lanes"], "n_cells": res["n_cells"]}
            vals = []
            for p in pols:
                lo, hi = pct(res["boot"][p][:, [j]])
                row[p] = res["est"][p][j]
                row[f"{p}_lo"], row[f"{p}_hi"] = lo[0], hi[0]
                row[f"{p}_n"] = int(res["n_valid"][p][j])
                vals.append(res["est"][p][j])
            if kind in ("binary", "step_fraction"):
                vals = np.array(vals)
                row["floor_ceiling"] = "floor" if np.all(vals < 5) else "ceiling" if np.all(vals > 95) else "informative"
            else:
                row["floor_ceiling"] = "not_rate"
            row["noise_metric"] = nf_name or ""
            row["noise_p90"] = nf_p90[nf_name] if nf_name else np.nan
            row["noise_p90_repl_cell"] = repl_p90_cell[name]
            row["noise_p90_repl_speed"] = repl_p90_speed[name]
            for b in others:
                e = res["diff"][b]
                est, lo, hi = e["est"][j], e["lo"][j], e["hi"][j]
                excl = bool(lo > 0 or hi < 0)
                row[f"diff_{b}"], row[f"diff_{b}_lo"], row[f"diff_{b}_hi"] = est, lo, hi
                row[f"ci_excl0_{b}"] = excl
                row[f"perm_p_{b}"] = e["p"][j]
                row[f"perm_method_{b}"] = e["method"]
                row[f"robust_{b}"] = (excl and abs(est) > row["noise_p90"]) if nf_name else pd.NA
                repl = row["noise_p90_repl_speed"] if scope in ("cell_speed", "speed_bin") else row["noise_p90_repl_cell"]
                row[f"robust_repl_{b}"] = excl and abs(est) > repl
                if better == "none":
                    row[f"outcome_{b}"] = "no_direction"
                elif not nf_name:
                    row[f"outcome_{b}"] = "no_noise_floor"
                else:
                    good = est > 0 if better == "higher" else est < 0
                    row[f"outcome_{b}"] = (f"{a} better" if good else f"{a} worse") if row[f"robust_{b}"] else "not robust"
            rows.append(row)
    table = pd.DataFrame(rows)
    for b in others:
        table[f"perm_p_holm_{b}"] = np.nan
        table[f"holm_m_{b}"] = 0
        for _, sub in table.groupby(["scope", "metric"]):
            pv = sub[f"perm_p_{b}"].to_numpy(dtype=float)
            table.loc[sub.index, f"perm_p_holm_{b}"] = holm(pv)
            table.loc[sub.index, f"holm_m_{b}"] = int(np.isfinite(pv).sum())
    lead = ["scope", "terrain", "difficulty", "speed_bin", "metric", "metric_kind", "tier", "better", "expected_mode",
            "n_robots", "n_lanes", "n_cells"]
    pol = [f"{p}{s}" for p in pols for s in ("", "_lo", "_hi", "_n")]
    comp = []
    for b in others:
        comp += [f"diff_{b}", f"diff_{b}_lo", f"diff_{b}_hi"] + [
            f"{k}_{b}" for k in ("ci_excl0", "perm_p", "perm_p_holm", "holm_m", "perm_method", "robust", "robust_repl",
                                 "outcome")]
    tail = ["floor_ceiling", "noise_metric", "noise_p90", "noise_p90_repl_cell", "noise_p90_repl_speed"]
    return table[lead + pol + comp + tail]


def gkey(scope, t, dd, sb) -> str:
    return f"{scope}|{t}|{dd if dd == 'ALL' else f'{dd:g}'}|{sb}"


# --------------------------------------------------------------------------------------


def main() -> int:
    started = time.time()
    warnings.simplefilter("ignore", RuntimeWarning)
    files = sorted(RUN.glob(f"*__{COMMAND}__*.npz"))
    rep_files = sorted((RUN / "replicate").glob(f"*__{COMMAND}__*.npz"))
    assert len(files) == 30 and len(rep_files) == 10, (len(files), len(rep_files))
    with ProcessPoolExecutor(WORKERS) as ex:
        results = list(ex.map(process, [str(f) for f in files + rep_files]))
    main_rows = [r for res in results[:len(files)] for r in res]
    rep_rows = [r for res in results[len(files):] for r in res]
    print(f"recomputed {len(main_rows)} + {len(rep_rows)} (replicate) trajectories in {time.time() - started:.0f} s")
    V: dict = {}

    # ---------------- validation 1: every trajectories.csv field, per row ----------------
    with (RUN / "trajectories.csv").open() as stream:
        ref = list(csv.DictReader(stream))
    fields = list(ref[0].keys())
    mine = {(r["rough"], r["ckpt"], r["env"]): r for r in main_rows}
    ref_keys = {(r["rough"], r["ckpt"], int(r["env"])) for r in ref}
    mismatch = {f: 0 for f in fields}
    for r in ref:
        m = mine[(r["rough"], r["ckpt"], int(r["env"]))]
        for f in fields:
            mismatch[f] += fmt(m[f]) != r[f]
    V["trajectories_csv"] = {
        "rows_csv": len(ref), "rows_recomputed": len(main_rows), "same_row_keys": ref_keys == set(mine),
        "fields_compared": len(fields), "compare": "string equality of the csv representation",
        "mismatch_per_field": mismatch, "switch_success_mismatch": mismatch["switch_success"],
        "completed_mismatch": mismatch["completed"], "total_mismatch": sum(mismatch.values()),
    }
    print("trajectories.csv mismatches:", sum(mismatch.values()), "switch_success", mismatch["switch_success"],
          "completed", mismatch["completed"])

    # ---------------- tidy frames ----------------
    def tidy(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df.insert(0, "terrain", df["rough"])
        df.insert(1, "policy", df["ckpt"])
        df.insert(2, "policy_label", df["ckpt"].map(LABELS))
        k = np.clip(np.searchsorted(SPEED_EDGES, df.cmd_speed.to_numpy(), side="right") - 1, 0, 2)
        df["speed_bin"] = np.array(SPEED_LABELS)[k]
        df["speed_bin_eval"] = [M.speed_bin(v) for v in df.cmd_speed]  # label used in stats.json keys
        df["cluster"] = df.rough + "|d=" + df.difficulty.map(lambda v: f"{v:g}") + "|lane=" + df.lane.astype(str)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["gate_share_rough"] = df.n_gate_rough / df.n_alive_rough.replace(0, np.nan)
            nf = df.n_alive_flat_in + df.n_alive_flat_out
            df["gate_share_flat"] = (df.n_gate_flat_in + df.n_gate_flat_out) / nf.replace(0, np.nan)
            df["gate_share_flat_in"] = df.n_gate_flat_in / df.n_alive_flat_in.replace(0, np.nan)
            df["gate_share_flat_out"] = df.n_gate_flat_out / df.n_alive_flat_out.replace(0, np.nan)
        for tag in ("lift", "dlift"):
            for name in WINDOW_NAMES:
                df[f"{tag}_{name}"] = df[f"{tag}_{name}_L"] + df[f"{tag}_{name}_R"]
        df["drift"] = df.max_abs_dy >= DRIFT_BAND
        return df

    df = tidy(main_rows)
    rep = tidy(rep_rows)

    # ---------------- validation 2: internal consistency of derived event counts ----------------
    internal = {}
    for data_name, frame in (("main", df), ("replicate", rep)):
        c = {
            "lift_flat_in == flat_in_steps": int((frame.lift_flat_in != frame.flat_in_steps).sum()),
            "lift_rough_L == rough_steps_left": int((frame.lift_rough_L != frame.rough_steps_left).sum()),
            "lift_rough_R == rough_steps_right": int((frame.lift_rough_R != frame.rough_steps_right).sum()),
            "lift_flat_out == flat_out_steps": int((frame.lift_flat_out != frame.flat_out_steps).sum()),
            "dlift_flat_in == flat_in_steps_dur": int((frame.dlift_flat_in != frame.flat_in_steps_dur).sum()),
            "dlift_rough_L == rough_steps_left_dur": int((frame.dlift_rough_L != frame.rough_steps_left_dur).sum()),
            "dlift_rough_R == rough_steps_right_dur": int((frame.dlift_rough_R != frame.rough_steps_right_dur).sum()),
            "dlift_flat_out == flat_out_steps_dur": int((frame.dlift_flat_out != frame.flat_out_steps_dur).sum()),
            "dlift >= lift (every window and wheel)": int(sum(
                (frame[f"dlift_{n}_{s}"] < frame[f"lift_{n}_{s}"]).sum() for n in WINDOW_NAMES for s in "LR")),
            # Duration-only events are a superset: stepping can only gain, rolling can only lose.
            "step_rough implies step_rough_dur": int(((frame.step_rough == True) & (frame.step_rough_dur != True)).sum()),  # noqa: E712
            "roll_flat_in_dur implies roll_flat_in": int(((frame.roll_flat_in_dur == True) & (frame.roll_flat_in != True)).sum()),  # noqa: E712
            "roll_flat_out_dur implies roll_flat_out": int(((frame.roll_flat_out_dur == True) & (frame.roll_flat_out != True)).sum()),  # noqa: E712
        }
        internal[data_name] = c
    V["internal_consistency_violations"] = internal
    print("internal consistency:", internal)

    # ---------------- gate-aware expected mode per cell ----------------
    g = df.groupby(["rough", "difficulty"])
    cm = pd.DataFrame({
        "gate_share_rough_pooled": g.n_gate_rough.sum() / g.n_alive_rough.sum(),
        "gate_share_rough_robot_mean": g.gate_share_rough.mean(),
        "gate_share_flat_pooled": (g.n_gate_flat_in.sum() + g.n_gate_flat_out.sum())
                                  / (g.n_alive_flat_in.sum() + g.n_alive_flat_out.sum()),
        "n_alive_rough_steps": g.n_alive_rough.sum(),
    })
    for p in POLICIES:
        gp = df[df.ckpt == p].groupby(["rough", "difficulty"])
        cm[f"gate_share_rough_{p}"] = gp.n_gate_rough.sum() / gp.n_alive_rough.sum()
    cm["expected_mode"] = np.where(cm.gate_share_rough_pooled >= GATE_STEP_SHARE, "step", "roll")
    cm["expected_mode_robot_mean_def"] = np.where(cm.gate_share_rough_robot_mean >= GATE_STEP_SHARE, "step", "roll")
    cm = cm.reset_index().rename(columns={"rough": "terrain"})
    cm.to_csv(HERE / "cell_modes.csv", index=False, float_format="%.6g")
    mode_of = {(t, dd): m for t, dd, m in zip(cm.terrain, cm.difficulty, cm.expected_mode)}
    V["expected_mode"] = {
        "rule": f"step if pooled (Ours+RGGP+LPGP, step-weighted) rough-window gate share >= {GATE_STEP_SHARE}",
        "cells_step": int((cm.expected_mode == "step").sum()), "cells_roll": int((cm.expected_mode == "roll").sum()),
        "cells_where_robot_mean_definition_differs": cm.loc[cm.expected_mode != cm.expected_mode_robot_mean_def,
                                                            ["terrain", "difficulty"]].values.tolist(),
        "min_abs_distance_of_pooled_share_to_threshold": float(np.min(np.abs(cm.gate_share_rough_pooled - GATE_STEP_SHARE))),
    }

    def add_mode(frame: pd.DataFrame) -> pd.DataFrame:
        frame["expected_mode"] = [mode_of[(t, dd)] for t, dd in zip(frame.rough, frame.difficulty)]
        base = frame.completed & (frame.lift_flat_in == 0) & (frame.lift_flat_out == 0)
        step_ok = (frame.lift_rough_L >= 1) & (frame.lift_rough_R >= 1)
        roll_ok = frame.lift_rough == 0
        frame["mode_success"] = base & np.where(frame.expected_mode == "step", step_ok, roll_ok)
        return frame

    df, rep = add_mode(df), add_mode(rep)
    step_cells = df.expected_mode == "step"
    V["internal_consistency_violations"]["main"]["mode_success == switch_success in 'step' cells"] = int(
        (df.mode_success[step_cells] != df.switch_success[step_cells]).sum())

    # ---------------- write per-trajectory tables ----------------
    front = ["terrain", "difficulty", "level", "lane", "env", "policy", "policy_label", "cluster", "speed_bin"]
    derived = [c for c in df.columns if c not in fields and c not in front]
    order = front + [f for f in fields if f not in front] + derived
    df = df[order].sort_values(["terrain", "difficulty", "lane", "env", "policy"]).reset_index(drop=True)
    rep = rep[order].sort_values(["terrain", "difficulty", "lane", "env", "policy"]).reset_index(drop=True)
    df.to_csv(HERE / "per_traj.csv.gz", index=False)
    rep.to_csv(HERE / "per_traj_replicate.csv.gz", index=False)
    V["per_traj"] = {"rows": len(df), "columns": len(df.columns), "replicate_rows": len(rep),
                     "drift_robots_per_policy": {p: int(df.drift[df.policy == p].sum()) for p in POLICIES}}
    # Drift flag vs the per-env warnings of tools/per_env_checks.py (main files first, then replicate/).
    seen, logged, name = set(), {}, None
    for line in (RUN / "logs/per_env_checks.log").read_text().splitlines():
        if ".npz:" in line:
            name = line.split(".npz:")[0]
            name = ("replicate/" if name in seen else "") + name
            seen.add(line.split(".npz:")[0])
            logged[name] = 0
        elif "robots reach" in line and name is not None:
            logged[name] = int(line.split()[1])
    ours = {}
    for frame, prefix in ((df, ""), (rep, "replicate/")):
        for (t, p), n in frame.groupby(["terrain", "policy"]).drift.sum().items():
            ours[f"{prefix}{t}__{COMMAND}__{p}"] = int(n)
    V["drift_vs_per_env_checks_log"] = {
        "files_in_log": len(logged), "files_compared": len(ours),
        "files_with_different_count": [k for k in ours if logged.get(k) != ours[k]],
        "robots_flagged_main": int(df.drift.sum()), "robots_flagged_replicate": int(rep.drift.sum()),
        "robots_warned_in_log": int(sum(logged.values())),
    }

    # ---------------- groups and statistics ----------------
    data = Data(df)
    meta = data.meta
    groups = []
    for t in data.terrains:
        for dd in data.diffs:
            cell = (meta.rough == t) & (meta.difficulty == dd)
            groups.append(("cell", t, dd, "ALL", np.flatnonzero(cell)))
            for sb in SPEED_LABELS:
                groups.append(("cell_speed", t, dd, sb, np.flatnonzero(cell & (meta.speed_bin == sb))))
    for dd in data.diffs:
        groups.append(("difficulty", "ALL", dd, "ALL", np.flatnonzero(meta.difficulty == dd)))
    for t in data.terrains:
        groups.append(("terrain", t, "ALL", "ALL", np.flatnonzero(meta.rough == t)))
    for sb in SPEED_LABELS:
        groups.append(("speed_bin", "ALL", "ALL", sb, np.flatnonzero(meta.speed_bin == sb)))
    groups.append(("all", "ALL", "ALL", "ALL", np.arange(len(meta))))

    t0 = time.time()
    results = {gkey(*g[:4]): group_stats(data, g[4], gkey(*g[:4])) for g in groups}
    print(f"bootstrap + permutation for {len(groups)} groups in {time.time() - t0:.0f} s")

    # ---------------- noise floors ----------------
    nf = json.loads((RUN / "noise_floor.json").read_text())
    nf_p90 = {m: v["p90_abs_diff"] for m, v in nf["summary_over_cells"].items()}
    both = pd.concat([df[df.policy == "Ours"].assign(ckpt="Ours"), rep.assign(ckpt="Ours_rep")])
    noise_data = Data(both, policies=("Ours", "Ours_rep"))
    rep_cell, rep_speed, nf_cell_check = {n: [] for n in NAMES}, {n: [] for n in NAMES}, []
    for scope, t, dd, sb, _ in groups:
        if scope not in ("cell", "cell_speed"):
            continue
        idx = np.flatnonzero((noise_data.meta.rough == t) & (noise_data.meta.difficulty == dd)
                             & ((noise_data.meta.speed_bin == sb) if scope == "cell_speed" else True))
        est = {}
        for p in ("Ours", "Ours_rep"):
            _, num, den = noise_data.sums(idx, p)
            est[p] = ratio(np.ones((1, len(num))), num, den)[0]
        diff = est["Ours_rep"] - est["Ours"]
        for j, n in enumerate(NAMES):
            (rep_cell if scope == "cell" else rep_speed)[n].append(abs(diff[j]))
        if scope == "cell":
            key = f"{t}__{COMMAND}__Ours".replace("__", "|") + f"|d={dd:g}"
            for j, n in enumerate(NAMES):
                if n in nf["cells"].get(key, {}):
                    nf_cell_check.append(abs(diff[j] - nf["cells"][key][n]["diff"]))
    repl_p90_cell = {n: float(np.nanpercentile(v, 90)) for n, v in rep_cell.items()}
    repl_p90_speed = {n: float(np.nanpercentile(v, 90)) for n, v in rep_speed.items()}
    V["noise_floor"] = {
        "noise_floor_json_p90": nf_p90,
        "replicate_recomputed_p90_cell": repl_p90_cell,
        "replicate_recomputed_p90_cell_speed": repl_p90_speed,
        "max_abs_dev_recomputed_vs_noise_floor_json_p90": float(max(
            abs(repl_p90_cell[m] - nf_p90[m]) for m in nf_p90 if m in repl_p90_cell)),
        "max_abs_dev_recomputed_vs_noise_floor_json_cell_diffs": float(max(nf_cell_check)),
        "n_cell_diffs_compared": len(nf_cell_check),
        "metrics_without_noise_floor_json_match": [m[0] for m in METRICS if m[5] is None],
    }
    print("noise floor recomputation max |dev| p90:", V["noise_floor"]["max_abs_dev_recomputed_vs_noise_floor_json_p90"])

    # ---------------- cells.csv ----------------
    floors = (nf_p90, repl_p90_cell, repl_p90_speed)
    cells = build_table(results, groups, POLICIES, mode_of, floors)
    cells.to_csv(HERE / "cells.csv", index=False, float_format="%.8g")
    V["cells_csv"] = {"rows": len(cells), "groups": len(groups),
                      "groups_per_scope": {s: int(v) for s, v in cells.groupby("scope").size().div(len(METRICS)).items()},
                      "metrics": NAMES}

    # ---------------- null comparison: Ours - Ours replicate (same policy, rerun) ----------------
    null_groups = []
    for scope, t, dd, sb, _ in groups:
        if scope in ("cell", "cell_speed"):
            m_ = (noise_data.meta.rough == t) & (noise_data.meta.difficulty == dd)
            if scope == "cell_speed":
                m_ &= noise_data.meta.speed_bin == sb
            null_groups.append((scope, t, dd, sb, np.flatnonzero(m_)))
    null_results = {gkey(*g[:4]): group_stats(noise_data, g[4], gkey(*g[:4]) + "|null") for g in null_groups}
    null = build_table(null_results, null_groups, ("Ours", "Ours_rep"), mode_of, floors)
    null.to_csv(HERE / "null_replicate_cells.csv", index=False, float_format="%.8g")
    V["null_replicate"] = {
        scope: {name: {
            "n": int(len(sub)),
            "ci_excl0": int(sub.ci_excl0_Ours_rep.sum()),
            "perm_p_lt_0.05": int((sub.perm_p_Ours_rep < 0.05).sum()),
            "robust (noise_floor.json rule)": (int(sub.robust_Ours_rep.fillna(False).astype(bool).sum())
                                               if sub.noise_metric.iloc[0] else None),
            "robust_repl": int(sub.robust_repl_Ours_rep.sum()),
        } for name, sub in null[null.scope == scope].groupby("metric", sort=False)}
        for scope in ("cell", "cell_speed")
    }

    # ---------------- validation 3: stats.json intervals ----------------
    stats = json.loads((RUN / "stats.json").read_text())
    point_dev, exact_dev, mc_dev, ref_dev = [], [], [], []  # |deviation| / CI half-width for the MC comparisons
    n_compared = 0
    missing = 0
    val_groups = [g for g in groups if g[0] in ("cell", "cell_speed", "terrain")]
    for scope, t, dd, sb, idx in val_groups:
        key = gkey(scope, t, dd, sb)
        res = results[key]
        alt = group_stats(data, idx, key, seed=BOOT_SEED_ALT, with_perm=False)
        uniq = res["clusters"]
        cl = [(data.diffs[(c // M.LANES) % len(data.diffs)], int(c % M.LANES)) for c in uniq]
        exact = group_stats(data, idx, key, weights=M.bootstrap_weights(cl), with_perm=False)
        suffix = ("" if dd == "ALL" else f"|d={dd:g}")
        if sb != "ALL":
            suffix += "|" + M.speed_bin((SPEED_EDGES[SPEED_LABELS.index(sb)] + SPEED_EDGES[SPEED_LABELS.index(sb) + 1]) / 2)
        entries = []
        for p in POLICIES:
            sk = f"{t}|{COMMAND}|{p}{suffix}"
            if sk not in stats["cells"]:
                missing += 1
                continue
            if stats["cells"][sk]["n"] != res["n_robots"] or stats["cells"][sk]["clusters"] != res["n_lanes"]:
                raise RuntimeError(f"n / clusters differ for {sk}")
            mine_lo, mine_hi = pct(res["boot"][p])
            alt_lo, alt_hi = pct(alt["boot"][p])
            ex_lo, ex_hi = pct(exact["boot"][p])
            entries.append((stats["cells"][sk], res["est"][p], (mine_lo, mine_hi), (alt_lo, alt_hi), (ex_lo, ex_hi)))
        for b in BASELINES:
            sk = f"{t}|{COMMAND}|Ours-{b}{suffix}"
            if sk not in stats["paired"]:
                missing += 1
                continue
            e, ea, ee = res["diff"][b], alt["diff"][b], exact["diff"][b]
            entries.append((stats["paired"][sk], e["est"], (e["lo"], e["hi"]), (ea["lo"], ea["hi"]), (ee["lo"], ee["hi"])))
        for ref_entry, est, mine_b, alt_b, ex_b in entries:
            for j, n in enumerate(NAMES):
                if n not in STATS_JSON_METRICS:
                    continue
                p0, lo0, hi0 = ref_entry[n][:3]
                if not (np.isfinite(p0) and np.isfinite(lo0) and np.isfinite(hi0)):
                    continue
                n_compared += 1
                point_dev.append(abs(est[j] - p0))
                half = (hi0 - lo0) / 2
                for bound, (a, b_) in enumerate(((lo0, mine_b[0][j]), (hi0, mine_b[1][j]))):
                    exact_dev.append(abs((ex_b[bound][j]) - (lo0, hi0)[bound]))
                    if half > 0:
                        mc_dev.append(abs(b_ - a) / half)
                        ref_dev.append(abs(mine_b[bound][j] - alt_b[bound][j]) / half)
    q = lambda v: {k: float(np.percentile(v, x)) for k, x in (("median", 50), ("p95", 95), ("p99", 99), ("max", 100))}  # noqa: E731
    V["stats_json_check"] = {
        "groups": "cell (terrain x d), cell_speed (terrain x d x speed bin), terrain (pooled over d); per policy and paired",
        "estimates_compared": n_compared, "stats_json_keys_missing": missing,
        "max_abs_point_estimate_deviation": float(max(point_dev)),
        "same_weights_as_stats_json_max_abs_bound_deviation": float(max(exact_dev)),
        "own_seed_vs_stats_json_bound_deviation_over_halfwidth": q(mc_dev),
        "own_seed_vs_second_own_seed_bound_deviation_over_halfwidth (Monte-Carlo reference)": q(ref_dev),
        "n_bounds_with_positive_halfwidth": len(mc_dev),
    }
    print(json.dumps(V["stats_json_check"], indent=1))

    # ---------------- counts for the README ----------------
    robust_counts = {}
    for scope in ("cell", "cell_speed"):
        sub = cells[cells.scope == scope]
        for b in BASELINES:
            for name in NAMES:
                s = sub[sub.metric == name]
                robust_counts.setdefault(scope, {}).setdefault(b, {})[name] = {
                    "Ours better": int((s[f"outcome_{b}"] == "Ours better").sum()),
                    "Ours worse": int((s[f"outcome_{b}"] == "Ours worse").sum()),
                    "robust_na": int(s[f"robust_{b}"].isna().sum()),
                    "floor_or_ceiling": int(s.floor_ceiling.isin(["floor", "ceiling"]).sum()),
                }
    V["outcome_counts_mechanical"] = robust_counts
    V["runtime_s"] = round(time.time() - started, 1)
    (HERE / "validation.json").write_text(json.dumps(V, indent=1, default=float))
    write_readme(V, df, cells, cm)
    print(f"done in {time.time() - started:.0f} s")
    ok = (V["trajectories_csv"]["total_mismatch"] == 0 and V["trajectories_csv"]["same_row_keys"]
          and all(v == 0 for c in V["internal_consistency_violations"].values() for v in c.values())
          and not V["drift_vs_per_env_checks_log"]["files_with_different_count"]
          and V["stats_json_check"]["max_abs_point_estimate_deviation"] < 1e-9
          and V["stats_json_check"]["same_weights_as_stats_json_max_abs_bound_deviation"] < 1e-9)
    print("ALL VALIDATIONS PASSED" if ok else "VALIDATION FAILED")
    return 0 if ok else 1


def write_readme(V: dict, df: pd.DataFrame, cells: pd.DataFrame, cm: pd.DataFrame) -> None:
    tc, sc, nfv, em = V["trajectories_csv"], V["stats_json_check"], V["noise_floor"], V["expected_mode"]
    mc, ref = sc["own_seed_vs_stats_json_bound_deviation_over_halfwidth"], \
        sc["own_seed_vs_second_own_seed_bound_deviation_over_halfwidth (Monte-Carlo reference)"]
    internal = V["internal_consistency_violations"]
    nf_json = nfv["noise_floor_json_p90"]
    dv = V["drift_vs_per_env_checks_log"]
    lines = [
        "# mode_switch_v2 tidy tables",
        "",
        "Produced by `analysis/tidy.py` (run from the repo root with",
        "`uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas python",
        "scripts/eval/mode_switch/analysis/tidy.py`). Reads the 30 rollouts, the 10 replicate rollouts,",
        "trajectories.csv, stats.json and noise_floor.json; writes only into `analysis/`. Data engineering only:",
        "no conclusions are drawn here. All numbers below are generated by tidy.py (also in validation.json).",
        "",
        "## Files",
        "",
        "| file | content |",
        "|---|---|",
        f"| per_traj.csv.gz | {V['per_traj']['rows']} rows (10 terrains x 4 difficulties x 16 lanes x 16 robots x 3 policies), {V['per_traj']['columns']} columns |",
        f"| per_traj_replicate.csv.gz | same columns for the Ours replicate run ({V['per_traj']['replicate_rows']} rows; `policy` = Ours) |",
        "| cell_modes.csv | per (terrain, difficulty): rough-window gate shares (pooled and per policy) and the expected mode |",
        f"| cells.csv | {V['cells_csv']['rows']} rows = {V['cells_csv']['groups']} groups x {len(METRICS)} metrics (long format) |",
        f"| null_replicate_cells.csv | same layout as cells.csv for Ours vs its own replicate run (cell and cell_speed scopes; comparison column suffix `Ours_rep`): a same-policy null comparison |",
        "| validation.json | every validation number |",
        "",
        "## per_traj.csv.gz",
        "",
        "Key: (terrain, difficulty, lane, env, policy). `env` is the index in the rollout (lane = env % 16,",
        "level = (env // 16) % 4); the same (terrain, env) is the same robot (terrain instance, reset state,",
        "command) for all policies. Booleans are `True`/`False`; an empty cell is None (window not traversed).",
        "",
        "* Keys / labels: `terrain` (= `rough`), `policy` (= `ckpt`: Ours / RGGP / LPGP), `policy_label`",
        "  (Ours / Ours w/o RG / Ours w/o LP), `level` (0..3), `cluster` (`terrain|d=..|lane=..`, the resampling",
        "  unit: 16 robots of a lane share one terrain instance).",
        "* All trajectories.csv fields, recomputed with `mode_switch_eval.trajectory_metrics` / `lateral_metrics`",
        "  exactly as `aggregate()` does (see validation 1).",
        "* `speed_bin`: commanded v_x in [0.5,1.0) / [1.0,1.5) / [1.5,2.0] (`speed_bin_eval`: the label used in",
        "  stats.json keys, same bins).",
        "* `x_start`: course x at the first recorded step (time_ratio's nominal distance starts here).",
        "* `n_alive_<w>`, `n_gate_<w>` (w = flat_in, rough, flat_out): alive steps with the base x inside the judging",
        "  window (the same judged span as trajectory_metrics: valid steps before the course end is reached) and",
        "  those with the reward roughness gate active (`rough`, lambda > 0.65).",
        "* `gate_share_rough` = n_gate_rough / n_alive_rough; `gate_share_flat` = gate steps / alive steps pooled over",
        "  flat_in and flat_out; `gate_share_flat_in`, `gate_share_flat_out` per window. NaN if no alive step.",
        "* `lift_<w>_<L|R>`: lift events (>= 0.06 s airborne and peak clearance >= 3 cm, `mode_switch_eval.step_events`)",
        "  starting with the base in window w, per wheel (L = wheel 0 = `rough_steps_left`); `lift_<w>` = L + R.",
        "* `dlift_<w>_<L|R>`, `dlift_<w>`: the same with the duration condition only (a module copy of",
        "  mode_switch_eval with STEP_MIN_CLEARANCE = -inf, as in scripts/eval/lipm_diagnostics/clearance_sensitivity.py).",
        "* `*_dur` (`roll_flat_in_dur`, `step_rough_dur`, `roll_flat_out_dur`, `switch_success_dur`, `flat_in_steps_dur`,",
        "  `rough_steps_left_dur`, `rough_steps_right_dur`, `flat_out_steps_dur`): trajectory_metrics of that module copy,",
        "  i.e. duration-only window verdicts and switch success.",
        "* `expected_mode` (cell level, identical for all policies): `step` if the cell's rough-window gate share",
        "  (sum of n_gate_rough / sum of n_alive_rough over the 768 trajectories of Ours + RGGP + LPGP) >= 0.5, else `roll`.",
        "* `mode_success` = completed AND lift_flat_in == 0 AND lift_flat_out == 0 AND (step: lift_rough_L >= 1 and",
        "  lift_rough_R >= 1; roll: lift_rough == 0). In `step` cells it equals switch_success by construction.",
        "  The replicate uses the expected modes of the main run.",
        f"* `drift`: max_abs_dy >= {DRIFT_BAND} m (robot leaves its lane instance). Robots per policy: {V['per_traj']['drift_robots_per_policy']}.",
        "",
        "## cell_modes.csv",
        "",
        "`gate_share_rough_pooled` (defines `expected_mode`), `gate_share_rough_robot_mean` (mean of per-robot shares)",
        "with `expected_mode_robot_mean_def`, per-policy pooled shares, `gate_share_flat_pooled`, `n_alive_rough_steps`.",
        f"Cells with expected mode step: {em['cells_step']}, roll: {em['cells_roll']}; cells where the robot-mean",
        f"definition gives a different mode: {em['cells_where_robot_mean_definition_differs']}; smallest distance of a",
        f"pooled share to the 0.5 threshold: {em['min_abs_distance_of_pooled_share_to_threshold']:.4f}.",
        "",
        "## cells.csv",
        "",
        "Scopes (`scope`): `cell` (terrain x difficulty, 256 robots, 16 lanes), `cell_speed` (cell x speed_bin),",
        "`difficulty` (all terrains), `terrain` (all difficulties), `speed_bin` (all cells), `all`. `ALL` marks a",
        "pooled dimension. Clusters are lanes nested in cells: bootstrap replicates resample the lanes present in",
        f"the group within each (terrain, difficulty) cell ({BOOTSTRAP} replicates, seed {BOOT_SEED} combined with the",
        "crc32 of the group key; the same resampled lanes for all policies = paired).",
        "",
        "Metrics (`metric`, `tier`, `metric_kind`, `better`): primary switch_success_pct, completion_pct, cot;",
        "secondary failure_pct, roll_flat_in_pct, step_rough_pct, roll_flat_out_pct (% of robots that traversed",
        "the window), flat_lift_rate_pct, rough_lift_rate_pct (mean per-robot share of window steps with any wheel",
        "airborne), peak_tilt_deg, time_ratio, mode_success_pct (gate-aware), switch_success_dur_pct (duration only).",
        "Estimates are means over the robots with a value (None / NaN excluded) = the stats.json / summary.json",
        "definition (ratio of lane sums). `better` = none for rough_lift_rate_pct and time_ratio (no direction).",
        "",
        "Columns:",
        "* `n_robots`, `n_lanes` (clusters), `n_cells`; `expected_mode` for cell / cell_speed rows.",
        "* `<P>`, `<P>_lo`, `<P>_hi`, `<P>_n` for P in Ours, RGGP, LPGP: estimate, 95 % percentile bootstrap interval,",
        "  number of robots with a value (denominator; window verdicts only count robots that traversed the window).",
        "* `diff_<B>` = Ours - B (B = RGGP, LPGP) on the same robots, `diff_<B>_lo/_hi` 95 % paired bootstrap CI,",
        "  `ci_excl0_<B>`. For window verdicts the two rates can have different denominators (as in stats.json).",
        "* `perm_p_<B>`: two-sided lane-level sign-flip permutation p: in each lane the Ours and B labels of all its",
        f"  robots are swapped or not; statistic = the same difference of lane-sum ratios. Exact enumeration when the",
        f"  group has <= {EXACT_MAX_LANES} lanes (cell, cell_speed: 2^16 = 65536 assignments, identity included, smallest",
        f"  possible p = 2/65536), otherwise {NPERM_MC} random assignments with p = (1 + #extreme) / (1 + {NPERM_MC})",
        "  (`perm_method_<B>`). Permutations with an undefined statistic (a zero denominator) are dropped.",
        "* `perm_p_holm_<B>`: Holm adjustment within each (scope, metric, comparison) family, e.g. over the 40 cells;",
        "  `holm_m_<B>`: family size (rows with a finite p).",
        "* `floor_ceiling`: `floor` / `ceiling` if all three policies are < 5 % / > 95 %, else `informative` (rate",
        "  metrics: binary and step_fraction kinds); `not_rate` for continuous metrics.",
        "* `noise_metric`, `noise_p90`: matching metric in noise_floor.json and its p90 |run A - run B| over the 40",
        "  (terrain, difficulty) cells of the Ours replicate. Empty / NaN: noise_floor.json has no matching metric",
        f"  ({', '.join(nfv['metrics_without_noise_floor_json_match'])}).",
        "* `robust_<B>` (pre-specified rule): ci_excl0 AND |diff| > noise_p90; `<NA>` if there is no matching",
        "  noise_floor.json metric. The noise floor is cell-level (256 robots) and is applied unchanged to every scope.",
        "* `outcome_<B>`: `Ours better` / `Ours worse` if robust (direction from `better`), `not robust`,",
        "  `no_direction` (better = none) or `no_noise_floor` (no noise_floor.json metric). Mechanical label only.",
        "* Supplementary (not the pre-specified rule): `noise_p90_repl_cell` / `noise_p90_repl_speed`: p90 of",
        "  |Ours replicate - Ours| recomputed by tidy.py for every metric over the 40 cells / 120 cell x speed-bin groups;",
        "  `robust_repl_<B>`: ci_excl0 AND |diff| > noise_p90_repl_speed (cell_speed and speed_bin rows) or",
        "  noise_p90_repl_cell (other rows).",
        "",
        "## Validation",
        "",
        f"1. trajectories.csv: {tc['rows_csv']} rows vs {tc['rows_recomputed']} recomputed, same row keys:",
        f"   {tc['same_row_keys']}; all {tc['fields_compared']} fields compared as csv strings (exact).",
        f"   Mismatches: switch_success {tc['switch_success_mismatch']}, completed {tc['completed_mismatch']},",
        f"   all fields together {tc['total_mismatch']}.",
        "2. Internal consistency (violations; all must be 0):",
        *[f"   - {k} [{name}]: {v}" for name, c in internal.items() for k, v in c.items()],
        f"3. stats.json intervals ({sc['estimates_compared']} (group, policy or pair, metric) estimates over the cell,",
        f"   cell_speed and terrain groups for the 11 metrics stats.json covers; keys missing: {sc['stats_json_keys_missing']}):",
        f"   - point estimates: max |deviation| {sc['max_abs_point_estimate_deviation']:.3g};",
        f"   - with stats.json's own resampling weights (mode_switch_eval.bootstrap_weights): max |bound deviation|",
        f"     {sc['same_weights_as_stats_json_max_abs_bound_deviation']:.3g} (exact reproduction of the definition);",
        f"   - with tidy.py's seed ({sc['n_bounds_with_positive_halfwidth']} bounds with a positive half-width),",
        "     |bound deviation| / stats.json CI half-width: median "
        f"{mc['median']:.3f}, p95 {mc['p95']:.3f}, p99 {mc['p99']:.3f}, max {mc['max']:.3f};",
        "     Monte-Carlo reference (tidy.py seed vs a second tidy.py seed, same quantity): median "
        f"{ref['median']:.3f}, p95 {ref['p95']:.3f}, p99 {ref['p99']:.3f}, max {ref['max']:.3f}.",
        f"4. Noise floor: tidy.py's recomputation from replicate/ reproduces noise_floor.json: max |p90 deviation|",
        f"   {nfv['max_abs_dev_recomputed_vs_noise_floor_json_p90']:.3g}, max |per-cell difference deviation|",
        f"   {nfv['max_abs_dev_recomputed_vs_noise_floor_json_cell_diffs']:.3g} over {nfv['n_cell_diffs_compared']} (cell, metric) pairs.",
        f"5. Drift flag vs the warnings in logs/per_env_checks.log: {dv['files_compared']} files compared",
        f"   ({dv['files_in_log']} in the log), files with a different count: {dv['files_with_different_count']};",
        f"   flagged robots main {dv['robots_flagged_main']} + replicate {dv['robots_flagged_replicate']}, warned in the log",
        f"   {dv['robots_warned_in_log']}.",
        "",
        "## Same-policy null comparison (Ours vs Ours replicate, null_replicate_cells.csv)",
        "",
        "Counts of rows flagged by each criterion when the two runs differ only by GPU nondeterminism (cell scope, 40",
        "rows per metric; the noise floors are derived from this same replicate, so the robust counts are not an",
        "independent calibration):",
        "",
        "| metric | CI excludes 0 | perm p < 0.05 | robust (noise_floor.json) | robust_repl |",
        "|---|---:|---:|---:|---:|",
        *[f"| {n} | {c['ci_excl0']}/{c['n']} | {c['perm_p_lt_0.05']}/{c['n']} | "
          f"{'-' if c['robust (noise_floor.json rule)'] is None else c['robust (noise_floor.json rule)']} | {c['robust_repl']} |"
          for n, c in V["null_replicate"]["cell"].items()],
        "",
        "## Noise floors used (p90 of |run A - run B|)",
        "",
        "| metric | noise_floor.json (rule) | recomputed, cell | recomputed, cell x speed bin |",
        "|---|---:|---:|---:|",
        *[f"| {n} | {format(nf_json[m5], '.4g') if m5 else '-'} | "
          f"{nfv['replicate_recomputed_p90_cell'][n]:.4g} | {nfv['replicate_recomputed_p90_cell_speed'][n]:.4g} |"
          for n, *_, m5 in METRICS],
        "",
        "## Caveats carried into the tables",
        "",
        "* n = 1 training seed per policy; intervals cover terrain-instance (lane) and robot variability only.",
        "* random_rough levels are replicates (geometry ignores difficulty); discrete_obstacles heights are",
        "  0.045 / 0.075 / 0.110 / 0.150 m (levels.json).",
        "* Pyramid-type sections: landings narrower than wheel track + spawn spread; `dy_centre` is in per_traj for",
        "  stratification. stepping_stones: floor effect at d >= 0.75 (see floor_ceiling).",
        "* Switch success demands stepping in the rough window even where rolling is legitimate; `mode_success`",
        "  uses the cell's gate share instead. The gate is terrain-driven (computed from the height scan for every",
        "  policy, including RGGP, whose rewards do not use it).",
        "* Noise floors come from one replicate of Ours only (10 x 4 cells) and are cell-level (256 robots); speed-bin",
        "  and pooled rows have different sampling noise.",
        "* Pooled rows (difficulty, terrain, speed_bin, all) resample lanes within each cell, so their intervals and",
        "  p-values are conditional on the fixed set of terrains / difficulties (cells are not treated as random).",
        "  Pooled estimates weight cells by their robots with a value (e.g. window verdicts by traversals).",
        "",
    ]
    (HERE / "TIDY_README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
