"""Mode-appropriateness lens on the mode_switch_v2 rollouts (flat -> rough -> flat, forward_wide, 4 levels).

Questions (analysis only; no simulation; writes only into analysis/ and analysis/figures/):
  A. Expected mode per (terrain, difficulty) cell from the reward roughness-gate share in the rough window
     (cell_modes.csv from tidy.py) and gate-aware mode success vs the original switch success: where does
     switch success mislead (roll-expected cells)?
  B. Clearance sensitivity: Ours - baseline switch success with the pre-specified lift event (>= 0.06 s and
     >= 3 cm clearance) vs duration-only lift events; cells where the Ours vs Ours w/o RG ranking flips; event
     level duration / peak clearance of the rough-window airborne phases (from the npz rollouts).
  C. Selectivity: lifting on the flat segments (roll_flat_in / roll_flat_out, flat lift rate) vs lifting on the
     rough section, per policy and where (terrain, speed, position along the course).

Statistics (pre-specified rules): unit of resampling = lane (16 robots share one terrain instance); lanes are
resampled within each (terrain, difficulty) cell, 2000 bootstrap replicates, 95 % percentile intervals; paired
Ours - baseline differences on the same robots; lane-level sign-flip permutation p (exact 2^16 enumeration for
single cells, 10000 random assignments for pooled groups); Holm adjustment within each (scope, metric,
comparison) family; ROBUST = CI excludes 0 AND |diff| > p90 run-to-run noise of the SAME metric in
noise_floor.json (NA if noise_floor.json has no such metric). Supplementary (not the rule): robust_repl uses
the p90 of |Ours replicate - Ours| over the 40 cells recomputed here for every metric.
Floor / ceiling: a rate metric is uninformative in a cell where all three policies are < 5 % or > 95 %.

Inputs: analysis/per_traj.csv.gz, per_traj_replicate.csv.gz, cell_modes.csv, cells.csv, null_replicate_cells.csv
(tidy.py), ../noise_floor.json, ../<terrain>__forward_wide__<policy>.npz (event pass).
Outputs (analysis/): lens_mode_groups.csv, lens_mode_flips.csv, lens_mode_events.csv, lens_mode_profile.csv,
lens_mode_segments.csv, lens_mode_null.csv, lens_mode_validation.json, figures/lens_mode_*.png; tables on stdout
(saved as analysis/lens_mode.out by the command below).

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/lens_mode.py | tee logs/lipm_eval/mode_switch_v2/analysis/lens_mode.out
"""

from __future__ import annotations

import json
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
RUN = HERE.parent
FIG = HERE / "figures"
POLICIES = ("Ours", "RGGP", "LPGP")
LABEL = {"Ours": "Ours", "RGGP": "Ours w/o RG", "LPGP": "Ours w/o LP", "Ours_rep": "Ours (rerun)"}
BASELINES = ("RGGP", "LPGP")
LANES = 16
BOOTSTRAP = 2000
SEED = 20261001
NPERM_MC = 10000
EXACT_MAX_LANES = 16
WORKERS = 6
STEP_MIN_DURATION = 3  # policy steps (0.06 s), mode_switch_eval.STEP_MIN_DURATION
STEP_MIN_CLEARANCE = 0.03  # m, mode_switch_eval.STEP_MIN_CLEARANCE
BIN = 0.1  # m, course-x bins of the position profile
N_BINS = 117  # 0 .. 11.7 m
ROUGH_SECTION = (4.0, 7.6)
WINDOW_IDS = {"flat_in": 0, "rough": 1, "flat_out": 2}
SPEED_BINS = ("[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0]")  # per_traj speed_bin labels
# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface).
COLOR = {"Ours": "#2a78d6", "RGGP": "#eb6834", "LPGP": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# name -> (numerator column, denominator column or None (= finite values of the numerator), scale, better,
#          noise_floor.json metric or None, kind)
# Denominator None: mean over robots with a value (None / NaN excluded) = the cells.csv / stats.json definition.
METRICS = {
    "switch_success_pct": ("switch_success", None, 100, "higher", "switch_success_pct", "rate"),
    "mode_success_pct": ("mode_success", None, 100, "higher", None, "rate"),
    "switch_success_dur_pct": ("switch_success_dur", None, 100, "higher", None, "rate"),
    "completion_pct": ("completed", None, 100, "higher", "completion_pct", "rate"),
    "step_rough_pct": ("step_rough", None, 100, "higher", "step_rough_pct", "rate"),
    "step_rough_dur_pct": ("step_rough_dur", None, 100, "higher", None, "rate"),
    "anylift_rough_pct": ("anylift_rough", None, 100, "none", None, "rate"),
    "roll_flat_in_pct": ("roll_flat_in", None, 100, "higher", "roll_flat_in_pct", "rate"),
    "roll_flat_out_pct": ("roll_flat_out", None, 100, "higher", "roll_flat_out_pct", "rate"),
    "roll_flat_in_dur_pct": ("roll_flat_in_dur", None, 100, "higher", None, "rate"),
    "roll_flat_out_dur_pct": ("roll_flat_out_dur", None, 100, "higher", None, "rate"),
    "flat_lift_rate_pct": ("flat_lift_rate", None, 100, "lower", None, "rate"),
    "rough_lift_rate_pct": ("rough_lift_rate", None, 100, "none", None, "rate"),
    "rough_clear_share_pct": ("lift_rough", "dlift_rough", 100, "none", None, "event_ratio"),
    "fail_incomplete_pct": ("f_incomplete", None, 100, "lower", None, "rate"),
    "fail_flatlift_pct": ("f_flatlift", None, 100, "lower", None, "rate"),
    "fail_roughlift_pct": ("f_roughlift", None, 100, "lower", None, "rate"),
    # Completers only (all windows traversed): stepped on rough (both wheels, 3 cm) x flat windows clean (no 3 cm
    # lift event on flat_in / flat_out). step_clean = switch success among completers.
    "c_step_clean_pct": ("c_step_clean", None, 100, "higher", None, "rate"),
    "c_step_flatlift_pct": ("c_step_flatlift", None, 100, "none", None, "rate"),
    "c_nostep_clean_pct": ("c_nostep_clean", None, 100, "none", None, "rate"),
    "c_nostep_flatlift_pct": ("c_nostep_flatlift", None, 100, "none", None, "rate"),
}
NAMES = list(METRICS)
SCALE = np.array([METRICS[m][2] for m in NAMES], dtype=float)
# Metrics also in cells.csv (validation: identical point estimates and exact permutation p).
SHARED = ["switch_success_pct", "mode_success_pct", "switch_success_dur_pct", "completion_pct", "step_rough_pct",
          "roll_flat_in_pct", "roll_flat_out_pct", "flat_lift_rate_pct", "rough_lift_rate_pct"]


# --------------------------------------------------------------------------------------
# Per-robot table.


def to_float(col: pd.Series) -> np.ndarray:
    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, str):
            return {"True": 1.0, "False": 0.0}.get(v, np.nan) if v in ("True", "False") else float(v)
        return float(v)
    return np.array([conv(v) for v in col], dtype=float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("switch_success", "mode_success", "switch_success_dur", "completed", "step_rough", "step_rough_dur",
              "roll_flat_in", "roll_flat_out", "roll_flat_in_dur", "roll_flat_out_dur", "flat_lift_rate",
              "rough_lift_rate", "lift_rough", "dlift_rough", "lift_flat_in", "lift_flat_out"):
        df[c] = to_float(df[c])
    traversed_rough = np.isfinite(df.step_rough.to_numpy())  # same denominator as step_rough
    df["anylift_rough"] = np.where(traversed_rough, (df.lift_rough >= 1).astype(float), np.nan)
    # Mode-failure decomposition (exclusive, sums to 100 - mode_success): not completed; completed but a
    # 3 cm lift event on a flat window; completed, flat windows clean, but the rough window violates the mode.
    comp = df.completed.to_numpy() == 1
    flat = (df.lift_flat_in.to_numpy() + df.lift_flat_out.to_numpy()) > 0
    ms = df.mode_success.to_numpy() == 1
    df["f_incomplete"] = (~comp).astype(float)
    df["f_flatlift"] = (comp & flat).astype(float)
    df["f_roughlift"] = (comp & ~flat & ~ms).astype(float)
    stepped = (df.lift_rough_L.to_numpy() >= 1) & (df.lift_rough_R.to_numpy() >= 1)
    for name, sel in (("c_step_clean", stepped & ~flat), ("c_step_flatlift", stepped & flat),
                      ("c_nostep_clean", ~stepped & ~flat), ("c_nostep_flatlift", ~stepped & flat)):
        df[name] = np.where(comp, sel.astype(float), np.nan)
    df["cell"] = df.terrain + "|" + df.difficulty.map(lambda v: f"{v:g}")
    return df


class Aligned:
    """Per-policy (numerator, denominator) matrices aligned on the same robots (terrain, env)."""

    def __init__(self, df: pd.DataFrame, policies):
        parts = {p: df[df.policy == p].sort_values(["terrain", "env"]).reset_index(drop=True) for p in policies}
        ref = parts[policies[0]]
        for p in policies[1:]:
            for k in ("terrain", "env", "difficulty", "lane", "cmd_speed"):
                if not np.array_equal(parts[p][k].to_numpy(), ref[k].to_numpy()):
                    raise RuntimeError(f"{p} not aligned with {policies[0]} on {k}")
        self.meta = ref[["terrain", "difficulty", "lane", "env", "speed_bin", "expected_mode", "cell"]].copy()
        self.cluster = (ref.cell + "|lane=" + ref.lane.astype(str)).to_numpy()
        self.stratum = ref.cell.to_numpy()
        self.num, self.den = {}, {}
        for p, part in parts.items():
            num, den = [], []
            for m in NAMES:
                ncol, dcol = METRICS[m][0], METRICS[m][1]
                v = part[ncol].to_numpy(dtype=float)
                if dcol is None:
                    ok = np.isfinite(v)
                    num.append(np.where(ok, v, 0.0))
                    den.append(ok.astype(float))
                else:
                    dv = part[dcol].to_numpy(dtype=float)
                    num.append(np.nan_to_num(v))
                    den.append(np.nan_to_num(dv))
            self.num[p], self.den[p] = np.column_stack(num), np.column_stack(den)


def rng_for(key: str, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED + salt, zlib.crc32(key.encode())]))


def ratio(w, num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return SCALE * (w @ num) / (w @ den)


def signflip_p(nA, dA, nB, dB, rng):
    """Two-sided lane-level sign-flip p of est_A - est_B (labels swapped per lane)."""
    L = len(nA)
    if L <= EXACT_MAX_LANES:
        S = ((np.arange(2 ** L)[:, None] >> np.arange(L)) & 1).astype(float)
        exact = True
    else:
        S = rng.integers(0, 2, size=(NPERM_MC, L)).astype(float)
        exact = False
    sAn, sAd, sBn, sBd = S @ nA, S @ dA, S @ nB, S @ dB
    tAn, tAd, tBn, tBd = nA.sum(0), dA.sum(0), nB.sum(0), dB.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (sAn + tBn - sBn) / (sAd + tBd - sBd)
        b = (sBn + tAn - sAn) / (sBd + tAd - sAd)
        T = SCALE * (a - b)
        obs = SCALE * (tAn / tAd - tBn / tBd)
    finite = np.isfinite(T)
    extreme = (np.abs(T) >= np.abs(obs) - 1e-9 * np.maximum(1.0, np.abs(obs))) & finite
    n_ok = finite.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = extreme.sum(0) / n_ok if exact else (1 + extreme.sum(0)) / (1 + n_ok)
    return np.where(np.isfinite(obs) & (n_ok > 0), p, np.nan), ("exact(65536)" if exact else f"mc({NPERM_MC})")


def holm(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok):
        order = ok[np.argsort(p[ok], kind="stable")]
        m = len(order)
        out[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return out


def group_stats(A: Aligned, mask: np.ndarray, key: str, pairs) -> dict:
    idx = np.flatnonzero(mask)
    uniq, inv = np.unique(A.cluster[idx], return_inverse=True)
    strata = np.array([c.rsplit("|lane=", 1)[0] for c in uniq])
    sums = {}
    for p in A.num:
        n = np.zeros((len(uniq), len(NAMES)))
        d = np.zeros_like(n)
        np.add.at(n, inv, A.num[p][idx])
        np.add.at(d, inv, A.den[p][idx])
        sums[p] = (n, d)
    rng = rng_for(key)
    w = np.zeros((BOOTSTRAP, len(uniq)))
    for s in np.unique(strata):
        j = np.flatnonzero(strata == s)
        w[:, j] = rng.multinomial(len(j), np.full(len(j), 1 / len(j)), size=BOOTSTRAP)
    ones = np.ones((1, len(uniq)))
    out = {"n_robots": len(idx), "n_lanes": len(uniq), "n_cells": len(np.unique(strata)), "pol": {}, "diff": {}}
    boots = {}
    for p, (n, d) in sums.items():
        boots[p] = ratio(w, n, d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(boots[p], [2.5, 97.5], axis=0)
        out["pol"][p] = (ratio(ones, n, d)[0], lo, hi, d.sum(0))
    for a, b in pairs:
        db = boots[a] - boots[b]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(db, [2.5, 97.5], axis=0)
        pv, meth = signflip_p(*sums[a], *sums[b], rng_for(f"{key}|perm|{b}", 7))
        out["diff"][b] = (out["pol"][a][0] - out["pol"][b][0], lo, hi, pv, meth)
    return out


def table(A: Aligned, groups: list, pairs, pols) -> pd.DataFrame:
    """groups: list of (scope, label, mask). Long table, one row per group x metric."""
    rows = []
    for scope, label, mask in groups:
        g = group_stats(A, mask, f"{scope}|{label}", pairs)
        for j, m in enumerate(NAMES):
            r = {"scope": scope, "group": label, "metric": m, "better": METRICS[m][3],
                 "n_robots": g["n_robots"], "n_lanes": g["n_lanes"], "n_cells": g["n_cells"]}
            for p in pols:
                est, lo, hi, n = g["pol"][p]
                r.update({p: est[j], f"{p}_lo": lo[j], f"{p}_hi": hi[j], f"{p}_n": n[j]})
            for a, b in pairs:
                est, lo, hi, pv, meth = g["diff"][b]
                r.update({f"diff_{b}": est[j], f"diff_{b}_lo": lo[j], f"diff_{b}_hi": hi[j], f"perm_p_{b}": pv[j],
                          f"perm_method_{b}": meth})
            rows.append(r)
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame, pairs, pols, nf_p90: dict, repl_p90: dict) -> pd.DataFrame:
    df = df.copy()
    for a, b in pairs:
        df[f"perm_p_holm_{b}"] = np.nan
        for _, g in df.groupby(["scope", "metric"]):
            df.loc[g.index, f"perm_p_holm_{b}"] = holm(g[f"perm_p_{b}"].to_numpy(dtype=float))
        excl = (df[f"diff_{b}_lo"] > 0) | (df[f"diff_{b}_hi"] < 0)
        df[f"ci_excl0_{b}"] = excl
        nfm = df.metric.map(lambda m: METRICS[m][4])
        df["noise_metric"] = nfm
        df["noise_p90"] = nfm.map(lambda k: nf_p90.get(k, np.nan) if isinstance(k, str) else np.nan)
        df["noise_p90_repl"] = df.metric.map(repl_p90)
        big = df[f"diff_{b}"].abs() > df.noise_p90
        df[f"robust_{b}"] = np.where(df.noise_p90.notna(), (excl & big).astype(object), None)
        df[f"robust_repl_{b}"] = excl & (df[f"diff_{b}"].abs() > df.noise_p90_repl)
    rate = df.metric.map(lambda m: METRICS[m][5] == "rate")
    allv = df[list(pols)]
    df["floor_ceiling"] = np.where(~rate, "not_rate",
                                   np.where((allv < 5).all(axis=1), "floor",
                                            np.where((allv > 95).all(axis=1), "ceiling", "informative")))
    return df


# --------------------------------------------------------------------------------------
# Event pass over the npz rollouts.


def scan(path_str: str) -> dict:
    path = Path(path_str)
    terrain, _, policy = path.stem.split("__")
    z = np.load(path)
    valid, xs = z["valid"], z["course_x"]
    contact, clear = z["wheel_contact"], z["wheel_clearance"]
    course_end = float(z["course_end"])
    windows = {k: tuple(map(float, b)) for k, b in zip(WINDOW_IDS, z["windows"])}
    difficulty, lane = z["difficulty"], z["lane"]
    n_env = valid.shape[1]
    ev = {k: [] for k in ("env", "wheel", "win", "start_x", "dur", "peak")}
    prof = np.zeros((4, n_env, N_BINS), dtype=np.int32)  # alive steps, airborne steps, 3 cm starts, dur starts
    for i in range(n_env):
        v, x = valid[:, i], xs[:, i]
        reached = np.flatnonzero(v & (x >= course_end))
        end = int(reached[0]) if len(reached) else int(v.sum())
        window = np.zeros_like(v)
        window[:end] = v[:end]
        air = ~contact[:, i]
        c = clear[:, i].astype(float)
        bins = np.clip((x / BIN).astype(int), 0, N_BINS - 1)
        prof[0, i] = np.bincount(bins[window], minlength=N_BINS)
        prof[1, i] = np.bincount(bins[window & air.any(-1)], minlength=N_BINS)
        for w in range(2):
            mask = np.concatenate([[False], air[:, w] & window, [False]]).astype(np.int8)
            edges = np.diff(mask)
            s_, e_ = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            keep = (e_ - s_) >= STEP_MIN_DURATION
            for s, e in zip(s_[keep], e_[keep]):
                peak = c[s:e, w].max()
                sx = float(x[s])
                win = next((WINDOW_IDS[k] for k, (lo, hi) in windows.items() if lo <= sx < hi), -1)
                ev["env"].append(i)
                ev["wheel"].append(w)
                ev["win"].append(win)
                ev["start_x"].append(sx)
                ev["dur"].append(e - s)
                ev["peak"].append(peak)
                prof[3, i, bins[s]] += 1
                if peak >= STEP_MIN_CLEARANCE:
                    prof[2, i, bins[s]] += 1
    ev = {k: np.asarray(vv) for k, vv in ev.items()}
    ev["difficulty"] = difficulty[ev["env"]] if len(ev["env"]) else np.zeros(0)
    ev["lane"] = lane[ev["env"]] if len(ev["env"]) else np.zeros(0, int)
    return {"terrain": terrain, "policy": policy, "events": ev, "prof": prof,
            "difficulty": np.asarray(difficulty), "lane": np.asarray(lane), "dt": float(z["step_dt"])}


# --------------------------------------------------------------------------------------
# Printing helpers.


def ci(e, lo, hi, nd=1) -> str:
    if not np.isfinite(e):
        return "NA"
    return f"{e:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def verdict(r, b) -> str:
    rob = r.get(f"robust_{b}")
    excl = bool(r[f"ci_excl0_{b}"])
    d = r[f"diff_{b}"]
    sign = "+" if d > 0 else "-"
    if rob is None or (isinstance(rob, float) and np.isnan(rob)):
        tag = f"CI{sign}" if excl else "n.s."
        return tag + (" (repl-robust)" if bool(r[f"robust_repl_{b}"]) else "")
    return f"ROBUST{sign}" if rob else ("CI" + sign + ", < noise" if excl else "n.s.")


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


SHORT = {"discrete_obstacles": "discrete_obst", "hf_pyramid_slope": "hf_slope", "hf_pyramid_slope_inv": "hf_slope_inv",
         "pyramid_stair": "pyr_stair", "pyramid_stair_inv": "pyr_stair_inv", "random_rough": "random_rough",
         "random_spread": "random_spread", "random_stairs": "random_stairs", "stepping_stones": "stepping_st",
         "tilted_grid": "tilted_grid"}


# Label positions (data coordinates) of the reversal cells in Fig 2a (layout only; cells without an entry get an
# offset label).
LABEL_POS = {"discrete_obstacles|1": (62, -31), "random_stairs|0.5": (50, -24), "random_stairs|0.25": (30, -40),
             "hf_pyramid_slope_inv|1": (-40, -30), "pyramid_stair|0.5": (-12, -42)}


def cell_label(cell: str) -> str:
    t, d = cell.split("|")
    return f"{SHORT[t]} d={d}"


# --------------------------------------------------------------------------------------


def main() -> int:
    FIG.mkdir(exist_ok=True)
    V: dict = {}
    pairs = [("Ours", b) for b in BASELINES]
    nf = json.loads((RUN / "noise_floor.json").read_text())
    nf_p90 = {m: v["p90_abs_diff"] for m, v in nf["summary_over_cells"].items()}
    df = prepare(pd.read_csv(HERE / "per_traj.csv.gz"))
    rep = prepare(pd.read_csv(HERE / "per_traj_replicate.csv.gz"))
    modes = pd.read_csv(HERE / "cell_modes.csv")
    cells_csv = pd.read_csv(HERE / "cells.csv")
    null_csv = pd.read_csv(HERE / "null_replicate_cells.csv")
    A = Aligned(df, POLICIES)
    meta = A.meta
    cells = sorted(meta.cell.unique(), key=lambda c: (c.split("|")[0], float(c.split("|")[1])))
    mode_of = dict(zip(meta.cell, meta.expected_mode))

    # ---- replicate: supplementary noise floors for every metric (p90 |rep - Ours| over the 40 cells) ----
    rep2 = rep.copy()
    rep2["policy"] = "Ours_rep"
    rep2["expected_mode"] = rep2.cell.map(mode_of)
    AR = Aligned(pd.concat([df[df.policy == "Ours"], rep2]), ("Ours", "Ours_rep"))
    null_groups = [("cell", c, (AR.meta.cell == c).to_numpy()) for c in cells]
    null = table(AR, null_groups, [("Ours", "Ours_rep")], ("Ours", "Ours_rep"))
    repl_p90 = {m: float(np.nanpercentile(np.abs(null[null.metric == m].diff_Ours_rep), 90)) for m in NAMES}
    null = annotate(null.rename(columns=lambda c: c), [("Ours", "Ours_rep")], ("Ours", "Ours_rep"), nf_p90, repl_p90)
    null.to_csv(HERE / "lens_mode_null.csv", index=False, float_format="%.6g")
    V["repl_p90"] = repl_p90
    V["repl_p90_vs_noise_floor_json"] = {m: [repl_p90[m], nf_p90[METRICS[m][4]]] for m in NAMES if METRICS[m][4]}

    # ---- groups ----
    roll = (meta.expected_mode == "roll").to_numpy()
    groups = [("cell", c, (meta.cell == c).to_numpy()) for c in cells]
    groups += [("mode", "roll (8 cells)", roll), ("mode", "step (32 cells)", ~roll),
               ("mode", "all (40 cells)", np.ones(len(meta), bool))]
    groups += [("terrain", t, (meta.terrain == t).to_numpy()) for t in sorted(meta.terrain.unique())]
    groups += [("speed_step", f"step cells v={s}", (~roll) & (meta.speed_bin == s).to_numpy())
               for s in ("[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0]")]
    groups += [("speed_roll", f"roll cells v={s}", roll & (meta.speed_bin == s).to_numpy())
               for s in ("[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0]")]
    G = annotate(table(A, groups, pairs, POLICIES), pairs, POLICIES, nf_p90, repl_p90)
    G["expected_mode"] = G.group.map(lambda g: mode_of.get(g, ""))
    G.to_csv(HERE / "lens_mode_groups.csv", index=False, float_format="%.6g")

    # ---- validation against cells.csv (cell scope): point estimates and exact permutation p ----
    cc = cells_csv[cells_csv.scope == "cell"].copy()
    cc["cell"] = cc.terrain + "|" + cc.difficulty.map(lambda v: f"{float(v):g}")
    gc = G[G.scope == "cell"].merge(cc, left_on=["group", "metric"], right_on=["cell", "metric"], suffixes=("", "_t"))
    V["validation_vs_cells_csv"] = {
        "rows_compared": int(len(gc)), "metrics": SHARED,
        "max_abs_est_dev": float(max(np.nanmax(np.abs(gc[p] - gc[f"{p}_t"])) for p in POLICIES)),
        "max_abs_n_dev": float(max(np.nanmax(np.abs(gc[f"{p}_n"] - gc[f"{p}_n_t"])) for p in POLICIES)),
        "max_abs_exact_perm_p_dev": float(max(np.nanmax(np.abs(gc[f"perm_p_{b}"] - gc[f"perm_p_{b}_t"]))
                                              for b in BASELINES)),
        "max_abs_holm_p_dev": float(max(np.nanmax(np.abs(gc[f"perm_p_holm_{b}"] - gc[f"perm_p_holm_{b}_t"]))
                                        for b in BASELINES)),
        "median_ci_bound_dev_over_halfwidth": float(np.nanmedian(np.concatenate([
            np.abs(gc[f"diff_{b}_{s}"] - gc[f"diff_{b}_{s}_t"]) / ((gc[f"diff_{b}_hi_t"] - gc[f"diff_{b}_lo_t"]) / 2)
            for b in BASELINES for s in ("lo", "hi")]))),
    }
    ga = G[(G.scope == "mode") & (G.group == "all (40 cells)") & G.metric.isin(SHARED)].set_index("metric")
    ca = cells_csv[cells_csv.scope == "all"].set_index("metric")
    V["validation_vs_cells_csv"]["all_scope_max_abs_est_dev"] = float(max(
        np.nanmax(np.abs(ga.loc[SHARED, p] - ca.loc[SHARED, p])) for p in POLICIES))

    # ---- event pass ----
    paths = [str(RUN / f"{t}__forward_wide__{p}.npz") for t in sorted(meta.terrain.unique()) for p in POLICIES]
    with ProcessPoolExecutor(WORKERS) as ex:
        scans = list(ex.map(scan, paths))
    evs, profs = [], []
    for s in scans:
        e = pd.DataFrame(s["events"])
        e["terrain"], e["policy"] = s["terrain"], s["policy"]
        evs.append(e)
        profs.append(s)
    E = pd.concat(evs, ignore_index=True)
    E["cell"] = E.terrain + "|" + E.difficulty.map(lambda v: f"{float(v):g}")
    E["mode"] = E.cell.map(mode_of)
    E["clear3"] = E.peak >= STEP_MIN_CLEARANCE
    dt = profs[0]["dt"]
    # Validation: event counts per robot and window equal per_traj lift_* / dlift_* (3 cm / duration only).
    cnt = E[E.win >= 0].groupby(["terrain", "policy", "env", "win", "wheel"]).agg(d=("dur", "size"), c=("clear3", "sum"))
    cnt = cnt.unstack(["win", "wheel"], fill_value=0)
    mism = 0
    for (dcol, tag) in (("d", "dlift"), ("c", "lift")):
        for wn, wid in WINDOW_IDS.items():
            for wh, side in enumerate("LR"):
                got = cnt[(dcol, wid, wh)] if (dcol, wid, wh) in cnt.columns else pd.Series(dtype=float)
                ref = df.set_index(["terrain", "policy", "env"])[f"{tag}_{wn}_{side}"]
                got = got.reindex(ref.index, fill_value=0)
                mism += int((got.to_numpy() != ref.to_numpy()).sum())
    V["event_counts_vs_per_traj_mismatches"] = mism

    # Event summaries: rough-window airborne phases (>= 0.06 s) per policy x expected mode / flip cells.
    def ev_summary(sel: pd.DataFrame, label: str) -> list[dict]:
        out = []
        for p in POLICIES:
            x = sel[sel.policy == p]
            n = len(x)
            out.append({"group": label, "policy": p, "phases": n,
                        "median_dur_s": float(np.median(x.dur) * dt) if n else np.nan,
                        "median_peak_cm": float(np.median(x.peak) * 100) if n else np.nan,
                        "p75_peak_cm": float(np.percentile(x.peak, 75) * 100) if n else np.nan,
                        "share_ge3cm_pct": float(100 * x.clear3.mean()) if n else np.nan,
                        "share_peak0_pct": float(100 * (x.peak <= 0).mean()) if n else np.nan})
        return out

    rough_ev = E[E.win == 1]
    ev_rows = ev_summary(rough_ev[rough_ev["mode"] == "step"], "rough window, step cells")
    ev_rows += ev_summary(rough_ev[rough_ev["mode"] == "roll"], "rough window, roll cells")
    ev_rows += ev_summary(E[E.win.isin([0, 2])], "flat windows (all cells)")
    for t in sorted(meta.terrain.unique()):
        ev_rows += ev_summary(rough_ev[(rough_ev.terrain == t) & (rough_ev["mode"] == "step")], f"rough, step cells, {t}")
    EV = pd.DataFrame(ev_rows)
    EV.to_csv(HERE / "lens_mode_events.csv", index=False, float_format="%.6g")

    # Position profile (robot-weighted): 3 cm lift-event starts per robot per metre, duration-only starts,
    # share of alive steps with a wheel airborne; per policy x expected mode x commanded-speed bin ('ALL' = pooled).
    speed_of = dict(zip(zip(meta.terrain, meta.env), meta.speed_bin))
    env_mode, env_speed = {}, {}
    for s in profs:
        cellv = np.array([f"{s['terrain']}|{float(d):g}" for d in s["difficulty"]])
        env_mode[(s["terrain"], s["policy"])] = np.array([mode_of[c] for c in cellv])
        env_speed[(s["terrain"], s["policy"])] = np.array([speed_of[(s["terrain"], e)] for e in range(len(cellv))])
    prof_rows = []
    for p in POLICIES:
        for mname in ("step", "roll"):
            for sb in ("ALL", *SPEED_BINS):
                def stack(j):
                    return np.concatenate([
                        s["prof"][j][(env_mode[(s["terrain"], p)] == mname)
                                     & ((env_speed[(s["terrain"], p)] == sb) if sb != "ALL" else True)]
                        for s in profs if s["policy"] == p])
                alive, air, e3, ed = stack(0), stack(1), stack(2), stack(3)
                present = alive > 0
                n_rob = present.sum(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    share = np.where(present, air / np.maximum(alive, 1), 0).sum(0) / n_rob
                    r3 = e3.sum(0) / n_rob / BIN
                    rd = ed.sum(0) / n_rob / BIN
                for k in range(N_BINS):
                    prof_rows.append({"policy": p, "mode": mname, "speed_bin": sb, "x_lo": round(k * BIN, 3),
                                      "x_hi": round((k + 1) * BIN, 3), "n_robots": int(n_rob[k]),
                                      "airborne_share_pct": 100 * share[k], "lift3_per_m": r3[k], "liftdur_per_m": rd[k]})
    P = pd.DataFrame(prof_rows)
    P.to_csv(HERE / "lens_mode_profile.csv", index=False, float_format="%.6g")

    # Segment rates per policy (step cells): 3 cm lift-event starts per robot-metre in course segments.
    seg_def = [("flat_in judged [1.8,3.2)", 1.8, 3.2), ("approach [3.2,4.0)", 3.2, 4.0),
               ("rough section [4.0,7.6)", 4.0, 7.6), ("exit transition [7.6,9.0)", 7.6, 9.0),
               ("flat_out judged [9.0,11.6)", 9.0, 11.6)]
    seg_rows = []
    for mname, sb in [("step", "ALL"), ("roll", "ALL")] + [("step", x) for x in SPEED_BINS]:
        for lab, lo, hi in seg_def:
            k0, k1 = int(round(lo / BIN)), int(round(hi / BIN))
            r = {"mode": mname, "speed_bin": sb, "segment": lab}
            for p in POLICIES:
                q = P[(P.policy == p) & (P["mode"] == mname) & (P.speed_bin == sb)].iloc[k0:k1]
                r[p] = float(q.lift3_per_m.mean())
                r[f"{p}_dur"] = float(q.liftdur_per_m.mean())
            seg_rows.append(r)
    SEG = pd.DataFrame(seg_rows)
    SEG.to_csv(HERE / "lens_mode_segments.csv", index=False, float_format="%.6g")

    # ---- flips ----
    cellG = G[G.scope == "cell"].set_index(["group", "metric"])
    flip_rows = []
    for c in cells:
        r3, rd = cellG.loc[(c, "switch_success_pct")], cellG.loc[(c, "switch_success_dur_pct")]
        rm = cellG.loc[(c, "mode_success_pct")]
        row = {"cell": c, "expected_mode": mode_of[c]}
        for b in BASELINES:
            def sgn(r):
                return int(np.sign(r[f"diff_{b}"])) if r[f"ci_excl0_{b}"] else 0
            s3, sd, sm = sgn(r3), sgn(rd), sgn(rm)
            row.update({f"d3_{b}": r3[f"diff_{b}"], f"d3_{b}_lo": r3[f"diff_{b}_lo"], f"d3_{b}_hi": r3[f"diff_{b}_hi"],
                        f"v3_{b}": verdict(r3, b), f"p3holm_{b}": r3[f"perm_p_holm_{b}"],
                        f"dd_{b}": rd[f"diff_{b}"], f"dd_{b}_lo": rd[f"diff_{b}_lo"], f"dd_{b}_hi": rd[f"diff_{b}_hi"],
                        f"vd_{b}": verdict(rd, b), f"pdholm_{b}": rd[f"perm_p_holm_{b}"],
                        f"dm_{b}": rm[f"diff_{b}"], f"dm_{b}_lo": rm[f"diff_{b}_lo"], f"dm_{b}_hi": rm[f"diff_{b}_hi"],
                        f"vm_{b}": verdict(rm, b), f"pmholm_{b}": rm[f"perm_p_holm_{b}"],
                        f"clear_flip_{b}": ("reversal" if s3 * sd == -1 else "lost" if s3 != 0 and sd == 0
                                            else "gained" if s3 == 0 and sd != 0 else "same"),
                        f"mode_flip_{b}": ("reversal" if s3 * sm == -1 else "lost" if s3 != 0 and sm == 0
                                           else "gained" if s3 == 0 and sm != 0 else "same")})
        for p in POLICIES:
            row[f"sw3_{p}"], row[f"swd_{p}"], row[f"mode_{p}"] = r3[p], rd[p], rm[p]
        row["fc_sw3"], row["fc_swd"], row["fc_mode"] = r3.floor_ceiling, rd.floor_ceiling, rm.floor_ceiling
        flip_rows.append(row)
    F = pd.DataFrame(flip_rows)
    F.to_csv(HERE / "lens_mode_flips.csv", index=False, float_format="%.6g")

    # ======================================================================================
    # Tables (stdout).
    pd.set_option("display.width", 250)
    g = lambda scope, grp, m: G[(G.scope == scope) & (G.group == grp) & (G.metric == m)].iloc[0]  # noqa: E731

    print("\n## T1 expected mode per cell (pooled rough-window gate share, cell_modes.csv)")
    mt = modes.pivot(index="terrain", columns="difficulty", values="gate_share_rough_pooled")
    mm = modes.pivot(index="terrain", columns="difficulty", values="expected_mode")
    t1 = pd.DataFrame({f"d={d:g}": [f"{mt.loc[t, d]:.2f} {mm.loc[t, d]}" for t in mt.index] for d in mt.columns},
                      index=mt.index).reset_index()
    print(md(t1))
    V["gate_share_policy_spread_max"] = float((modes[["gate_share_rough_Ours", "gate_share_rough_RGGP",
                                                       "gate_share_rough_LPGP"]].max(axis=1)
                                               - modes[["gate_share_rough_Ours", "gate_share_rough_RGGP",
                                                        "gate_share_rough_LPGP"]].min(axis=1)).max())
    V["gate_share_flat_max"] = float(modes.gate_share_flat_pooled.max())
    print("max per-policy spread of the rough gate share in a cell:", round(V["gate_share_policy_spread_max"], 4),
          "; max flat-window gate share:", round(V["gate_share_flat_max"], 6))

    print("\n## T2 pooled by expected mode: estimates (95% CI) and Ours - baseline")
    rows = []
    for grp in ("roll (8 cells)", "step (32 cells)", "all (40 cells)"):
        for m in ("switch_success_pct", "mode_success_pct", "switch_success_dur_pct", "completion_pct"):
            r = g("mode", grp, m)
            rows.append({"group": grp, "metric": m, **{LABEL[p]: ci(r[p], r[f"{p}_lo"], r[f"{p}_hi"]) for p in POLICIES},
                         **{f"Ours-{LABEL[b]}": ci(r[f"diff_{b}"], r[f"diff_{b}_lo"], r[f"diff_{b}_hi"]) + f" p={r[f'perm_p_{b}']:.4f} {verdict(r, b)}"
                            for b in BASELINES}})
    print(md(pd.DataFrame(rows)))

    print("\n## T3 roll-expected cells: switch success vs gate-aware mode success (and why mode success fails)")
    rows = []
    for c in [c for c in cells if mode_of[c] == "roll"]:
        r = {"cell": cell_label(c)}
        for m, short in (("switch_success_pct", "switch"), ("mode_success_pct", "mode")):
            q = g("cell", c, m)
            r[f"{short} O/R/L"] = f"{q.Ours:.1f}/{q.RGGP:.1f}/{q.LPGP:.1f}"
            for b in BASELINES:
                r[f"{short} O-{b[:1]}"] = (f"{q[f'diff_{b}']:+.1f} [{q[f'diff_{b}_lo']:+.1f},{q[f'diff_{b}_hi']:+.1f}] "
                                           f"{verdict(q, b)} pHolm={q[f'perm_p_holm_{b}']:.3f}")
        q = g("cell", c, "mode_success_pct")
        r["fc(sw/mode)"] = f"{g('cell', c, 'switch_success_pct').floor_ceiling}/{q.floor_ceiling}"
        for m, short in (("fail_incomplete_pct", "incompl"), ("fail_flatlift_pct", "flatlift"),
                         ("fail_roughlift_pct", "roughlift"), ("step_rough_pct", "step_rough")):
            qq = g("cell", c, m)
            r[f"{short} O/R/L"] = f"{qq.Ours:.1f}/{qq.RGGP:.1f}/{qq.LPGP:.1f}"
        rows.append(r)
    print(md(pd.DataFrame(rows)))
    print("mode-failure decomposition, roll cells pooled:")
    rows = []
    for m in ("fail_incomplete_pct", "fail_flatlift_pct", "fail_roughlift_pct", "anylift_rough_pct", "step_rough_pct",
              "rough_lift_rate_pct"):
        r = g("mode", "roll (8 cells)", m)
        rows.append({"metric": m, **{LABEL[p]: ci(r[p], r[f"{p}_lo"], r[f"{p}_hi"]) for p in POLICIES},
                     **{f"Ours-{LABEL[b]}": ci(r[f"diff_{b}"], r[f"diff_{b}_lo"], r[f"diff_{b}_hi"]) + " " + verdict(r, b)
                        for b in BASELINES}})
    print(md(pd.DataFrame(rows)))

    print("\n## T4 verdict counts over the informative cells (switch success 3 cm / duration only / mode success)")
    rows = []
    for m in ("switch_success_pct", "switch_success_dur_pct", "mode_success_pct"):
        q_all = G[(G.scope == "cell") & (G.metric == m)]
        q = q_all[q_all.floor_ceiling == "informative"]  # floor / ceiling cells are uninformative (rule)
        for b in BASELINES:
            inf = q
            rob = q[f"robust_{b}"]
            has_rule = rob.notna().all()
            rows.append({"metric": m, "vs": LABEL[b], "informative cells": len(inf),
                         "CI excl 0 (Ours+/Ours-)": f"{int(((q[f'ci_excl0_{b}']) & (q[f'diff_{b}'] > 0)).sum())}/{int(((q[f'ci_excl0_{b}']) & (q[f'diff_{b}'] < 0)).sum())}",
                         "robust rule (Ours+/Ours-)": (f"{int((rob.astype(bool) & (q[f'diff_{b}'] > 0)).sum())}/{int((rob.astype(bool) & (q[f'diff_{b}'] < 0)).sum())}"
                                                       if has_rule else "NA (no noise_floor.json metric)"),
                         "robust_repl (Ours+/Ours-)": f"{int((q[f'robust_repl_{b}'] & (q[f'diff_{b}'] > 0)).sum())}/{int((q[f'robust_repl_{b}'] & (q[f'diff_{b}'] < 0)).sum())}",
                         "Holm p<0.05 (Ours+/Ours-)": f"{int(((q[f'perm_p_holm_{b}'] < 0.05) & (q[f'diff_{b}'] > 0)).sum())}/{int(((q[f'perm_p_holm_{b}'] < 0.05) & (q[f'diff_{b}'] < 0)).sum())}"})
    print(md(pd.DataFrame(rows)))
    print("same-policy null (Ours vs rerun, 40 cells): rows flagged")
    rows = []
    for m in ("switch_success_pct", "switch_success_dur_pct", "mode_success_pct", "step_rough_dur_pct",
              "roll_flat_in_pct", "roll_flat_out_pct", "flat_lift_rate_pct", "rough_clear_share_pct"):
        q = null[null.metric == m]
        rob = q["robust_Ours_rep"]
        rows.append({"metric": m, "CI excl 0": int(q.ci_excl0_Ours_rep.sum()),
                     "robust rule": int(rob.astype(bool).sum()) if rob.notna().all() else "NA",
                     "robust_repl": int(q.robust_repl_Ours_rep.sum()),
                     "Holm p<0.05": int((q.perm_p_holm_Ours_rep < 0.05).sum()), "p90 repl": round(repl_p90[m], 3)})
    print(md(pd.DataFrame(rows)))
    V["null_counts_tidy_switch_success_robust"] = int(null_csv[(null_csv.scope == "cell") & (null_csv.metric == "switch_success_pct")].robust_Ours_rep.astype(str).eq("True").sum())

    print("\n## T5 clearance sensitivity per cell: Ours - Ours w/o RG switch success, 3 cm vs duration only")
    rows = []
    for _, r in F.iterrows():
        rows.append({"cell": cell_label(r.cell), "mode": r.expected_mode,
                     "3cm O/R/L": f"{r.sw3_Ours:.1f}/{r.sw3_RGGP:.1f}/{r.sw3_LPGP:.1f}",
                     "dur O/R/L": f"{r.swd_Ours:.1f}/{r.swd_RGGP:.1f}/{r.swd_LPGP:.1f}",
                     "O-R 3cm": f"{r.d3_RGGP:+.1f} [{r.d3_RGGP_lo:+.1f},{r.d3_RGGP_hi:+.1f}] {r.v3_RGGP}",
                     "O-R dur": f"{r.dd_RGGP:+.1f} [{r.dd_RGGP_lo:+.1f},{r.dd_RGGP_hi:+.1f}] {r.vd_RGGP} pHolm={r.pdholm_RGGP:.3f}",
                     "flip O-R": r.clear_flip_RGGP, "flip O-L": r.clear_flip_LPGP})
    print(md(pd.DataFrame(rows)))
    for b in BASELINES:
        for mname in ("step", "roll"):
            sub = F[F.expected_mode == mname]
            V[f"clear_flip_counts_{b}_{mname}"] = sub[f"clear_flip_{b}"].value_counts().to_dict()
            V[f"mode_flip_counts_{b}_{mname}"] = sub[f"mode_flip_{b}"].value_counts().to_dict()
    print("CI-verdict change 3 cm -> duration only (reversal: both CIs exclude 0 with opposite signs; lost: 3 cm CI "
          "excludes 0, duration-only does not; gained: the reverse):")
    for b in BASELINES:
        for mname in ("step", "roll"):
            print(f"  Ours vs {LABEL[b]}, {mname} cells:", V[f"clear_flip_counts_{b}_{mname}"],
                  "| reversal cells:", [cell_label(c) for c in F[(F.expected_mode == mname) & (F[f"clear_flip_{b}"] == "reversal")].cell])
    print("CI-verdict change switch success -> mode success (roll cells only can differ):")
    for b in BASELINES:
        print(f"  Ours vs {LABEL[b]}, roll cells:", V[f"mode_flip_counts_{b}_roll"],
              "| step cells:", V[f"mode_flip_counts_{b}_step"])

    print("\n## T6 window verdicts, 3 cm vs duration only (pooled by expected mode)")
    rows = []
    for grp in ("step (32 cells)", "roll (8 cells)", "all (40 cells)"):
        for m in ("step_rough_pct", "step_rough_dur_pct", "anylift_rough_pct", "roll_flat_in_pct", "roll_flat_in_dur_pct",
                  "roll_flat_out_pct", "roll_flat_out_dur_pct", "rough_clear_share_pct"):
            r = g("mode", grp, m)
            rows.append({"group": grp, "metric": m, **{LABEL[p]: ci(r[p], r[f"{p}_lo"], r[f"{p}_hi"]) for p in POLICIES},
                         **{f"Ours-{LABEL[b]}": ci(r[f"diff_{b}"], r[f"diff_{b}_lo"], r[f"diff_{b}_hi"]) + " " + verdict(r, b)
                            for b in BASELINES}})
    print(md(pd.DataFrame(rows)))

    print("\n## T7 airborne phases >= 0.06 s (event level, npz): duration and peak clearance")
    print(md(EV[~EV.group.str.startswith("rough, step cells,")].round(2)))
    print(md(EV[EV.group.str.startswith("rough, step cells,")].round(2)))

    print("\n## T8 selectivity: flat vs rough lifting (pooled by expected mode, terrain, speed)")
    rows = []
    for scope, grp in ([("mode", x) for x in ("step (32 cells)", "roll (8 cells)", "all (40 cells)")]
                       + [("speed_step", f"step cells v={s}") for s in ("[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0]")]):
        for m in ("roll_flat_in_pct", "roll_flat_out_pct", "flat_lift_rate_pct", "rough_lift_rate_pct", "step_rough_pct"):
            r = g(scope, grp, m)
            rows.append({"group": grp, "metric": m, **{LABEL[p]: ci(r[p], r[f"{p}_lo"], r[f"{p}_hi"]) for p in POLICIES},
                         **{f"Ours-{LABEL[b]}": ci(r[f"diff_{b}"], r[f"diff_{b}_lo"], r[f"diff_{b}_hi"]) + " " + verdict(r, b)
                            for b in BASELINES}, "fc": r.floor_ceiling})
    print(md(pd.DataFrame(rows)))
    print("completers, step cells: stepped on rough (both wheels, 3 cm) x flat windows clean (no 3 cm lift event)")
    rows = []
    for scope, grp in [("mode", "step (32 cells)")] + [("speed_step", f"step cells v={x}") for x in SPEED_BINS]:
        for m in ("c_step_clean_pct", "c_step_flatlift_pct", "c_nostep_clean_pct", "c_nostep_flatlift_pct"):
            r = g(scope, grp, m)
            rows.append({"group": grp, "metric": m, **{LABEL[p]: ci(r[p], r[f"{p}_lo"], r[f"{p}_hi"]) for p in POLICIES},
                         **{f"Ours-{LABEL[b]}": ci(r[f"diff_{b}"], r[f"diff_{b}_lo"], r[f"diff_{b}_hi"]) + " " + verdict(r, b)
                            for b in BASELINES}})
    print(md(pd.DataFrame(rows)))
    print("per terrain (all difficulties): roll_flat_out / flat_lift_rate / step_rough, O/R/L and verdicts")
    rows = []
    for t in sorted(meta.terrain.unique()):
        r = {"terrain": t}
        for m, short in (("roll_flat_out_pct", "roll_flat_out"), ("flat_lift_rate_pct", "flat_lift"),
                         ("step_rough_pct", "step_rough")):
            q = g("terrain", t, m)
            r[f"{short} O/R/L"] = f"{q.Ours:.1f}/{q.RGGP:.1f}/{q.LPGP:.1f}"
            if m != "step_rough_pct":
                r[f"{short} O-R"] = f"{q.diff_RGGP:+.1f} {verdict(q, 'RGGP')}"
                r[f"{short} O-L"] = f"{q.diff_LPGP:+.1f} {verdict(q, 'LPGP')}"
        rows.append(r)
    print(md(pd.DataFrame(rows)))
    print("cell-level verdict counts (informative cells only) for the flat metrics:")
    rows = []
    for m in ("roll_flat_in_pct", "roll_flat_out_pct", "flat_lift_rate_pct"):
        q_all = G[(G.scope == "cell") & (G.metric == m)]
        q = q_all[q_all.floor_ceiling == "informative"]
        for b in BASELINES:
            rob = q[f"robust_{b}"]
            has = rob.notna().all()
            good = (q[f"diff_{b}"] > 0) if METRICS[m][3] == "higher" else (q[f"diff_{b}"] < 0)
            rb = rob.astype(bool) if has else q[f"robust_repl_{b}"]
            rows.append({"metric": m, "vs": LABEL[b], "informative": int((q.floor_ceiling == "informative").sum()),
                         "rule": "noise_floor.json" if has else "robust_repl (supplementary)",
                         "robust Ours better": int((rb & good).sum()), "robust Ours worse": int((rb & ~good).sum()),
                         "Holm<0.05 better/worse": f"{int(((q[f'perm_p_holm_{b}'] < 0.05) & good).sum())}/{int(((q[f'perm_p_holm_{b}'] < 0.05) & ~good).sum())}",
                         "worse cells": ", ".join(cell_label(c) for c in q[rb & ~good].group)})
    print(md(pd.DataFrame(rows)))
    print("3 cm lift-event starts per robot-metre by course segment (robot-weighted profile; *_dur = duration only)")
    print(md(SEG.round(3)))

    # ======================================================================================
    # Figures.
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK})
    terr = sorted(meta.terrain.unique())
    diffs = sorted(modes.difficulty.unique())

    # Fig 1: gate-share map + roll-cell differences (switch vs mode success).
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    mat = mt.loc[terr, diffs].to_numpy()
    im = ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"]), vmin=0, vmax=1, aspect="auto")
    for i, t in enumerate(terr):
        for j, dd in enumerate(diffs):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}\n{mm.loc[t, dd]}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if val > 0.6 else INK, fontweight="bold" if mm.loc[t, dd] == "roll" else "normal")
    ax.set_xticks(range(len(diffs)), [f"{d:g}" for d in diffs])
    ax.set_yticks(range(len(terr)), [SHORT[t] for t in terr])
    ax.set_xlabel("difficulty d")
    ax.set_title("(a) rough-window gate share (λ>0.65)\nexpected mode: step if ≥ 0.5", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)
    del im  # values are annotated in every cell; no colour bar
    roll_cells = [c for c in cells if mode_of[c] == "roll"]
    for k, b in enumerate(BASELINES):
        ax = fig.add_subplot(gs[k + 1])
        style(ax)
        ys = np.arange(len(roll_cells))[::-1]
        for yy, c in zip(ys, roll_cells):
            for off, m, mk, fill in ((0.17, "switch_success_pct", "s", False), (-0.17, "mode_success_pct", "o", True)):
                q = g("cell", c, m)
                ax.errorbar(q[f"diff_{b}"], yy + off, xerr=[[q[f"diff_{b}"] - q[f"diff_{b}_lo"]],
                                                              [q[f"diff_{b}_hi"] - q[f"diff_{b}"]]],
                            fmt=mk, ms=6, color=COLOR[b], mfc=COLOR[b] if fill else SURFACE, mew=1.5, elinewidth=1.5,
                            capsize=0, zorder=3)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(ys, [cell_label(c) for c in roll_cells])
        ax.set_xlabel(f"Ours − {LABEL[b]} (pp)\n> 0 = Ours higher; 95% lane-bootstrap CI")
        ax.set_title(f"({'bc'[k]}) roll-expected cells: Ours − {LABEL[b]}", loc="left")
        ax.plot([], [], "s", color=COLOR[b], mfc=SURFACE, mew=1.5, label="switch success (demands stepping)")
        ax.plot([], [], "o", color=COLOR[b], label="gate-aware mode success (demands rolling)")
        ax.legend(loc="lower left", fontsize=6.5, frameon=False, bbox_to_anchor=(0.0, 1.07), borderaxespad=0)
        ax.set_ylim(-0.7, len(roll_cells) - 0.3)
    fig.savefig(FIG / "lens_mode_fig1_mode_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: clearance sensitivity.
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": 0.32})
    ax = axs[0]
    style(ax)
    for _, r in F.iterrows():
        mk = "o" if r.expected_mode == "step" else "D"
        flip = r.clear_flip_RGGP
        col = COLOR["RGGP"] if flip == "reversal" else INK2
        ax.errorbar(r.d3_RGGP, r.dd_RGGP, xerr=[[r.d3_RGGP - r.d3_RGGP_lo], [r.d3_RGGP_hi - r.d3_RGGP]],
                    yerr=[[r.dd_RGGP - r.dd_RGGP_lo], [r.dd_RGGP_hi - r.dd_RGGP]], fmt=mk, ms=5,
                    color=col, mfc=col if flip == "reversal" else SURFACE, elinewidth=0.7, alpha=0.9, zorder=3)
        if flip == "reversal":
            tx, ty = LABEL_POS.get(r.cell, (r.d3_RGGP + 3, r.dd_RGGP - 5))
            ax.annotate(cell_label(r.cell), (r.d3_RGGP, r.dd_RGGP), xytext=(tx, ty), textcoords="data",
                        fontsize=6.3, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    lim = (-45, 90)
    ax.plot(lim, lim, color=GRID, lw=1, ls="--", zorder=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(-45, 55)
    ax.set_xlabel("Ours − Ours w/o RG, switch success with 3 cm lift events (pp)")
    ax.set_ylabel("same, duration-only lift events (pp)")
    ax.set_title("(a) per cell (40): ranking under the two lift definitions", loc="left")
    ax.plot([], [], "o", color=COLOR["RGGP"], label="reversal (both CIs exclude 0, opposite signs)")
    ax.plot([], [], "o", color=INK2, mfc=SURFACE, label="other step-expected cell")
    ax.plot([], [], "D", color=INK2, mfc=SURFACE, label="roll-expected cell")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    step_rough_ev = rough_ev[rough_ev["mode"] == "step"]
    for k, (col, xl, scale, xmax) in enumerate((("peak", "peak clearance of the airborne phase (cm)", 100, 20),
                                                 ("dur", "airborne-phase duration (s)", dt, 1.0))):
        ax = axs[k + 1]
        style(ax)
        for p in POLICIES:
            v = np.sort(step_rough_ev[step_rough_ev.policy == p][col].to_numpy(dtype=float) * scale)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLOR[p], lw=2, label=f"{LABEL[p]} (n={len(v)})")
        if col == "peak":
            ax.axvline(3, color=INK, lw=1, ls="--")
            ax.text(3.3, 0.04, "3 cm criterion", fontsize=6.5, color=INK)
        ax.set_xlim(-0.5 if col == "peak" else 0, xmax)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(xl)
        ax.set_ylabel("cumulative share of phases")
        ax.set_title(f"({'bc'[k]}) rough-window airborne phases ≥ 0.06 s,\nstep-expected cells", loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.savefig(FIG / "lens_mode_fig2_clearance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: selectivity along the course (step cells by commanded speed; roll cells) + per-cell scatter.
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.22)

    def profile_panel(ax, mname, sb, title, ylim):
        style(ax)
        for lo, hi, lab in ((1.8, 3.2, "flat_in"), (4.2, 7.4, "rough"), (9.0, 11.6, "flat_out")):
            ax.axvspan(lo, hi, color="#f0efec", zorder=0)
            ax.text((lo + hi) / 2, 1.0, lab + " (judged)", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6.3, color=INK2)
        for xx in ROUGH_SECTION:
            ax.axvline(xx, color=INK2, lw=0.8, ls=":")
        for p in POLICIES:
            q = P[(P.policy == p) & (P["mode"] == mname) & (P.speed_bin == sb) & (P.x_lo >= X0) & (P.x_hi <= 11.6)]
            ax.plot((q.x_lo + q.x_hi) / 2, q.lift3_per_m, color=COLOR[p], lw=1.8, label=LABEL[p])
        ax.set_xlim(X0, 11.6)
        ax.set_ylim(*ylim)
        ax.set_xlabel("course x (m)")
        ax.set_title(title, loc="left", pad=13)

    X0 = 1.5  # m; robots start at x = 0.70-1.31 m and settle after the reset (lift events there are reset transients)
    ymax = 1.05 * P[(P["mode"] == "step") & (P.x_lo >= X0) & (P.x_hi <= 11.6)].lift3_per_m.max()
    for k, sb in enumerate(SPEED_BINS):
        ax = fig.add_subplot(gs[0, k])
        profile_panel(ax, "step", sb, f"({'abc'[k]}) step-expected cells, v_cmd ∈ {sb} m/s", (0, ymax))
        if k == 0:
            ax.set_ylabel("3 cm lift events started\nper robot per metre")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax = fig.add_subplot(gs[1, :2])
    profile_panel(ax, "roll", "ALL", "(d) roll-expected cells (8), all speeds", (0, None))
    ax.set_ylabel("3 cm lift events started\nper robot per metre")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    qo = G[(G.scope == "cell") & (G.metric == "roll_flat_out_pct")].set_index("group")
    qs = G[(G.scope == "cell") & (G.metric == "step_rough_pct")].set_index("group")
    for p in POLICIES:
        for mname, mk in (("step", "o"), ("roll", "D")):
            cs = [c for c in cells if mode_of[c] == mname]
            ax.scatter(100 - qo.loc[cs, p], qs.loc[cs, p], s=24, marker=mk, color=COLOR[p],
                       edgecolor=SURFACE, linewidth=0.8, zorder=3, alpha=0.9)
    for p in POLICIES:
        ax.scatter([], [], s=24, color=COLOR[p], label=LABEL[p])
    ax.scatter([], [], s=24, marker="o", color=INK2, label="step-expected cell")
    ax.scatter([], [], s=24, marker="D", color=INK2, label="roll-expected cell")
    ax.set_xlabel("lifting on flat_out: 100 − roll_flat_out (% of completers)")
    ax.set_ylabel("step_rough (% of rough traversers)")
    ax.set_title("(e) per cell: rough stepping vs flat_out lifting", loc="left", pad=13)
    ax.legend(fontsize=6.3, frameon=False, loc="center right")
    ax.set_xlim(-2, 72)
    ax.set_ylim(-3, 103)
    fig.text(0.1, 0.035, f"Dotted lines: rough section 4.0–7.6 m; shaded: judged windows. Robot-weighted mean over the robots "
             f"alive in each 0.1 m bin. x < {X0} m (post-reset transient) not shown.", fontsize=7, color=INK2)
    fig.savefig(FIG / "lens_mode_fig3_selectivity.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    (HERE / "lens_mode_validation.json").write_text(json.dumps(V, indent=1, default=str))
    print("\nvalidation:", json.dumps({k: V[k] for k in ("validation_vs_cells_csv", "event_counts_vs_per_traj_mismatches")},
                                      default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
