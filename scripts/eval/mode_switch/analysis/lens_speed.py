"""Speed lens on mode_switch_v2: does the Ours advantage change with the commanded speed?

Reads only analysis/per_traj.csv.gz, analysis/per_traj_replicate.csv.gz, analysis/cells.csv and
../noise_floor.json (no simulation). Writes analysis/lens_speed/*.csv, analysis/lens_speed/lens_speed.json
and analysis/figures/speed_*.png. Deterministic (fixed seeds, exact enumeration where cells.csv uses it).

Statistics reuse tidy.py (same bootstrap, sign-flip permutation, Holm and robust rule as cells.csv):
  * speed-bin groups (bins [0.5,1.0) / [1.0,1.5) / [1.5,2.0] m/s) pooled over all 40 cells, per difficulty
    (pooled over terrains), per terrain (pooled over difficulties) and leave-one-terrain-out (LOTO):
    per-policy estimates, paired Ours - B differences with 95 % lane-bootstrap CI (lanes resampled within
    each (terrain, difficulty) cell), lane sign-flip permutation p (10000 random assignments for pooled
    groups), Holm within (scope, metric, comparison), robust = CI excludes 0 AND |diff| > noise_floor.json p90.
    The speed_bin rows use cells.csv's group keys and seeds and are checked to reproduce cells.csv exactly.
  * bin contrasts: [Delta(bin k) - Delta(bin j)], all bins evaluated on the SAME lane resamples (joint bootstrap).
  * trend: slope of the per-robot paired difference D = y_Ours - y_B on the commanded speed, with lane fixed
    effects (within-lane OLS; the command is randomised per robot), in units per 1 m/s. Only robots where both
    policies have a value enter (window verdicts: both traversed the window). 95 % lane-bootstrap CI (lanes
    resampled within cells) and a within-lane speed-permutation p (commands shuffled among the robots of a
    lane, 5000 permutations; H0: D does not depend on the command). noise_floor.json has no slope metric, so
    the pre-specified robust rule cannot be applied; as a supplementary reference the same slope is computed
    for Ours vs its own replicate run (pooled, and p90 over the 40 cells).
  * same-policy null: every group statistic is also computed for Ours vs Ours replicate.
  * exposure: lift events per second of window time in the flat_out / rough windows (explains why
    'no lift in flat_out' is harder at low speed: the window takes longer to cross).

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/lens_speed.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tidy as T  # noqa: E402  (bootstrap / permutation / Holm / table code of cells.csv)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

RUN = HERE.parent
OUT = HERE / "lens_speed"
FIG = HERE / "figures"

FOCUS = ["switch_success_pct", "completion_pct", "cot", "step_rough_pct", "roll_flat_out_pct"]
SUPP = ["mode_success_pct", "switch_success_dur_pct", "failure_pct", "roll_flat_in_pct", "flat_lift_rate_pct",
        "time_ratio", "peak_tilt_deg"]
TREND = FOCUS + SUPP
J = {name: j for j, name in enumerate(T.NAMES)}
BETTER = {m[0]: m[4] for m in T.METRICS}
SB = list(T.SPEED_LABELS)
SB_MID = np.array([0.75, 1.25, 1.75])
FINE_EDGES = np.round(np.arange(0.5, 2.0001, 0.25), 4)
BASE = ("RGGP", "LPGP")
LABEL = {"Ours": "Ours", "RGGP": "Ours w/o RG", "LPGP": "Ours w/o LP", "Ours_rep": "Ours (replicate)"}
SLOPE_SEED = 20261001
CONTRAST_SEED = 20261002
PERM_SEED = 20261003
NPERM_SLOPE = 5000
DT = 0.02  # policy step (s); step_dt of every rollout (mode_switch_eval STEP_MIN_DURATION = 3 steps = 0.06 s)

# Palette (dataviz reference palette, categorical slots 1-3; diverging blue <-> red, gray midpoint).
COL = {"Ours": "#2a78d6", "RGGP": "#eb6834", "LPGP": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIVERGE = LinearSegmentedColormap.from_list(
    "adv", ["#e34948", "#f3a6a3", "#f0efec", "#86b6ef", "#1c5cab"])  # low = Ours worse, high = Ours better
NICE = {"switch_success_pct": "switch success (%)", "completion_pct": "completion (%)", "cot": "CoT",
        "step_rough_pct": "step_rough (%)", "roll_flat_out_pct": "roll_flat_out (%)",
        "mode_success_pct": "gate-aware mode success (%)", "switch_success_dur_pct": "duration-only switch (%)",
        "failure_pct": "failure (%)", "roll_flat_in_pct": "roll_flat_in (%)",
        "flat_lift_rate_pct": "flat lift rate (%)", "time_ratio": "time_ratio", "peak_tilt_deg": "peak tilt (deg)"}


# --------------------------------------------------------------------------------------------------
# Helpers


def adv(metric: str, diff):
    """Direction-adjusted Ours advantage (positive = Ours better); NaN sign for metrics without direction."""
    b = BETTER[metric]
    return diff if b == "higher" else -diff if b == "lower" else np.nan * diff


def floors_from_cells(cells: pd.DataFrame):
    nf = json.loads((RUN / "noise_floor.json").read_text())["summary_over_cells"]
    nf_p90 = {k: v["p90_abs_diff"] for k, v in nf.items()}
    first = cells.drop_duplicates("metric").set_index("metric")
    return nf_p90, first.noise_p90_repl_cell.to_dict(), first.noise_p90_repl_speed.to_dict()


def run_groups(data: T.Data, groups: list, with_perm=True) -> dict:
    """groups: (scope, terrain, difficulty, speed_bin, idx); key = tidy.gkey (cells.csv seeds)."""
    res = {}
    for scope, t, dd, sb, idx in groups:
        key = T.gkey(scope, t, dd, sb)
        res[key] = T.group_stats(data, idx, key, with_perm=with_perm)
    return res


def table(data: T.Data, groups: list, pols, floors, with_perm=True) -> pd.DataFrame:
    res = run_groups(data, groups, with_perm)
    if not with_perm:
        for r in res.values():
            for e in r["diff"].values():
                e["p"], e["method"] = np.full(len(T.NAMES), np.nan), "none"
    return T.build_table(res, [g[:4] + (None,) for g in groups], pols, {}, floors)


def cluster_moments(data, idx, y, v):
    """Per lane: n, within-lane centred cross moments of (v, y) for finite y."""
    ok = np.isfinite(y)
    idx, y, v = idx[ok], y[ok], v[ok]
    uniq, inv = np.unique(data.cluster[idx], return_inverse=True)
    n = np.bincount(inv).astype(float)
    sv, sy = np.bincount(inv, v), np.bincount(inv, y)
    cvv = np.bincount(inv, v * v) - sv ** 2 / n
    cvy = np.bincount(inv, v * y) - sv * sy / n
    return uniq, inv, y, v, cvv, cvy


def lane_fe_slope(data, idx, speed, a, b, j, key, boot=True, perm=True, nperm=NPERM_SLOPE) -> dict:
    """Within-lane OLS slope of D = y_a - y_b (or y_a if b is None) on the commanded speed."""
    y = T.SCALE[j] * (data.V[a][idx, j] - (data.V[b][idx, j] if b else 0.0))
    uniq, inv, yy, vv, cvv, cvy = cluster_moments(data, idx, y, speed[idx])
    out = {"slope": float(cvy.sum() / cvv.sum()), "n_robots": int(len(yy)), "n_lanes": int(len(uniq)),
           "lo": np.nan, "hi": np.nan, "p_perm": np.nan}
    if boot:
        w = T.lane_weights(uniq // T.M.LANES, T.group_seed(SLOPE_SEED, key))
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (w @ cvy) / (w @ cvv)
        out["lo"], out["hi"] = (float(x) for x in np.nanpercentile(bs, [2.5, 97.5]))
    if perm:
        order = np.argsort(inv, kind="stable")
        inv_s, y_s, v_s = inv[order], yy[order], vv[order]
        ybar = np.bincount(inv_s, y_s) / np.bincount(inv_s)
        yc = y_s - ybar[inv_s]
        den = cvv.sum()
        obs = float(yc @ v_s / den)
        rng = T.group_seed(PERM_SEED, key)
        extreme, done = 0, 0
        while done < nperm:
            k = min(500, nperm - done)
            perm_idx = np.argsort(inv_s[:, None] + rng.random((len(inv_s), k)), axis=0)
            stat = (yc[:, None] * v_s[perm_idx]).sum(0) / den
            extreme += int(np.sum(np.abs(stat) >= abs(obs) - 1e-12 * max(1.0, abs(obs))))
            done += k
        out["p_perm"] = (1 + extreme) / (1 + nperm)
    return out


def bin_contrasts(data, idx, bins, pols, key) -> list[dict]:
    """Delta(bin k) - Delta(bin j) for a = pols[0] vs each other policy, all bins on the same lane resamples."""
    uniq, inv = np.unique(data.cluster[idx], return_inverse=True)
    w = T.lane_weights(uniq // T.M.LANES, T.group_seed(CONTRAST_SEED, key))
    ones = np.ones((1, len(uniq)))
    est, boot = {}, {}
    for k in range(len(SB)):
        m = bins[idx] == k
        for p in pols:
            v = data.V[p][idx[m]]
            num = np.zeros((len(uniq), v.shape[1]))
            den = np.zeros_like(num)
            np.add.at(num, inv[m], np.where(np.isfinite(v), v, 0.0))
            np.add.at(den, inv[m], np.isfinite(v).astype(float))
            est[k, p], boot[k, p] = T.ratio(ones, num, den)[0], T.ratio(w, num, den)
    rows = []
    a = pols[0]
    for b in pols[1:]:
        for hi_bin, lo_bin in ((1, 0), (2, 0), (2, 1)):
            d_est = (est[hi_bin, a] - est[hi_bin, b]) - (est[lo_bin, a] - est[lo_bin, b])
            d_boot = (boot[hi_bin, a] - boot[hi_bin, b]) - (boot[lo_bin, a] - boot[lo_bin, b])
            lo, hi = T.pct(d_boot)
            for name in TREND:
                j = J[name]
                rows.append({"comparison": b, "contrast": f"{SB[hi_bin]} - {SB[lo_bin]}", "metric": name,
                             "estimate": d_est[j], "lo": lo[j], "hi": hi[j],
                             "ci_excl0": bool(lo[j] > 0 or hi[j] < 0)})
    return rows


def ratio_ci(num, den, clusters, key):
    """Ratio of sums with a 95 % lane-bootstrap CI (lanes resampled within cells)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    n, d = np.bincount(inv, num), np.bincount(inv, den)
    w = T.lane_weights(uniq // T.M.LANES, T.group_seed(CONTRAST_SEED + 11, key))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (w @ n) / (w @ d)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(n.sum() / d.sum()), float(lo), float(hi)


# --------------------------------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()
    warnings.simplefilter("ignore", RuntimeWarning)
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "per_traj.csv.gz", low_memory=False)
    rep = pd.read_csv(HERE / "per_traj_replicate.csv.gz", low_memory=False)
    cells = pd.read_csv(HERE / "cells.csv")
    floors = floors_from_cells(cells)
    nf_p90 = floors[0]
    data = T.Data(df)
    null = T.Data(pd.concat([df[df.ckpt == "Ours"], rep.assign(ckpt="Ours_rep")], ignore_index=True),
                  policies=("Ours", "Ours_rep"))
    ref = df[df.ckpt == "Ours"].sort_values(["rough", "env"]).reset_index(drop=True)
    assert np.array_equal(ref.env.to_numpy(), data.meta.env.to_numpy())
    assert np.array_equal(null.meta.env.to_numpy(), data.meta.env.to_numpy())
    speed = ref.cmd_speed.to_numpy(float)
    sbin = np.searchsorted(SB, data.meta.speed_bin.to_numpy())
    assert np.array_equal(np.array(SB)[sbin], data.meta.speed_bin.to_numpy())
    terr = data.meta.rough.to_numpy()
    diff = data.meta.difficulty.to_numpy(float)
    terrains, diffs = data.terrains, data.diffs
    J_OUT = {"inputs": ["analysis/per_traj.csv.gz", "analysis/per_traj_replicate.csv.gz", "analysis/cells.csv",
                        "noise_floor.json"], "noise_floor_p90": nf_p90}

    # ---------------- groups ----------------
    g_speed = [("speed_bin", "ALL", "ALL", s, np.flatnonzero(sbin == k)) for k, s in enumerate(SB)]
    g_dspeed = [("difficulty_speed", "ALL", d, s, np.flatnonzero((diff == d) & (sbin == k)))
                for d in diffs for k, s in enumerate(SB)]
    g_tspeed = [("terrain_speed", t, "ALL", s, np.flatnonzero((terr == t) & (sbin == k)))
                for t in terrains for k, s in enumerate(SB)]
    g_loto = [("loto_speed", f"-{t}", "ALL", s, np.flatnonzero((terr != t) & (sbin == k)))
              for t in terrains for k, s in enumerate(SB)]
    fine = np.clip(np.searchsorted(FINE_EDGES, speed, side="right") - 1, 0, len(FINE_EDGES) - 2)
    g_fine = [("fine_speed", "ALL", "ALL", f"[{FINE_EDGES[k]:.2f},{FINE_EDGES[k + 1]:.2f})",
               np.flatnonzero(fine == k)) for k in range(len(FINE_EDGES) - 1)]
    g_tfine = [("terrain_speed", t, "ALL", s, idx) for (_, t, _, s, idx) in g_tspeed]  # same as g_tspeed

    tabs = {
        "speed_bin": table(data, g_speed, T.POLICIES, floors),
        "difficulty_speed": table(data, g_dspeed, T.POLICIES, floors),
        "terrain_speed": table(data, g_tspeed, T.POLICIES, floors),
        "loto_speed": table(data, g_loto, T.POLICIES, floors),
        "fine_speed": table(data, g_fine, T.POLICIES, floors, with_perm=False),
    }
    nulls = {
        "speed_bin": table(null, g_speed, ("Ours", "Ours_rep"), floors),
        "difficulty_speed": table(null, g_dspeed, ("Ours", "Ours_rep"), floors),
        "terrain_speed": table(null, g_tspeed, ("Ours", "Ours_rep"), floors),
    }
    del g_tfine
    for name, tab in tabs.items():
        tab.to_csv(OUT / f"{name}.csv", index=False)
    for name, tab in nulls.items():
        tab.to_csv(OUT / f"null_{name}.csv", index=False)
    print(f"group tables done ({time.time() - t0:.0f} s)")

    # ---------------- check: speed_bin rows reproduce cells.csv ----------------
    mine = tabs["speed_bin"].set_index(["speed_bin", "metric"])
    theirs = cells[cells.scope == "speed_bin"].set_index(["speed_bin", "metric"]).loc[mine.index]
    cols = [c for c in mine.columns if c.startswith(("Ours", "RGGP", "LPGP", "diff_", "perm_p_")) and
            mine[c].dtype.kind == "f"]
    a_, b_ = mine[cols].to_numpy(float), theirs[cols].to_numpy(float)
    dev = float(np.nanmax(np.abs(a_ - b_)))
    rel = float(np.nanmax(np.abs(a_ - b_) / np.maximum(np.abs(b_), 1e-12)))
    same_nan = bool(np.array_equal(np.isnan(a_), np.isnan(b_)))
    J_OUT["check_speed_bin_vs_cells_csv"] = {"columns": len(cols), "max_abs_dev": dev, "max_rel_dev": rel,
                                             "same_nan_pattern": same_nan,
                                             "note": "cells.csv is written with float_format %.8g"}
    print(f"speed_bin vs cells.csv: max |dev| {dev:.3g}, max rel dev {rel:.3g} (cells.csv has 8 significant digits)")
    assert rel < 1e-7 and same_nan, (dev, rel)

    # ---------------- bin contrasts (joint bootstrap) ----------------
    allidx = np.arange(len(speed))
    contr = [dict(scope="all", group="ALL", **r) for r in bin_contrasts(data, allidx, sbin, T.POLICIES, "all")]
    for d in diffs:
        idx = np.flatnonzero(diff == d)
        contr += [dict(scope="difficulty", group=f"{d:g}", **r)
                  for r in bin_contrasts(data, idx, sbin, T.POLICIES, f"d={d:g}")]
    for t in terrains:
        idx = np.flatnonzero(terr != t)
        contr += [dict(scope="loto", group=f"-{t}", **r)
                  for r in bin_contrasts(data, idx, sbin, T.POLICIES, f"loto-{t}")]
    contr += [dict(scope="null_all", group="ALL", **r)
              for r in bin_contrasts(null, allidx, sbin, ("Ours", "Ours_rep"), "null_all")]
    contr = pd.DataFrame(contr)
    contr.to_csv(OUT / "bin_contrasts.csv", index=False)
    print(f"contrasts done ({time.time() - t0:.0f} s)")

    # ---------------- trend: lane-FE slope of the paired difference on the commanded speed ----------------
    srows = []

    def add_slope(scope, group, idx, dat, a, b, name, **kw):
        key = f"slope|{scope}|{group}|{a}-{b}|{name}"
        r = lane_fe_slope(dat, idx, speed, a, b, J[name], key, **kw)
        srows.append({"scope": scope, "group": group, "comparison": b if b else f"{a} level", "metric": name, **r})

    for name in TREND:
        for b in BASE:
            add_slope("all", "ALL", allidx, data, "Ours", b, name)
            for d in diffs:
                add_slope("difficulty", f"{d:g}", np.flatnonzero(diff == d), data, "Ours", b, name)
            for t in terrains:
                add_slope("terrain", t, np.flatnonzero(terr == t), data, "Ours", b, name)
                add_slope("loto", f"-{t}", np.flatnonzero(terr != t), data, "Ours", b, name, perm=False)
        for p in T.POLICIES:
            add_slope("all_level", "ALL", allidx, data, p, None, name, perm=False)
        add_slope("null_all", "ALL", allidx, null, "Ours", "Ours_rep", name)
        for t in terrains:
            for d in diffs:
                add_slope("null_cell", f"{t}|{d:g}", np.flatnonzero((terr == t) & (diff == d)), null, "Ours",
                          "Ours_rep", name, boot=False, perm=False)
    slopes = pd.DataFrame(srows)
    # Holm over the family of groups within (scope, metric, comparison) for permutation p-values.
    slopes["p_perm_holm"] = np.nan
    slopes["holm_m"] = 0
    for _, sub in slopes.groupby(["scope", "metric", "comparison"]):
        pv = sub.p_perm.to_numpy(float)
        slopes.loc[sub.index, "p_perm_holm"] = T.holm(pv)
        slopes.loc[sub.index, "holm_m"] = int(np.isfinite(pv).sum())
    slopes["ci_excl0"] = (slopes.lo > 0) | (slopes.hi < 0)
    nullcell = slopes[slopes.scope == "null_cell"]
    p90_slope = nullcell.groupby("metric").slope.apply(lambda s: float(np.nanpercentile(np.abs(s), 90)))
    slopes["noise_p90_slope_repl_cell"] = slopes.metric.map(p90_slope)
    slopes["robust_repl_slope"] = slopes.ci_excl0 & (slopes.slope.abs() > slopes.noise_p90_slope_repl_cell)
    slopes.to_csv(OUT / "slopes.csv", index=False)
    print(f"slopes done ({time.time() - t0:.0f} s)")

    # ---------------- exposure: lift events per second of window time ----------------
    erows = []
    for p in T.POLICIES:
        part = df[df.ckpt == p].sort_values(["rough", "env"]).reset_index(drop=True)
        for w in ("flat_in", "rough", "flat_out"):
            for k, s in enumerate(SB):
                m = sbin == k
                cl = data.cluster[m]
                ev = part[f"lift_{w}"].to_numpy(float)[m]
                secs = part[f"n_alive_{w}"].to_numpy(float)[m] * DT
                rate = ratio_ci(ev, secs, cl, f"exp|{p}|{w}|{s}")
                per_robot = ratio_ci(ev, (secs > 0).astype(float), cl, f"expr|{p}|{w}|{s}")
                secs_mean = ratio_ci(secs, (secs > 0).astype(float), cl, f"exps|{p}|{w}|{s}")
                gate = part[f"n_gate_{w}"].to_numpy(float)[m].sum() / part[f"n_alive_{w}"].to_numpy(float)[m].sum()
                erows.append({"policy": p, "window": w, "speed_bin": s, "events_per_s": rate[0],
                              "events_per_s_lo": rate[1], "events_per_s_hi": rate[2],
                              "events_per_robot": per_robot[0], "events_per_robot_lo": per_robot[1],
                              "events_per_robot_hi": per_robot[2], "window_time_s": secs_mean[0],
                              "gate_share": gate})
    expo = pd.DataFrame(erows)
    expo.to_csv(OUT / "exposure.csv", index=False)

    # Low-speed detail of the flat_out verdict (sub-bins of the commanded speed), per policy.
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0001]
    frows = []
    for p in T.POLICIES:
        part = df[df.ckpt == p].sort_values(["rough", "env"]).reset_index(drop=True)
        rf = data.V[p][:, J["roll_flat_out_pct"]]
        ev = part.lift_flat_out.to_numpy(float)
        secs = part.n_alive_flat_out.to_numpy(float) * DT
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            m = (speed >= lo_) & (speed < hi_)
            ok = m & np.isfinite(rf)
            est = ratio_ci(np.where(ok, rf, 0.0) * 100, ok.astype(float), data.cluster,
                           f"fo|{p}|{lo_:g}")
            frows.append({"policy": p, "v_lo": lo_, "v_hi": min(hi_, 2.0), "n_robots": int(m.sum()),
                          "n_completed": int(ok.sum()), "roll_flat_out_pct": est[0], "lo": est[1], "hi": est[2],
                          "flat_out_events_per_s": ev[m].sum() / secs[m].sum(),
                          "gate_share_flat_out": part.n_gate_flat_out.to_numpy(float)[m].sum() /
                          part.n_alive_flat_out.to_numpy(float)[m].sum()})
    pd.DataFrame(frows).to_csv(OUT / "flat_out_low_speed.csv", index=False)

    # ---------------- summaries for the report ----------------
    summ = {}
    sp = tabs["speed_bin"]
    for b in BASE:
        for name in TREND:
            sub = sp[sp.metric == name].set_index("speed_bin").loc[SB]
            a_ = adv(name, sub[f"diff_{b}"].to_numpy(float))
            best = SB[int(np.nanargmax(a_))] if np.isfinite(a_).any() else None
            # Same-best-bin count across LOTO sets.
            lt = tabs["loto_speed"]
            lt = lt[lt.metric == name]
            best_loto = []
            for t in terrains:
                s2 = lt[lt.terrain == f"-{t}"].set_index("speed_bin").loc[SB]
                a2 = adv(name, s2[f"diff_{b}"].to_numpy(float))
                best_loto.append(SB[int(np.nanargmax(a2))] if np.isfinite(a2).any() else None)
            ts = tabs["terrain_speed"]
            ts = ts[ts.metric == name]
            best_terr = {}
            for t in terrains:
                s3 = ts[ts.terrain == t].set_index("speed_bin").loc[SB]
                a3 = adv(name, s3[f"diff_{b}"].to_numpy(float))
                best_terr[t] = SB[int(np.nanargmax(a3))] if np.isfinite(a3).any() else None
            sl = slopes[(slopes.metric == name) & (slopes.comparison == b)]
            pooled = sl[sl.scope == "all"].iloc[0]
            ter = sl[sl.scope == "terrain"]
            loto = sl[sl.scope == "loto"]
            sign = np.sign(pooled.slope)
            summ[f"{b}|{name}"] = {
                "best_bin": best, "best_bin_loto_counts": {s: best_loto.count(s) for s in SB},
                "best_bin_terrain_counts": {s: list(best_terr.values()).count(s) for s in SB},
                "best_bin_per_terrain": best_terr,
                "slope": pooled.slope, "slope_ci": [pooled.lo, pooled.hi], "slope_p_perm": pooled.p_perm,
                "terrain_slopes_same_sign": int(np.sum(np.sign(ter.slope) == sign)),
                "terrain_slopes_ci_same_side": int(np.sum(ter.ci_excl0 & (np.sign(ter.slope) == sign))),
                "terrain_slopes_ci_opposite": int(np.sum(ter.ci_excl0 & (np.sign(ter.slope) == -sign))),
                "loto_slope_min": float(loto.slope.min()), "loto_slope_max": float(loto.slope.max()),
                "loto_ci_excl0_same_side": int(np.sum(loto.ci_excl0 & (np.sign(loto.slope) == sign))),
                "loto_argmin_terrain": loto.loc[loto.slope.abs().idxmin(), "group"],
            }
    J_OUT["summary"] = summ

    # Chance expectations: robust flags in the same-policy null at each scope.
    chance = {}
    for scope, tab in nulls.items():
        for name in FOCUS + SUPP:
            sub = tab[tab.metric == name]
            r = sub["robust_Ours_rep"]
            chance[f"{scope}|{name}"] = {
                "groups": int(len(sub)), "ci_excl0": int(sub.ci_excl0_Ours_rep.sum()),
                "perm_p_lt_0.05": int((sub.perm_p_Ours_rep < 0.05).sum()),
                "holm_lt_0.05": int((sub.perm_p_holm_Ours_rep < 0.05).sum()),
                "robust": None if r.isna().all() else int(r.fillna(False).astype(bool).sum()),
                "max_abs_diff": float(np.nanmax(np.abs(sub.diff_Ours_rep))),
            }
    J_OUT["null_replicate_counts"] = chance
    rob = {}
    for scope in ("speed_bin", "difficulty_speed", "terrain_speed"):
        tab = tabs[scope]
        for name in FOCUS + SUPP:
            sub = tab[tab.metric == name]
            info = sub.floor_ceiling.isin(["informative", "not_rate"])  # floor / ceiling rows are uninformative
            for b in BASE:
                oc = sub[f"outcome_{b}"]
                holm_ok = sub[f"perm_p_holm_{b}"] < 0.05
                rob[f"{scope}|{name}|{b}"] = {
                    "groups": int(len(sub)), "informative": int(info.sum()),
                    "Ours better": int(((oc == "Ours better") & info).sum()),
                    "Ours worse": int(((oc == "Ours worse") & info).sum()),
                    "Ours better & holm<0.05": int(((oc == "Ours better") & info & holm_ok).sum()),
                    "Ours worse & holm<0.05": int(((oc == "Ours worse") & info & holm_ok).sum()),
                    "robust but floor/ceiling (excluded)": int(oc.isin(["Ours better", "Ours worse"]).sum()
                                                               - oc[info].isin(["Ours better", "Ours worse"]).sum()),
                    "holm_lt_0.05": int(holm_ok.sum()),
                }
    J_OUT["robust_counts"] = rob
    (OUT / "lens_speed.json").write_text(json.dumps(J_OUT, indent=1, default=float))
    print(f"tables written ({time.time() - t0:.0f} s)")

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
                         "axes.titlecolor": INK, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fine_tab = tabs["fine_speed"]
    fine_mid = 0.5 * (FINE_EDGES[:-1] + FINE_EDGES[1:])
    fine_lab = [g[3] for g in g_fine]

    def fine_series(name, col):
        sub = fine_tab[fine_tab.metric == name].set_index("speed_bin").loc[fine_lab]
        return sub[col].to_numpy(float), sub[f"{col}_lo"].to_numpy(float), sub[f"{col}_hi"].to_numpy(float)

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Figure 1: per-policy levels vs commanded speed.
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), constrained_layout=True)
    for ax, name in zip(axes, FOCUS):
        for p in T.POLICIES:
            est, lo, hi = fine_series(name, p)
            ax.fill_between(fine_mid, lo, hi, color=COL[p], alpha=0.18, lw=0)
            ax.plot(fine_mid, est, color=COL[p], lw=2, marker="o", ms=4, label=LABEL[p])
        ax.set_title(NICE[name])
        ax.set_xlabel("commanded v_x (m/s), 0.25 m/s bins")
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
        style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Per-policy level vs commanded speed (all 40 cells pooled; band = 95 % lane-bootstrap CI)",
                 color=INK, fontsize=11)
    fig.savefig(FIG / "speed_levels.png", dpi=150)
    plt.close(fig)

    # Figure 2: paired differences vs speed, with per-terrain 3-bin lines and the noise floor band.
    tsp = tabs["terrain_speed"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.8), constrained_layout=True)
    for r, b in enumerate(BASE):
        for c, name in enumerate(FOCUS):
            ax = axes[r, c]
            nf = nf_p90.get(name, np.nan)
            ax.axhspan(-nf, nf, color="#e1e0d9", alpha=0.7, lw=0, label="+-noise p90 (cell)")
            ax.axhline(0, color="#898781", lw=0.8)
            for t in terrains:
                sub = tsp[(tsp.metric == name) & (tsp.terrain == t)].set_index("speed_bin").loc[SB]
                ax.plot(SB_MID, sub[f"diff_{b}"].to_numpy(float), color="#b8b7b0", lw=0.9, zorder=1,
                        label="single terrain (3 bins)" if t == terrains[0] else None)
            est, lo, hi = fine_series(name, f"diff_{b}")
            ax.fill_between(fine_mid, lo, hi, color=COL[b], alpha=0.22, lw=0, zorder=2)
            ax.plot(fine_mid, est, color=COL[b], lw=2, marker="o", ms=4, zorder=3,
                    label="all cells, 0.25 m/s bins")
            sp_sub = sp[sp.metric == name].set_index("speed_bin").loc[SB]
            e3 = sp_sub[f"diff_{b}"].to_numpy(float)
            ax.errorbar(SB_MID, e3, yerr=[e3 - sp_sub[f"diff_{b}_lo"].to_numpy(float),
                                          sp_sub[f"diff_{b}_hi"].to_numpy(float) - e3],
                        fmt="s", color=INK, ms=5, capsize=3, lw=1, zorder=4, label="all cells, 3 speed bins")
            sl = slopes[(slopes.scope == "all") & (slopes.metric == name) & (slopes.comparison == b)].iloc[0]
            fmtv = "{:+.3f}" if name == "cot" else "{:+.1f}"
            slope_txt = ("lane-FE slope " + fmtv + " [" + fmtv + ", " + fmtv + "] per m/s").format(
                sl.slope, sl.lo, sl.hi)
            ax.set_xticks([0.5, 1.0, 1.5, 2.0])
            style(ax)
            if r == 1:
                ax.set_xlabel("commanded v_x (m/s)")
            if c == 0:
                ax.set_ylabel(f"Ours - {LABEL[b]}", color=INK)
            ax.set_title(NICE[name] + (" (neg. = Ours better)" if BETTER[name] == "lower" else "") + "\n" +
                         slope_txt, fontsize=9.5)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Paired difference Ours - ablation vs commanded speed (gray lines: each terrain, pooled over "
                 "difficulty)", color=INK, fontsize=11)
    fig.savefig(FIG / "speed_diffs.png", dpi=150)
    plt.close(fig)

    # Figures 3/4: heatmaps of the direction-adjusted advantage (terrain x speed, difficulty x speed).
    def heat(tab, rows_key, row_vals, row_names, pooled_tab, fname, title):
        metrics = ["switch_success_pct", "completion_pct", "cot", "step_rough_pct", "roll_flat_out_pct"]
        nrow = len(row_vals) + 1
        fig, axes = plt.subplots(2, 5, figsize=(17, 0.42 * nrow * 2 + 2.2), constrained_layout=True)
        for c, name in enumerate(metrics):
            mats, robs = [], []
            for b in BASE:
                m = np.full((nrow, 3), np.nan)
                rb = np.zeros((nrow, 3), bool)
                for i, rv in enumerate(row_vals):
                    sub = tab[(tab.metric == name) & (tab[rows_key] == rv)].set_index("speed_bin").loc[SB]
                    m[i] = sub[f"diff_{b}"].to_numpy(float)
                    rb[i] = sub[f"robust_{b}"].fillna(False).astype(bool).to_numpy()
                sub = pooled_tab[pooled_tab.metric == name].set_index("speed_bin").loc[SB]
                m[-1] = sub[f"diff_{b}"].to_numpy(float)
                rb[-1] = sub[f"robust_{b}"].fillna(False).astype(bool).to_numpy()
                mats.append(m)
                robs.append(rb)
            vmax = float(np.nanmax(np.abs(np.concatenate(mats))))
            for r, b in enumerate(BASE):
                ax = axes[r, c]
                a_ = adv(name, mats[r])
                ax.imshow(a_, cmap=DIVERGE, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
                for i in range(nrow):
                    for k in range(3):
                        v = mats[r][i, k]
                        txt = ("{:+.2f}" if name == "cot" else "{:+.1f}").format(v) + ("*" if robs[r][i, k] else "")
                        ax.text(k, i, txt, ha="center", va="center", fontsize=7.5,
                                color="#ffffff" if abs(a_[i, k]) > 0.6 * vmax else INK,
                                fontweight="bold" if i == nrow - 1 else "normal")
                ax.axhline(nrow - 1.5, color="#fcfcfb", lw=3)
                ax.set_xticks(range(3), SB, fontsize=8)
                ax.set_yticks(range(nrow), list(row_names) + ["pooled"] if c == 0 else [""] * nrow, fontsize=8)
                ax.tick_params(length=0)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_title(f"{NICE[name]}\nOurs - {LABEL[b]}", fontsize=9)
        fig.suptitle(title, color=INK, fontsize=11)
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    heat(tsp, "terrain", terrains, terrains, sp, "speed_terrain_heatmap.png",
         "Ours - ablation by terrain x commanded-speed bin (pooled over difficulty). Blue = Ours better, "
         "red = Ours worse (CoT: negative is better); * = robust (CI excl. 0 and |diff| > noise_floor.json p90)")
    dsp = tabs["difficulty_speed"].copy()
    dsp["dlab"] = dsp.difficulty.astype(float)
    heat(dsp, "dlab", [float(d) for d in diffs], [f"d={d:g}" for d in diffs], sp, "speed_difficulty_heatmap.png",
         "Ours - ablation by difficulty x commanded-speed bin (pooled over terrains). Blue = Ours better, "
         "red = Ours worse; * = robust")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
