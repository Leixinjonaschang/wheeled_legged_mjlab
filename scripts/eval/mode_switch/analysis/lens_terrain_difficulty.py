#!/usr/bin/env python3
"""Terrain x difficulty lens on the mode_switch_v2 paired differences (Ours - RGGP, Ours - LPGP).

Run (from anywhere; deterministic, about 20 s):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas python \
    scripts/eval/mode_switch/analysis/lens_terrain_difficulty.py

Reads (read only): analysis/cells.csv, analysis/per_traj.csv.gz, analysis/null_replicate_cells.csv,
analysis/cell_modes.csv (all written by tidy.py) and, for the failure locations only, the arrays `valid`,
`course_x`, `difficulty`, `lane` of the 30 rollouts ../<terrain>__forward_wide__<policy>.npz.
Writes only: analysis/lens_td/*.csv|json and analysis/figures/td_*.png|pdf.

Pre-specified rule (from the analysis plan, applied as tidy.py already did in cells.csv):
  robust = 95 % lane-bootstrap CI of the paired difference excludes 0 AND |difference| > p90 run-to-run
  noise (noise_floor.json, matching metric); floor / ceiling = all three policies < 5 % / > 95 % (rates).
  Holm-adjusted lane sign-flip permutation p over the 40 cells of each (metric, comparison) family.
Everything the lens adds on top (not part of the pre-specified rule, labelled exploratory):
  * difficulty trend: least-squares slope of the paired difference over d = 0.25..1.0, reported as the
    fitted change from d = 0.25 to d = 1.0 (= 0.75 x slope), with a 95 % lane-cluster bootstrap CI
    (2000 replicates, lanes resampled within each cell, the same resampled lanes for all policies);
    random_rough is excluded from the pooled trend because its levels are replicates (its own trend is
    reported as a null check); a second pooled set also drops stepping_stones (floor effect d >= 0.75);
  * pyramid-like sections split by the lateral offset: |dy_start| (pre-treatment, identical across
    policies, so the paired difference stays valid) with bootstrap CIs, and |dy_centre| per policy
    (post-treatment, conditional on reaching the apex; descriptive only); threshold 0.14 m from the
    mode_switch_eval docstring (outer wheel one ring lower beyond it);
  * where the failed robots stopped (last valid base x): flat_in (< 4.0 m), rough first half [4.0, 5.8),
    rough second half [5.8, 7.6), flat_out (>= 7.6 m).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
RUN = HERE.parent
OUT = HERE / "lens_td"
FIG = HERE / "figures"

POLICIES = ("Ours", "RGGP", "LPGP")
BASELINES = ("RGGP", "LPGP")
LABEL = {"Ours": "Ours", "RGGP": "Ours w/o RG", "LPGP": "Ours w/o LP"}
DIFFS = (0.25, 0.5, 0.75, 1.0)
# Row order: sections with continuous / small features first, stairs and gaps last.
TERRAINS = ("random_rough", "tilted_grid", "random_spread", "discrete_obstacles", "hf_pyramid_slope",
            "hf_pyramid_slope_inv", "pyramid_stair", "pyramid_stair_inv", "random_stairs", "stepping_stones")
PYRAMIDS = ("hf_pyramid_slope", "hf_pyramid_slope_inv", "pyramid_stair", "pyramid_stair_inv", "random_stairs")
REPLICATE_TERRAIN = "random_rough"
FLOOR_TERRAIN = "stepping_stones"
DY_BAND = 0.14  # m (mode_switch_eval docstring: landings narrower than track + spawn spread)
APEX_X = 5.8  # m, section centre (lateral_metrics)
FAIL_BINS = (("flat_in", -np.inf, 4.0), ("rough_1st_half", 4.0, APEX_X), ("rough_2nd_half", APEX_X, 7.6),
             ("flat_out", 7.6, np.inf))

# metric name in cells.csv -> (per_traj column, scale, better, panel title, number format)
METRICS = {
    "switch_success_pct": ("switch_success", 100.0, "higher", "Switch success (pp)", "{:+.1f}"),
    "completion_pct": ("completed", 100.0, "higher", "Completion (pp)", "{:+.1f}"),
    "cot": ("cot", 1.0, "lower", "CoT", "{:+.3f}"),
    "failure_pct": ("failed", 100.0, "lower", "Failure (pp)", "{:+.1f}"),
}
TREND_EXTRA = {"mode_success_pct": ("mode_success", 100.0, "higher", "Gate-aware mode success (pp)", "{:+.1f}")}
BOOTSTRAP = 2000
SEED = 20260930 + 77  # lens-specific seed (tidy.py uses 20260930)

# Colours (dataviz reference palette: diverging blue <-> red with a neutral grey midpoint; ink tokens).
INK, INK2, MUTED, HAIR, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE_ARM = ["#104281", "#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6"]  # Ours better (dark -> light)
RED_ARM = ["#7a1518", "#a8292a", "#d64541", "#ec8a7d", "#f6c3bb"]  # Ours worse (dark -> light)
MID = "#f0efec"
CMP_COLOUR = {"RGGP": "#2a78d6", "LPGP": "#eb6834"}  # categorical slots 1, 2 (validated pair)
CMP_MARKER = {"RGGP": "o", "LPGP": "s"}
CELL_GAP = 0.02  # heatmap cell gap (data units)


def cmap_ours_better_high() -> LinearSegmentedColormap:
    """Low values red (Ours worse), high values blue (Ours better), grey midpoint."""
    return LinearSegmentedColormap.from_list("red_grey_blue", RED_ARM + [MID] + BLUE_ARM[::-1], N=256)


def read_inputs():
    cells = pd.read_csv(HERE / "cells.csv")
    cells["difficulty_f"] = pd.to_numeric(cells.difficulty.replace("ALL", np.nan))
    per = pd.read_csv(HERE / "per_traj.csv.gz")
    null = pd.read_csv(HERE / "null_replicate_cells.csv")
    null["difficulty_f"] = pd.to_numeric(null.difficulty.replace("ALL", np.nan))
    modes = pd.read_csv(HERE / "cell_modes.csv")
    return cells, per, null, modes


def to_float(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(col), errors="coerce") \
        .to_numpy(dtype=float) if col.dtype == object else col.to_numpy(dtype=float)


# --------------------------------------------------------------------------------------
# Lane sums and bootstrap (for the exploratory trend and split analyses).


class LaneSums:
    """num / den per (policy, metric, terrain, difficulty, lane), optionally restricted by a robot mask."""

    def __init__(self, per: pd.DataFrame, metrics: dict, mask_fn=None):
        self.metrics = list(metrics)
        T, D, L = len(TERRAINS), len(DIFFS), 16
        self.num = np.zeros((len(POLICIES), len(metrics), T, D, L))
        self.den = np.zeros_like(self.num)
        ti = per.terrain.map({t: i for i, t in enumerate(TERRAINS)}).to_numpy()
        di = per.difficulty.map({d: i for i, d in enumerate(DIFFS)}).to_numpy()
        li = per.lane.to_numpy()
        keep = np.ones(len(per), bool) if mask_fn is None else mask_fn(per)
        for p, pol in enumerate(POLICIES):
            sel = (per.policy.to_numpy() == pol) & keep
            for m, (col, scale, *_rest) in enumerate(metrics.values()):
                v = to_float(per.loc[sel, col]) * scale
                ok = np.isfinite(v)
                np.add.at(self.num[p, m], (ti[sel][ok], di[sel][ok], li[sel][ok]), v[ok])
                np.add.at(self.den[p, m], (ti[sel][ok], di[sel][ok], li[sel][ok]), 1.0)

    def point(self) -> np.ndarray:
        """[P, M, T, D]"""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.num.sum(-1) / self.den.sum(-1)

    def boot(self, W: np.ndarray) -> np.ndarray:
        """[B, P, M, T, D] with lane weights W [B, T, D, L]."""
        n = np.einsum("btdl,pmtdl->bpmtd", W, self.num, optimize=True)
        d = np.einsum("btdl,pmtdl->bpmtd", W, self.den, optimize=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return n / d


def lane_weights(rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(16, np.full(16, 1 / 16), size=(BOOTSTRAP, len(TERRAINS), len(DIFFS))).astype(float)


def ci(boot: np.ndarray, axis=0):
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanpercentile(boot, [2.5, 97.5], axis=axis)


# --------------------------------------------------------------------------------------
# 1. Grid table and counts (from cells.csv; the pre-specified statistics).


def grid_table(cells: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    g = cells[(cells.scope == "cell") & cells.metric.isin(list(METRICS) + ["mode_success_pct"])].copy()
    g["difficulty"] = g.difficulty_f
    g = g.merge(modes[["terrain", "difficulty", "gate_share_rough_pooled"]], on=["terrain", "difficulty"], how="left")
    g["levels_are_replicates"] = g.terrain == REPLICATE_TERRAIN
    g["stepping_stones_floor_zone"] = (g.terrain == FLOOR_TERRAIN) & (g.difficulty >= 0.75)
    for b in BASELINES:
        good = np.where(g.better == "higher", g[f"diff_{b}"] > 0, g[f"diff_{b}"] < 0)
        rob = g[f"robust_{b}"].astype(str) == "True"
        g[f"robust_win_{b}"] = rob & good
        g[f"robust_loss_{b}"] = rob & ~good
        g[f"holm05_win_{b}"] = (g[f"perm_p_holm_{b}"] < 0.05) & good
        g[f"holm05_loss_{b}"] = (g[f"perm_p_holm_{b}"] < 0.05) & ~good
    cols = ["terrain", "difficulty", "metric", "better", "expected_mode", "gate_share_rough_pooled",
            "levels_are_replicates", "stepping_stones_floor_zone", "floor_ceiling", "noise_p90",
            "Ours", "RGGP", "LPGP"]
    for b in BASELINES:
        cols += [f"diff_{b}", f"diff_{b}_lo", f"diff_{b}_hi", f"perm_p_{b}", f"perm_p_holm_{b}", f"robust_{b}",
                 f"outcome_{b}", f"robust_win_{b}", f"robust_loss_{b}", f"holm05_win_{b}", f"holm05_loss_{b}"]
    return g[cols].sort_values(["metric", "terrain", "difficulty"]).reset_index(drop=True)


def counts_table(grid: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nullc = null[(null.scope == "cell")]
    for m in METRICS:
        g = grid[grid.metric == m]
        nm = nullc[nullc.metric == m]
        n_good_null = int(((nm.robust_Ours_rep.astype(str) == "True")
                           & np.where(nm.better == "higher", nm.diff_Ours_rep > 0, nm.diff_Ours_rep < 0)).sum())
        n_bad_null = int((nm.robust_Ours_rep.astype(str) == "True").sum()) - n_good_null
        for b in BASELINES:
            inf = g.floor_ceiling != "floor"
            inf &= g.floor_ceiling != "ceiling"
            rows.append({
                "metric": m, "comparison": f"Ours-{b}", "cells": len(g),
                "floor_ceiling_cells": int((~inf).sum()),
                "robust_wins": int(g[f"robust_win_{b}"].sum()), "robust_losses": int(g[f"robust_loss_{b}"].sum()),
                "robust_wins_informative": int((g[f"robust_win_{b}"] & inf).sum()),
                "robust_losses_informative": int((g[f"robust_loss_{b}"] & inf).sum()),
                "holm05_wins": int(g[f"holm05_win_{b}"].sum()), "holm05_losses": int(g[f"holm05_loss_{b}"].sum()),
                "robust_and_holm05_wins": int((g[f"robust_win_{b}"] & g[f"holm05_win_{b}"]).sum()),
                "robust_and_holm05_losses": int((g[f"robust_loss_{b}"] & g[f"holm05_loss_{b}"]).sum()),
                # Under the global null each two-sided 95 % CI excludes 0 in a given direction with prob. <= 2.5 %;
                # robust additionally needs |diff| > p90 noise, so 0.025 x cells is an upper bound per direction.
                "chance_upper_bound_per_direction": 0.025 * len(g),
                "same_policy_null_robust_better": n_good_null, "same_policy_null_robust_worse": n_bad_null,
                "same_policy_null_cells": len(nm),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. Difficulty trend (exploratory).


def trend_contrast() -> np.ndarray:
    d = np.array(DIFFS)
    c = (d - d.mean()) / np.sum((d - d.mean()) ** 2)
    return 0.75 * c  # fitted change of a linear fit from d = 0.25 to d = 1.0


def trend_table(sums: LaneSums, W: np.ndarray, metrics: dict) -> pd.DataFrame:
    pt = sums.point()  # [P, M, T, D]
    bt = sums.boot(W)  # [B, P, M, T, D]
    c = trend_contrast()
    rows = []
    sets = {t: [t] for t in TERRAINS}
    sets["pooled_9 (excl. random_rough)"] = [t for t in TERRAINS if t != REPLICATE_TERRAIN]
    sets["pooled_8 (excl. random_rough, stepping_stones)"] = [t for t in TERRAINS if t not in (REPLICATE_TERRAIN,
                                                                                                FLOOR_TERRAIN)]
    for m, name in enumerate(metrics):
        for bi, b in enumerate(BASELINES, start=1):
            dpt = pt[0, m] - pt[bi, m]  # [T, D]
            dbt = bt[:, 0, m] - bt[:, bi, m]  # [B, T, D]
            for label, ts in sets.items():
                idx = [TERRAINS.index(t) for t in ts]
                # Equal-weight mean over the terrains of the set, then the linear contrast over difficulty.
                per_d_pt = dpt[idx].mean(0)
                per_d_bt = dbt[:, idx].mean(1)
                fit_pt = float(per_d_pt @ c)
                fit_bt = per_d_bt @ c
                end_pt = float(per_d_pt[-1] - per_d_pt[0])
                end_bt = per_d_bt[:, -1] - per_d_bt[:, 0]
                (flo, fhi), (elo, ehi) = ci(fit_bt), ci(end_bt)
                dlo, dhi = ci(per_d_bt)
                rows.append({
                    "metric": name, "comparison": f"Ours-{b}", "set": label, "n_terrains": len(ts),
                    **{f"diff_d{d:g}": float(v) for d, v in zip(DIFFS, per_d_pt)},
                    **{f"diff_d{d:g}_lo": float(v) for d, v in zip(DIFFS, dlo)},
                    **{f"diff_d{d:g}_hi": float(v) for d, v in zip(DIFFS, dhi)},
                    "fitted_change_0.25_to_1": fit_pt, "fitted_lo": float(flo), "fitted_hi": float(fhi),
                    "fitted_ci_excl0": bool(flo > 0 or fhi < 0),
                    "end_minus_start": end_pt, "end_lo": float(elo), "end_hi": float(ehi),
                    "peak_difficulty": float(DIFFS[int(np.nanargmax(per_d_pt if metrics[name][2] == "higher"
                                                                    else -per_d_pt))]),
                    "better": metrics[name][2],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Pyramid-like sections: lateral-offset splits.


def dy_start_split(per: pd.DataFrame, W: np.ndarray) -> pd.DataFrame:
    metrics = {k: METRICS[k] for k in ("switch_success_pct", "completion_pct")}
    rows = []
    for band, fn in (("|dy_start|<=0.14", lambda df: df.dy_start.abs().to_numpy() <= DY_BAND),
                     ("|dy_start|>0.14", lambda df: df.dy_start.abs().to_numpy() > DY_BAND)):
        s = LaneSums(per, metrics, fn)
        pt, bt = s.point(), s.boot(W)
        n = s.den[0, 0].sum(-1)  # robots per cell in the band (same for all policies: pre-treatment)
        for t in PYRAMIDS:
            ti = TERRAINS.index(t)
            for di, d in enumerate(DIFFS):
                for m, name in enumerate(metrics):
                    row = {"terrain": t, "difficulty": d, "band": band, "metric": name, "n_robots": int(n[ti, di])}
                    for p, pol in enumerate(POLICIES):
                        row[pol] = float(pt[p, m, ti, di])
                    for bi, b in enumerate(BASELINES, start=1):
                        lo, hi = ci(bt[:, 0, m, ti, di] - bt[:, bi, m, ti, di])
                        row[f"diff_{b}"] = float(pt[0, m, ti, di] - pt[bi, m, ti, di])
                        row[f"diff_{b}_lo"], row[f"diff_{b}_hi"] = float(lo), float(hi)
                    rows.append(row)
    return pd.DataFrame(rows)


def dy_centre_split(per: pd.DataFrame) -> pd.DataFrame:
    s = per[per.terrain.isin(PYRAMIDS)].copy()
    s["centre_band"] = np.where(s.dy_centre.isna(), "not_reached_apex",
                                np.where(s.dy_centre.abs() <= DY_BAND, "|dy_centre|<=0.14", "|dy_centre|>0.14"))
    s["completion_pct"] = s.completed.astype(float) * 100
    s["switch_success_pct"] = s.switch_success.astype(float) * 100
    g = s.groupby(["terrain", "difficulty", "policy", "centre_band"]).agg(
        n=("env", "size"), completion_pct=("completion_pct", "mean"),
        switch_success_pct=("switch_success_pct", "mean")).reset_index()
    tot = s.groupby(["terrain", "difficulty", "policy"]).env.size().rename("n_cell").reset_index()
    g = g.merge(tot, on=["terrain", "difficulty", "policy"])
    g["share_pct"] = 100 * g.n / g.n_cell
    return g


# --------------------------------------------------------------------------------------
# 4. Failure locations.


def failure_locations(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TERRAINS:
        for pol in POLICIES:
            z = np.load(RUN / f"{t}__forward_wide__{pol}.npz")
            valid, x = z["valid"], z["course_x"]
            diff, lane = z["difficulty"], z["lane"]
            sub = per[(per.terrain == t) & (per.policy == pol)].set_index("env")
            if not (np.allclose(sub.difficulty.sort_index().to_numpy(), diff)
                    and np.array_equal(sub.lane.sort_index().to_numpy(), lane)):
                raise RuntimeError(f"{t} {pol}: env mapping differs from per_traj")
            failed = sub.failed.sort_index().to_numpy().astype(bool)
            n_valid = valid.sum(0)
            last = np.array([np.flatnonzero(valid[:, i])[-1] if n_valid[i] else 0 for i in range(valid.shape[1])])
            if not np.all(valid[:, failed].cumsum(0)[n_valid[failed] - 1, np.arange(failed.sum())] == n_valid[failed]):
                raise RuntimeError(f"{t} {pol}: valid mask not contiguous for failed robots")
            x_last = x[last, np.arange(valid.shape[1])].astype(float)
            for d in DIFFS:
                sel = failed & np.isclose(diff, d)
                row = {"terrain": t, "difficulty": d, "policy": pol, "n_failed": int(sel.sum()),
                       "median_x_fail": float(np.median(x_last[sel])) if sel.any() else np.nan}
                for name, lo, hi in FAIL_BINS:
                    row[f"n_{name}"] = int(np.sum(sel & (x_last >= lo) & (x_last < hi)))
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures.


def text_colour(rgb) -> str:
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.23 else INK


def heatmap_figure(grid: pd.DataFrame, b: str, vmax: dict, path: Path) -> None:
    base = cmap_ours_better_high()
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 7.2), gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    for k, (ax, (m, (_, _, better, title, fmt))) in enumerate(zip(axes, METRICS.items())):
        g = grid[grid.metric == m].set_index(["terrain", "difficulty"])
        cmap = base if better == "higher" else base.reversed()
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax[m], vmax=vmax[m])
        ax.set_facecolor(SURFACE)
        for i, t in enumerate(TERRAINS):
            for j, d in enumerate(DIFFS):
                r = g.loc[(t, d)]
                v = float(r[f"diff_{b}"])
                fc = r.floor_ceiling in ("floor", "ceiling")
                rgb = to_rgb(SURFACE) if fc else cmap(norm(np.clip(v, -vmax[m], vmax[m])))[:3]
                # Inset fill: a surface-coloured gap between cells (no edge line, which the axes would clip).
                ax.add_patch(Rectangle((j + CELL_GAP, i + CELL_GAP), 1 - 2 * CELL_GAP, 1 - 2 * CELL_GAP,
                                       facecolor=rgb, edgecolor="none", linewidth=0))
                if fc:
                    ax.add_patch(Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, facecolor="none", edgecolor=HAIR,
                                           hatch="////", linewidth=0))
                robust = str(r[f"robust_{b}"]) == "True"
                if robust:
                    ax.add_patch(Rectangle((j + 0.045, i + 0.045), 0.91, 0.91, facecolor="none", edgecolor=INK,
                                           linewidth=1.8))
                txt = fmt.format(v).replace("-", "−")
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=8.6,
                        fontweight="bold" if robust else "normal",
                        color=MUTED if fc else text_colour(rgb))
                tags = []
                if fc:
                    tags.append("F" if r.floor_ceiling == "floor" else "C")
                if m == "switch_success_pct" and r.expected_mode == "roll":
                    tags.append("r")
                if tags:
                    ax.text(j + 0.93, i + 0.12, "".join(tags), ha="right", va="top", fontsize=6.5,
                            color=MUTED if fc else text_colour(rgb), style="italic")
        ax.set_xlim(0, 4)
        ax.set_ylim(len(TERRAINS), 0)
        ax.set_xticks(np.arange(4) + 0.5, [f"{d:g}" for d in DIFFS], fontsize=8.5, color=INK2)
        ax.set_yticks(np.arange(len(TERRAINS)) + 0.5)
        if k == 0:
            labels = [t + (" †" if t == REPLICATE_TERRAIN else " ‡" if t == FLOOR_TERRAIN else "")
                      for t in TERRAINS]
            ax.set_yticklabels(labels, fontsize=9, color=INK)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.xaxis.set_ticks_position("top")
        ax.set_xlabel("difficulty d", fontsize=8.5, color=INK2, labelpad=4)
        ax.xaxis.set_label_position("top")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10.5, color=INK, pad=26)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, aspect=28)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, colors=INK2, length=2)
        cb.set_label(f"Ours − {LABEL[b]}   (blue = Ours better)", fontsize=7.8, color=INK2)
    fig.suptitle(f"Ours − {LABEL[b]}: paired difference per terrain × difficulty "
                 "(256 robots = 16 lanes × 16 per cell; one training seed per policy)",
                 fontsize=11.5, color=INK, y=0.995)
    note = ("Bold + black frame: robust (95 % lane-bootstrap CI excludes 0 and |Δ| > p90 run-to-run noise: "
            "4.30 pp switch, 3.95 pp completion / failure, 0.0103 CoT).   Hatched F / C: floor / ceiling "
            "(all three policies < 5 % / > 95 %).\n"
            "r: rough-window gate share < 0.5 (rolling is legitimate; switch success still demands stepping).   "
            "† random_rough: levels are replicates (geometry ignores d).   "
            "‡ stepping_stones: completion ≤ 15 % for every policy at d ≥ 0.75.   "
            "Colour scales shared with the other comparison.")
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7.6, color=INK2, wrap=True)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.86, bottom=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE,
                    metadata={"CreationDate": None} if ext == "pdf" else {"Software": None})
    plt.close(fig)


def trend_figure(tr: pd.DataFrame, path: Path) -> None:
    metrics = {**METRICS, **TREND_EXTRA}
    order = ["pooled_8 (excl. random_rough, stepping_stones)", "pooled_9 (excl. random_rough)"] + \
        [t for t in TERRAINS if t != REPLICATE_TERRAIN] + [REPLICATE_TERRAIN]
    ylabels = {"pooled_8 (excl. random_rough, stepping_stones)": "pooled, 8 terrains§",
               "pooled_9 (excl. random_rough)": "pooled, 9 terrains¶",
               REPLICATE_TERRAIN: "random_rough (null check)"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 6.4), sharey=True, gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    for ax, (m, (_, _, better, title, _)) in zip(axes, metrics.items()):
        ax.set_facecolor(SURFACE)
        sub = tr[tr.metric == m].set_index(["set", "comparison"])
        for i, s in enumerate(order):
            for k, b in enumerate(BASELINES):
                r = sub.loc[(s, f"Ours-{b}")]
                y = i + (-0.17 if k == 0 else 0.17)
                ax.plot([r.fitted_lo, r.fitted_hi], [y, y], color=CMP_COLOUR[b], linewidth=2.0,
                        solid_capstyle="round")
                ax.plot(r["fitted_change_0.25_to_1"], y, marker=CMP_MARKER[b], markersize=6.5, color=CMP_COLOUR[b],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none",
                        label=f"Ours − {LABEL[b]}" if i == 0 else None)
        ax.axvline(0, color="#c3c2b7", linewidth=1.0, zorder=0)
        ax.axhline(1.5, color=HAIR, linewidth=0.8, zorder=0)
        ax.axhline(len(order) - 1.5, color=HAIR, linewidth=0.8, zorder=0)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_yticks(range(len(order)), [ylabels.get(s, s) for s in order], fontsize=8.8, color=INK)
        ax.tick_params(axis="x", labelsize=8, colors=INK2, length=2)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=HAIR, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.set_title(title, fontsize=10, color=INK, pad=6)
        arrow = "→" if better == "higher" else "←"
        ax.set_xlabel(f"fitted change of the difference, d 0.25→1.0\n({arrow} Ours' advantage grows with d)",
                      fontsize=8, color=INK2)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2, frameon=False, fontsize=9,
                   handletextpad=0.3, columnspacing=1.2)
    fig.suptitle("Does the paired difference change with difficulty?  Least-squares change from d = 0.25 to 1.0 "
                 "with 95 % lane-bootstrap CI (exploratory, not part of the pre-specified rule)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.01, "¶ equal-weight mean over the 9 terrains whose geometry depends on d.   "
             "§ additionally without stepping_stones (floor effect at d ≥ 0.75).   "
             "random_rough levels are replicates, so its 'trend' shows what instance sampling alone produces.",
             ha="center", va="bottom", fontsize=7.8, color=INK2)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.84, bottom=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE,
                    metadata={"CreationDate": None} if ext == "pdf" else {"Software": None})
    plt.close(fig)


def pooled_figure(pooled: pd.DataFrame, path: Path) -> None:
    """cells.csv difficulty scope (all 10 terrains pooled): paired difference per d with 95 % CI;
    filled marker = robust under the pre-specified rule, hollow = not robust (or no noise floor)."""
    metrics = {**METRICS, **TREND_EXTRA}
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 3.9), gridspec_kw={"wspace": 0.28})
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(DIFFS))
    for ax, (m, (_, _, better, title, _)) in zip(axes, metrics.items()):
        ax.set_facecolor(SURFACE)
        sub = pooled[pooled.metric == m].copy()
        sub["d"] = pd.to_numeric(sub.difficulty)
        sub = sub.sort_values("d")
        for k, b in enumerate(BASELINES):
            off = -0.07 if k == 0 else 0.07
            ax.fill_between(x + off, sub[f"diff_{b}_lo"], sub[f"diff_{b}_hi"], color=CMP_COLOUR[b], alpha=0.14,
                            linewidth=0)
            ax.plot(x + off, sub[f"diff_{b}"], color=CMP_COLOUR[b], linewidth=2.0,
                    label=f"Ours \u2212 {LABEL[b]}")
            rob = (sub[f"robust_{b}"].astype(str) == "True").to_numpy()
            ax.plot((x + off)[rob], sub[f"diff_{b}"].to_numpy()[rob], marker=CMP_MARKER[b], linestyle="none",
                    markersize=7, color=CMP_COLOUR[b], markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.plot((x + off)[~rob], sub[f"diff_{b}"].to_numpy()[~rob], marker=CMP_MARKER[b], linestyle="none",
                    markersize=7, markerfacecolor=SURFACE, markeredgecolor=CMP_COLOUR[b], markeredgewidth=1.6)
        ax.axhline(0, color="#c3c2b7", linewidth=1.0, zorder=0)
        ax.set_xticks(x, [f"{d:g}" for d in DIFFS])
        ax.set_xlabel("difficulty d", fontsize=8.5, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2, length=2)
        ax.grid(axis="y", color=HAIR, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        arrow = "\u2191" if better == "higher" else "\u2193"
        ax.set_title(f"{title}\n({arrow} = Ours better)", fontsize=9.5, color=INK, pad=6)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.22), ncol=2, frameon=False, fontsize=9,
                   handletextpad=0.4, columnspacing=1.2)
    fig.suptitle("Difficulty-pooled paired difference (all 10 terrains, 2560 robots per d; cells.csv scope "
                 "'difficulty'), 95 % lane-bootstrap CI.  Filled marker: robust; hollow: not robust / no noise floor",
                 fontsize=10.5, color=INK, y=1.02)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.74, bottom=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE, bbox_inches="tight",
                    metadata={"CreationDate": None} if ext == "pdf" else {"Software": None})
    plt.close(fig)


# --------------------------------------------------------------------------------------


def main() -> int:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    cells, per, null, modes = read_inputs()
    per = per[per.terrain.isin(TERRAINS)].copy()
    assert len(per) == 30720 and set(per.difficulty.unique()) == set(DIFFS)

    grid = grid_table(cells, modes)
    grid.to_csv(OUT / "grid_cells.csv", index=False)
    counts = counts_table(grid, null)
    counts.to_csv(OUT / "counts.csv", index=False)

    # "Best settings for Ours" on switch success: robust win AND Holm p < 0.05 against BOTH ablations, with the
    # completion / CoT / failure differences of the same cells (losses reported next to the wins).
    ss = grid[grid.metric == "switch_success_pct"].set_index(["terrain", "difficulty"])
    both = ss[ss.robust_win_RGGP & ss.holm05_win_RGGP & ss.robust_win_LPGP & ss.holm05_win_LPGP]
    best = both[["expected_mode", "Ours", "RGGP", "LPGP", "diff_RGGP", "perm_p_holm_RGGP", "diff_LPGP",
                 "perm_p_holm_LPGP"]].copy()
    for m in ("completion_pct", "cot"):
        other = grid[grid.metric == m].set_index(["terrain", "difficulty"])
        for b in BASELINES:
            best[f"{m}_diff_{b}"] = other.loc[best.index, f"diff_{b}"]
            best[f"{m}_outcome_{b}"] = other.loc[best.index, f"outcome_{b}"]
    best.sort_values("diff_LPGP", ascending=False).reset_index().to_csv(OUT / "best_cells_switch_success.csv",
                                                                           index=False)

    # Point estimates of the lane sums must reproduce cells.csv (same definition: ratio of lane sums).
    all_metrics = {**METRICS, **TREND_EXTRA}
    sums = LaneSums(per, all_metrics)
    pt = sums.point()
    max_dev = 0.0
    for m, name in enumerate(all_metrics):
        g = grid[grid.metric == name].set_index(["terrain", "difficulty"])
        for ti, t in enumerate(TERRAINS):
            for di, d in enumerate(DIFFS):
                for p, pol in enumerate(POLICIES):
                    max_dev = max(max_dev, abs(pt[p, m, ti, di] - g.loc[(t, d), pol]))
    assert max_dev < 1e-6, max_dev

    rng = np.random.default_rng(SEED)
    W = lane_weights(rng)
    trend = trend_table(sums, W, all_metrics)
    trend.to_csv(OUT / "difficulty_trend.csv", index=False)

    pooled = cells[(cells.scope == "difficulty") & cells.metric.isin(list(all_metrics))][
        ["metric", "difficulty", "Ours", "RGGP", "LPGP", "diff_RGGP", "diff_RGGP_lo", "diff_RGGP_hi", "robust_RGGP",
         "perm_p_holm_RGGP", "diff_LPGP", "diff_LPGP_lo", "diff_LPGP_hi", "robust_LPGP", "perm_p_holm_LPGP"]]
    pooled.to_csv(OUT / "difficulty_pooled_from_cells.csv", index=False)

    split_start = dy_start_split(per, W)
    split_start.to_csv(OUT / "pyramid_dy_start_split.csv", index=False)
    split_centre = dy_centre_split(per)
    split_centre.to_csv(OUT / "pyramid_dy_centre_split.csv", index=False)
    fails = failure_locations(per)
    fails.to_csv(OUT / "failure_locations.csv", index=False)

    # random_rough replicate levels: spread of the cell differences across the 4 replicate levels.
    rr = {}
    for m in METRICS:
        g = grid[(grid.metric == m) & (grid.terrain == REPLICATE_TERRAIN)]
        rr[m] = {b: {"min": float(g[f"diff_{b}"].min()), "max": float(g[f"diff_{b}"].max()),
                     "range": float(g[f"diff_{b}"].max() - g[f"diff_{b}"].min())} for b in BASELINES}

    # Context only: the previous experiment (mode_switch/, forward v_x U(0.6, 1.0), d = 1) used these three
    # sections with older embeddings; the matching v2 cells, equal-weight mean, all speeds and v in [0.5, 1.0).
    prev_terrains = ["discrete_obstacles", "random_spread", "tilted_grid"]
    prev = {}
    for scope, sb in (("cell", "ALL"), ("cell_speed", "[0.5,1.0)")):
        sub = cells[(cells.scope == scope) & (cells.speed_bin == sb) & (cells.difficulty_f == 1.0)
                    & cells.terrain.isin(prev_terrains) & cells.metric.isin(list(METRICS))]
        assert sub.terrain.nunique() == 3, (scope, sub.terrain.unique())
        prev[f"{scope}|{sb}"] = {m: {p: float(g[p].mean()) for p in POLICIES} for m, g in sub.groupby("metric")}

    vmax = {"switch_success_pct": 85.0, "completion_pct": 40.0, "cot": 0.40, "failure_pct": 40.0}
    for m, v in vmax.items():
        worst = max(grid.loc[grid.metric == m, f"diff_{b}"].abs().max() for b in BASELINES)
        assert worst <= v, (m, worst)
    for b in BASELINES:
        heatmap_figure(grid, b, vmax, FIG / f"td_heatmap_ours_minus_{b.lower()}")
    trend_figure(trend, FIG / "td_difficulty_trend")
    pooled_figure(pooled, FIG / "td_difficulty_pooled")

    results = {
        "inputs": ["analysis/cells.csv", "analysis/per_traj.csv.gz", "analysis/null_replicate_cells.csv",
                   "analysis/cell_modes.csv", "<terrain>__forward_wide__<policy>.npz (valid, course_x)"],
        "check_point_estimates_vs_cells_csv_max_abs_dev": max_dev,
        "bootstrap": {"replicates": BOOTSTRAP, "seed": SEED, "unit": "lane within (terrain, difficulty)"},
        "counts": counts.to_dict(orient="records"),
        "random_rough_replicate_level_spread": rr,
        "v2_cells_matching_previous_experiment_d1_mean_of_3": prev,
        "colour_scale_vmax": vmax,
        "figures": [str(FIG / f"td_heatmap_ours_minus_{b.lower()}.png") for b in BASELINES]
        + [str(FIG / "td_difficulty_trend.png"), str(FIG / "td_difficulty_pooled.png")],
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(counts.to_string())
    print(f"max |point estimate - cells.csv| = {max_dev:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
