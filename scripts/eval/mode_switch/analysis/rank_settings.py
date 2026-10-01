"""Rank the evaluation settings (cells / slices) most favourable to Ours, under the pre-specified rules.

Reads only files already written by tidy.py / lens_speed.py (no simulation, no new bootstrap):
  analysis/cells.csv                      scopes all, difficulty, terrain, speed_bin, cell, cell_speed
  analysis/lens_speed/difficulty_speed.csv, terrain_speed.csv   (same columns, lens_speed.py)
  analysis/null_replicate_cells.csv, lens_overall_replicate_pooled.csv, lens_speed/null_*.csv
                                          same-policy null (Ours vs its own replicate run) = chance level
  analysis/per_traj.csv.gz, per_traj_replicate.csv.gz   supplementary replicate check (point estimates)
Writes analysis/favourable_settings.csv, analysis/unfavourable_settings.csv, analysis/rank_settings.json
and prints a summary (analysis/rank_settings.out when redirected).

Candidates ("examined settings"): every group of the 8 scopes = 1 + 4 + 10 + 3 + 40 + 120 + 12 + 30 = 220.

Rule (fixed before looking at the ranking; follows the pre-specified analysis rules):
  * robust difference = 95 % lane-bootstrap CI excludes 0 AND |diff| > noise_floor.json p90 (the `robust_<B>`
    column of the inputs); a rate metric is uninformative in a group where all three policies are < 5 % or
    > 95 % (`floor_ceiling`).
  * eligible (favourable) = switch success informative AND robust Ours win vs BOTH ablations AND no robust
    completion loss vs either ablation (a completion floor/ceiling row counts as "no evidence", flagged).
  * score = min(Ours - w/o RG, Ours - w/o LP) on switch success (primary key, descending);
    min of the two completion differences = secondary key. `min4` (min over all four differences) is
    reported as a stricter alternative ordering.
  * random_rough: its geometry ignores difficulty, so its cell / cell_speed rows are replicate splits of the
    same setting. They stay in the Holm families (they were examined) but are not ranked; random_rough enters
    the ranking through its terrain and terrain_speed rows.
  * Holm: raw lane sign-flip permutation p of every candidate row (exact for <= 16 lanes, 10000 MC otherwise),
    adjusted over ALL 220 examined candidates, separately per (metric, comparison).
  * chance expectation: number of rows the same-policy null (Ours vs Ours replicate, identical 220 groups)
    flags as robust, per metric; and the nominal bound 0.025 x m per direction.
Supplementary (not a pre-specified rule): the switch/completion differences recomputed with the Ours
replicate run in place of Ours (same robots, same baselines) - a partial winner's-curse check of the ranking
(it replaces only Ours; split_half.py re-selects on held-out lanes for all three policies).
Supplementary (added after verification, not a pre-specified rule): `comp_ci_loss_<B>` = informative completion
difference whose 95 % CI excludes 0 with Ours worse, whatever its size relative to the cell-level noise floor. The
noise floor is a 256-robot cell-level p90 and is applied unchanged to pooled rows with up to 10240 robots, where the
run-to-run difference is much smaller (lens_overall_replicate_pooled.csv), so pooled completion losses can be
significant but "not robust". `eligible_strict` = eligible AND no such CI completion loss; `rank_strict` orders
them by the same keys. The pre-specified `status` / `rank` are unchanged.

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/rank_settings.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
RUN = HERE.parent
BASE = ("RGGP", "LPGP")
LAB = {"RGGP": "w/o RG", "LPGP": "w/o LP"}
PRIMARY = ("switch_success_pct", "completion_pct")
CONTEXT = ("cot", "mode_success_pct", "switch_success_dur_pct", "roll_flat_out_pct", "step_rough_pct")
SCOPES = ("all", "difficulty", "terrain", "speed_bin", "difficulty_speed", "terrain_speed", "cell", "cell_speed")
KEY = ["scope", "terrain", "difficulty", "speed_bin"]
PYRAMIDS = {"pyramid_stair", "pyramid_stair_inv", "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_stairs"}


def holm(p: np.ndarray) -> np.ndarray:  # identical to tidy.holm
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    m = len(order)
    out[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return out


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def norm_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["difficulty"] = df["difficulty"].astype(str).map(lambda v: v if v == "ALL" else f"{float(v):g}")
    for c in ("terrain", "speed_bin", "scope"):
        df[c] = df[c].astype(str)
    return df


def load_candidates() -> pd.DataFrame:
    parts = [pd.read_csv(HERE / "cells.csv"),
             pd.read_csv(HERE / "lens_speed" / "difficulty_speed.csv"),
             pd.read_csv(HERE / "lens_speed" / "terrain_speed.csv")]
    df = norm_key(pd.concat(parts, ignore_index=True))
    df = df[df.metric.isin(PRIMARY + CONTEXT)]
    assert not df.duplicated(KEY + ["metric"]).any()
    return df


def load_null() -> pd.DataFrame:
    parts = [pd.read_csv(HERE / "null_replicate_cells.csv"),
             pd.read_csv(HERE / "lens_overall_replicate_pooled.csv"),
             pd.read_csv(HERE / "lens_speed" / "null_difficulty_speed.csv"),
             pd.read_csv(HERE / "lens_speed" / "null_terrain_speed.csv")]
    return norm_key(pd.concat(parts, ignore_index=True))


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate; columns <short>_<field> for the metrics used."""
    short = {"switch_success_pct": "sw", "completion_pct": "comp", "cot": "cot", "mode_success_pct": "mode",
             "switch_success_dur_pct": "swdur", "roll_flat_out_pct": "rfo", "step_rough_pct": "sr"}
    keep = ["Ours", "RGGP", "LPGP", "Ours_lo", "Ours_hi", "RGGP_lo", "RGGP_hi", "LPGP_lo", "LPGP_hi",
            "floor_ceiling", "noise_p90"]
    for b in BASE:
        keep += [f"diff_{b}", f"diff_{b}_lo", f"diff_{b}_hi", f"perm_p_{b}", f"robust_{b}", f"outcome_{b}"]
    out = None
    for metric, s in short.items():
        sub = df[df.metric == metric].set_index(KEY)[keep + ["n_robots", "n_lanes", "n_cells", "expected_mode"]]
        sub = sub.rename(columns={c: f"{s}_{c}" for c in keep})
        if out is None:
            out = sub
        else:
            out = out.join(sub.drop(columns=["n_robots", "n_lanes", "n_cells", "expected_mode"]), how="outer")
    out = out.reset_index()
    for s in short.values():
        for b in BASE:
            out[f"{s}_robust_{b}"] = as_bool(out[f"{s}_robust_{b}"].fillna(False))
    return out


def roll_cells() -> set:
    cm = pd.read_csv(HERE / "cell_modes.csv")
    cm["difficulty"] = cm["difficulty"].map(lambda v: f"{float(v):g}")
    return {(r.terrain, r.difficulty) for r in cm.itertuples() if r.expected_mode == "roll"}


def group_mask(pt: pd.DataFrame, scope: str, t: str, d: str, sb: str) -> np.ndarray:
    m = np.ones(len(pt), bool)
    if t != "ALL":
        m &= pt.terrain.to_numpy() == t
    if d != "ALL":
        m &= pt.dkey.to_numpy() == d
    if sb != "ALL":
        m &= pt.speed_bin.to_numpy() == sb
    return m


def replicate_check(w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Point estimates of Ours - B and Ours_rep - B (switch success, completion) from the per-robot tables."""
    cols = ["terrain", "difficulty", "env", "speed_bin", "policy", "completed", "switch_success"]
    pt = pd.read_csv(HERE / "per_traj.csv.gz", usecols=cols)
    rep = pd.read_csv(HERE / "per_traj_replicate.csv.gz", usecols=cols)
    rep["policy"] = "Ours_rep"
    pt = pd.concat([pt, rep], ignore_index=True)
    pt["dkey"] = pt["difficulty"].map(lambda v: f"{float(v):g}")
    for c in ("completed", "switch_success"):
        pt[c] = as_bool(pt[c]).astype(float) * 100.0
    pols = {}
    ref = None
    for p in ("Ours", "RGGP", "LPGP", "Ours_rep"):
        s = pt[pt.policy == p].sort_values(["terrain", "env"]).reset_index(drop=True)
        if ref is None:
            ref = s
        else:
            assert np.array_equal(s.env.to_numpy(), ref.env.to_numpy())
            assert np.array_equal(s.terrain.to_numpy(), ref.terrain.to_numpy())
            assert np.array_equal(s.speed_bin.to_numpy(), ref.speed_bin.to_numpy())
        pols[p] = s
    rows, dev = [], 0.0
    for r in w.itertuples():
        m = group_mask(ref, r.scope, r.terrain, r.difficulty, r.speed_bin)
        rec = {}
        for short, col in (("sw", "switch_success"), ("comp", "completed")):
            est = {p: float(pols[p][col].to_numpy()[m].mean()) for p in pols}
            dev = max(dev, abs(est["Ours"] - getattr(r, f"{short}_Ours")))
            for b in BASE:
                dev = max(dev, abs(est[b] - getattr(r, f"{short}_{b}")))
                rec[f"{short}_rep_diff_{b}"] = est["Ours_rep"] - est[b]
            rec[f"{short}_Ours_rep"] = est["Ours_rep"]
        rows.append(rec)
    return pd.DataFrame(rows, index=w.index), {"max_abs_dev_point_estimates_vs_inputs": dev}


def main() -> None:
    df = load_candidates()
    w = wide(df)
    w["scope"] = pd.Categorical(w["scope"], SCOPES, ordered=True)
    w = w.sort_values(KEY).reset_index(drop=True)
    w["scope"] = w["scope"].astype(str)
    n_by_scope = w.scope.value_counts().reindex(SCOPES).to_dict()
    assert len(w) == 220, len(w)

    # Holm over all examined candidates, per (metric, comparison).
    for s in ("sw", "comp", "cot"):
        for b in BASE:
            w[f"{s}_holm220_{b}"] = holm(w[f"{s}_perm_p_{b}"].to_numpy(float))

    rolls = roll_cells()

    def n_roll(r) -> int:
        return sum(1 for (t, d) in rolls
                   if (r.terrain in ("ALL", t)) and (r.difficulty in ("ALL", d)))

    w["n_roll_cells_in_group"] = [n_roll(r) for r in w.itertuples()]
    w["replicate_split"] = (w.terrain == "random_rough") & w.scope.isin(["cell", "cell_speed"])

    def flags(r) -> str:
        f = []
        if r.replicate_split:
            f.append("random_rough replicate level (difficulty ignored by geometry)")
        if r.terrain == "stepping_stones" and r.difficulty in ("0.75", "1"):
            f.append("stepping_stones floor effect d>=0.75")
        if r.terrain in PYRAMIDS:
            f.append("pyramid-type: narrow landing, check dy split")
        if r.n_roll_cells_in_group:
            f.append(f"{r.n_roll_cells_in_group} gate-roll cell(s) in group")
        if r.comp_floor_ceiling != "informative":
            f.append(f"completion {r.comp_floor_ceiling}")
        if r.scope in ("cell_speed", "difficulty_speed", "terrain_speed", "speed_bin"):
            f.append("speed-bin group: cell-level noise floor applied (lenient)")
        return "; ".join(f)

    w["flags"] = [flags(r) for r in w.itertuples()]

    sw_inf = w.sw_floor_ceiling.eq("informative")
    comp_inf = w.comp_floor_ceiling.eq("informative")
    sw_win = {b: sw_inf & w[f"sw_robust_{b}"] & (w[f"sw_diff_{b}"] > 0) for b in BASE}
    sw_loss = {b: sw_inf & w[f"sw_robust_{b}"] & (w[f"sw_diff_{b}"] < 0) for b in BASE}
    c_win = {b: comp_inf & w[f"comp_robust_{b}"] & (w[f"comp_diff_{b}"] > 0) for b in BASE}
    c_loss = {b: comp_inf & w[f"comp_robust_{b}"] & (w[f"comp_diff_{b}"] < 0) for b in BASE}
    w["sw_min"] = w[["sw_diff_RGGP", "sw_diff_LPGP"]].min(axis=1)
    w["comp_min"] = w[["comp_diff_RGGP", "comp_diff_LPGP"]].min(axis=1)
    w["min4"] = w[["sw_diff_RGGP", "sw_diff_LPGP", "comp_diff_RGGP", "comp_diff_LPGP"]].min(axis=1)
    w["sw_robust_win_both"] = sw_win["RGGP"] & sw_win["LPGP"]
    w["comp_robust_loss_any"] = c_loss["RGGP"] | c_loss["LPGP"]
    w["comp_robust_win_n"] = c_win["RGGP"].astype(int) + c_win["LPGP"].astype(int)
    w["sw_holm_both_lt05"] = (w.sw_holm220_RGGP < 0.05) & (w.sw_holm220_LPGP < 0.05)
    w["comp_loss_holm_lt05"] = ((c_loss["RGGP"] & (w.comp_holm220_RGGP < 0.05))
                                | (c_loss["LPGP"] & (w.comp_holm220_LPGP < 0.05)))

    def status(r, i) -> str:
        if r.replicate_split:
            return "not ranked (replicate split)"
        if r.sw_floor_ceiling != "informative":
            return "switch uninformative"
        if r.sw_robust_win_both and not r.comp_robust_loss_any:
            return "eligible"
        if r.sw_robust_win_both:
            return "trade-off: switch win both, completion robust loss"
        if sw_loss["RGGP"][i] or sw_loss["LPGP"][i]:
            return "switch robust loss"
        return "switch not robust vs both"

    w["status"] = [status(r, i) for i, r in zip(w.index, w.itertuples())]
    el = w.status.eq("eligible")
    order = w[el].sort_values(["sw_min", "comp_min"], ascending=[False, False]).index
    w["rank"] = np.nan
    w.loc[order, "rank"] = np.arange(1, len(order) + 1)
    w["rank_min4"] = np.nan
    o4 = w[el].sort_values(["min4", "sw_min"], ascending=[False, False]).index
    w.loc[o4, "rank_min4"] = np.arange(1, len(o4) + 1)
    w["rank_within_scope"] = np.nan
    for sc in SCOPES:
        idx = w[el & w.scope.eq(sc)].sort_values(["sw_min", "comp_min"], ascending=[False, False]).index
        w.loc[idx, "rank_within_scope"] = np.arange(1, len(idx) + 1)

    # Supplementary strict eligibility (see docstring): no completion loss whose CI excludes 0.
    for b in BASE:
        w[f"comp_ci_loss_{b}"] = comp_inf & (w[f"comp_diff_{b}_hi"] < 0)
    w["comp_ci_loss_any"] = w.comp_ci_loss_RGGP | w.comp_ci_loss_LPGP
    w["eligible_strict"] = el & ~w.comp_ci_loss_any
    w["rank_strict"] = np.nan
    ost = w[w.eligible_strict].sort_values(["sw_min", "comp_min"], ascending=[False, False]).index
    w.loc[ost, "rank_strict"] = np.arange(1, len(ost) + 1)

    rep, rep_val = replicate_check(w)
    w = w.join(rep)
    w["sw_rep_min"] = w[["sw_rep_diff_RGGP", "sw_rep_diff_LPGP"]].min(axis=1)
    w["comp_rep_min"] = w[["comp_rep_diff_RGGP", "comp_rep_diff_LPGP"]].min(axis=1)
    p90_sw = float(w.sw_noise_p90.dropna().iloc[0])
    w["rep_sw_min_gt_p90"] = w.sw_rep_min > p90_sw

    # Losses: every informative robust loss of Ours on switch success or completion (either comparison).
    loss_rows = []
    for r in w.itertuples():
        for s, lab in (("sw", "switch_success_pct"), ("comp", "completion_pct")):
            for b in BASE:
                inf = getattr(r, f"{s}_floor_ceiling") == "informative"
                if inf and getattr(r, f"{s}_robust_{b}") and getattr(r, f"{s}_diff_{b}") < 0:
                    loss_rows.append({
                        "scope": r.scope, "terrain": r.terrain, "difficulty": r.difficulty,
                        "speed_bin": r.speed_bin, "metric": lab, "vs": LAB[b],
                        "Ours": getattr(r, f"{s}_Ours"), "baseline": getattr(r, f"{s}_{b}"),
                        "diff": getattr(r, f"{s}_diff_{b}"), "diff_lo": getattr(r, f"{s}_diff_{b}_lo"),
                        "diff_hi": getattr(r, f"{s}_diff_{b}_hi"), "perm_p": getattr(r, f"{s}_perm_p_{b}"),
                        "holm220": getattr(r, f"{s}_holm220_{b}"), "n_robots": r.n_robots,
                        "n_lanes": r.n_lanes, "replicate_diff": getattr(r, f"{s}_rep_diff_{b}"),
                        "expected_mode": r.expected_mode, "n_roll_cells_in_group": r.n_roll_cells_in_group,
                        "mode_success_diff": getattr(r, f"mode_diff_{b}"),
                        "mode_success_diff_lo": getattr(r, f"mode_diff_{b}_lo"),
                        "mode_success_diff_hi": getattr(r, f"mode_diff_{b}_hi"),
                        "flags": r.flags})
    losses = pd.DataFrame(loss_rows).sort_values(["holm220", "diff"]).reset_index(drop=True)
    losses.insert(0, "loss_rank", np.arange(1, len(losses) + 1))

    # Chance level: same-policy null over the identical 220 groups.
    null = load_null()
    null = null[null.metric.isin(PRIMARY)]
    null["robust"] = as_bool(null["robust_Ours_rep"].fillna(False))
    null_counts = {}
    for metric in PRIMARY:
        sub = null[null.metric == metric]
        null_counts[metric] = {
            "rows": int(len(sub)),
            "robust_any_direction": int(sub.robust.sum()),
            "robust_Ours_better": int((sub.robust & (sub.diff_Ours_rep > 0)).sum()),
            "robust_Ours_worse": int((sub.robust & (sub.diff_Ours_rep < 0)).sum()),
            "by_scope": {sc: int(sub[sub.scope == sc].robust.sum()) for sc in SCOPES},
        }

    counts = {}
    for s, metric in (("sw", "switch_success_pct"), ("comp", "completion_pct")):
        inf = w[f"{s}_floor_ceiling"].eq("informative")
        for b in BASE:
            rob = inf & w[f"{s}_robust_{b}"]
            win, loss = rob & (w[f"{s}_diff_{b}"] > 0), rob & (w[f"{s}_diff_{b}"] < 0)
            h = w[f"{s}_holm220_{b}"] < 0.05
            counts[f"{metric} vs {LAB[b]}"] = {
                "examined": int(w[f"{s}_perm_p_{b}"].notna().sum()), "informative": int(inf.sum()),
                "robust_wins": int(win.sum()), "robust_losses": int(loss.sum()),
                "robust_wins_holm220_lt05": int((win & h).sum()), "robust_losses_holm220_lt05": int((loss & h).sum()),
                "nominal_chance_bound_per_direction": 0.025 * int(inf.sum()),
            }

    status_counts = w.status.value_counts().to_dict()

    # Output tables.
    first = ["rank", "rank_within_scope", "rank_min4", "rank_strict", "status", "eligible_strict",
             "scope", "terrain", "difficulty", "speed_bin",
             "n_robots", "n_lanes", "n_cells", "expected_mode", "n_roll_cells_in_group",
             "sw_min", "comp_min", "min4"]
    blocks = []
    for s in ("sw", "comp"):
        blocks += [f"{s}_Ours", f"{s}_Ours_lo", f"{s}_Ours_hi", f"{s}_RGGP", f"{s}_RGGP_lo", f"{s}_RGGP_hi",
                   f"{s}_LPGP", f"{s}_LPGP_lo", f"{s}_LPGP_hi"]
        for b in BASE:
            blocks += [f"{s}_diff_{b}", f"{s}_diff_{b}_lo", f"{s}_diff_{b}_hi", f"{s}_robust_{b}",
                       f"{s}_perm_p_{b}", f"{s}_holm220_{b}"]
        blocks += [f"{s}_floor_ceiling", f"{s}_noise_p90"]
    ctx = []
    for s in ("cot", "mode", "swdur", "rfo"):
        ctx += [f"{s}_Ours", f"{s}_RGGP", f"{s}_LPGP"]
        for b in BASE:
            ctx += [f"{s}_diff_{b}", f"{s}_diff_{b}_lo", f"{s}_diff_{b}_hi", f"{s}_outcome_{b}"]
    ctx += ["cot_holm220_RGGP", "cot_holm220_LPGP"]
    tail = ["sw_robust_win_both", "sw_holm_both_lt05", "comp_robust_loss_any", "comp_loss_holm_lt05",
            "comp_ci_loss_RGGP", "comp_ci_loss_LPGP", "comp_ci_loss_any",
            "comp_robust_win_n", "sw_Ours_rep", "sw_rep_diff_RGGP", "sw_rep_diff_LPGP", "sw_rep_min",
            "rep_sw_min_gt_p90", "comp_Ours_rep", "comp_rep_diff_RGGP", "comp_rep_diff_LPGP", "comp_rep_min",
            "flags"]
    out = w[first + blocks + ctx + tail].copy()
    stat_order = {"eligible": 0, "trade-off: switch win both, completion robust loss": 1,
                  "switch not robust vs both": 2, "switch robust loss": 3, "switch uninformative": 4,
                  "not ranked (replicate split)": 5}
    out["_s"] = out.status.map(stat_order)
    out = out.sort_values(["_s", "rank", "sw_min"], ascending=[True, True, False]).drop(columns="_s")
    out.to_csv(HERE / "favourable_settings.csv", index=False, float_format="%.6g")
    losses.to_csv(HERE / "unfavourable_settings.csv", index=False, float_format="%.6g")

    summary = {
        "n_candidates": int(len(w)), "n_by_scope": n_by_scope, "status_counts": status_counts,
        "counts_over_220": counts, "null_same_policy_over_220": null_counts,
        "switch_noise_p90": p90_sw, "replicate_check": rep_val,
        "n_eligible": int(el.sum()),
        "n_eligible_holm_both_lt05": int((el & w.sw_holm_both_lt05).sum()),
        "n_eligible_rep_sw_min_gt_p90": int((el & w.rep_sw_min_gt_p90).sum()),
        "n_eligible_with_ci_completion_loss": int((el & w.comp_ci_loss_any).sum()),
        "n_eligible_strict": int(w.eligible_strict.sum()),
        "n_losses_rows": int(len(losses)),
    }
    (HERE / "rank_settings.json").write_text(json.dumps(summary, indent=1, default=str))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    print(json.dumps(summary, indent=1, default=str))
    show = ["rank", "rank_within_scope", "rank_min4", "scope", "terrain", "difficulty", "speed_bin", "n_robots",
            "sw_Ours", "sw_RGGP", "sw_LPGP", "sw_diff_RGGP", "sw_diff_RGGP_lo", "sw_diff_RGGP_hi", "sw_holm220_RGGP",
            "sw_diff_LPGP", "sw_diff_LPGP_lo", "sw_diff_LPGP_hi", "sw_holm220_LPGP",
            "comp_Ours", "comp_RGGP", "comp_LPGP", "comp_diff_RGGP", "comp_diff_RGGP_lo", "comp_diff_RGGP_hi",
            "comp_robust_RGGP", "comp_holm220_RGGP", "comp_diff_LPGP", "comp_diff_LPGP_lo", "comp_diff_LPGP_hi",
            "comp_robust_LPGP", "comp_holm220_LPGP", "sw_rep_min", "comp_rep_min", "flags"]
    elig = out[out.status == "eligible"]
    with pd.option_context("display.float_format", "{:.4g}".format):
        print("\n=== eligible, top 25 overall ===")
        print(elig[show].head(25).to_string(index=False))
        for sc in SCOPES:
            print(f"\n=== eligible, top 5 within scope {sc} ===")
            print(elig[elig.scope == sc][show].head(5).to_string(index=False))
        print("\n=== eligible, top 10 by min4 ===")
        print(elig.sort_values("rank_min4")[show].head(10).to_string(index=False))
        print("\n=== trade-off rows (switch win both, completion robust loss), by sw_min ===")
        tr = out[out.status.str.startswith("trade-off")].sort_values("sw_min", ascending=False)
        print(tr[show].to_string(index=False))
        print("\n=== switch robust loss rows ===")
        print(out[out.status == "switch robust loss"][show].to_string(index=False))
        print("\n=== losses (all informative robust losses; sorted by Holm-220 p, then diff) ===")
        print(losses.to_string(index=False))


if __name__ == "__main__":
    main()
