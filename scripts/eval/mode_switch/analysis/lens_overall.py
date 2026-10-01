"""OVERALL lens of mode_switch_v2: pooled / per-difficulty / per-terrain results and the comparison with
the previous mode_switch experiment (logs/lipm_eval/mode_switch, full/results/unified_table.json).

Inputs (read only): analysis/cells.csv, analysis/per_traj.csv.gz, analysis/per_traj_replicate.csv.gz,
analysis/null_replicate_cells.csv, noise_floor.json, ../mode_switch/summary.json,
../full/results/unified_table.json. Statistics reuse analysis/tidy.py (Data, group_stats: lane bootstrap
within (terrain, difficulty) cells, 2000 reps, fixed seeds; lane sign-flip permutation p).

Writes (analysis/ only):
  lens_overall_groups.csv          all / difficulty / terrain / speed_bin rows of cells.csv (selected columns)
  lens_overall_prev_compare.csv    previous experiment vs v2 subsets (matched, chain, single-factor restrictions)
  lens_overall_replicate_pooled.csv  Ours - Ours rerun at pooled scopes (run-to-run difference, supplementary)
  lens_overall.json                counts of robust wins / losses, Holm, chance expectation, self-checks
  figures/lens_overall_forest.png, figures/lens_overall_prev_vs_v2.png

Pre-specified robust rule: 95 % lane-bootstrap CI of the paired difference excludes 0 AND |difference| >
p90 run-to-run noise of the same metric in noise_floor.json (cell level); NA if noise_floor.json has no
matching metric.

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/lens_overall.py
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
RUN = HERE.parent
PREV = RUN.parent / "mode_switch"
UNIFIED = RUN.parent / "full/results/unified_table.json"
FIG = HERE / "figures"

spec = importlib.util.spec_from_file_location("tidy_mod", Path(__file__).resolve().parent / "tidy.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)  # tidy.main() is guarded; only helpers and constants are loaded.

POL = ("Ours", "RGGP", "LPGP")
LABEL = {"Ours": "Ours", "RGGP": "Ours w/o RG", "LPGP": "Ours w/o LP"}
BASE = ("RGGP", "LPGP")
PRIMARY = ("switch_success_pct", "completion_pct", "cot")
REPORT = PRIMARY + ("failure_pct", "roll_flat_in_pct", "step_rough_pct", "roll_flat_out_pct", "flat_lift_rate_pct",
                    "rough_lift_rate_pct", "peak_tilt_deg", "time_ratio", "mode_success_pct", "switch_success_dur_pct")
OLD3 = ("tilted_grid", "random_spread", "discrete_obstacles")  # terrains of the previous experiment
PREV_V = (0.6, 1.0)  # previous command range (forward: v_x ~ U(0.6, 1.0))
BINARY = [col for _, col, kind, *_ in T.METRICS if kind == "binary"]
COL = {n: c for n, c, *_ in T.METRICS}
IDX = {n: j for j, n in enumerate(T.NAMES)}
COLORS = {"Ours": "#2a78d6", "RGGP": "#eb6834", "LPGP": "#1baf7a"}  # categorical slots 1-3 (all-pairs valid)
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"

warnings.simplefilter("ignore", RuntimeWarning)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={c: "object" for c in BINARY})
    for c in BINARY:  # 'True' / 'False' / missing (window not traversed) -> 1 / 0 / NaN
        s = df[c].astype("object")
        bad = ~s.isin(["True", "False"]) & s.notna()
        assert not bad.any(), (c, s[bad].unique()[:5])
        df[c] = s.map({"True": 1.0, "False": 0.0}).astype(float)
    return df


def noise_p90() -> dict:
    nf = json.loads((RUN / "noise_floor.json").read_text())
    return {m: v["p90_abs_diff"] for m, v in nf["summary_over_cells"].items()}


def stats_rows(data: T.Data, idx: np.ndarray, key: str, meta: dict, nf: dict, pols=POL) -> list[dict]:
    """One row per metric: estimates + CIs per policy, paired differences pols[0] - other, robust flag."""
    res = T.group_stats(data, idx, key)
    rows = []
    for name in REPORT:
        j = IDX[name]
        row = {**meta, "metric": name, "n_robots": res["n_robots"], "n_lanes": res["n_lanes"],
               "n_cells": res["n_cells"]}
        for p in pols:
            lo, hi = T.pct(res["boot"][p][:, [j]])
            row[p], row[f"{p}_lo"], row[f"{p}_hi"] = res["est"][p][j], lo[0], hi[0]
            row[f"{p}_n"] = int(res["n_valid"][p][j])
        for b in pols[1:]:
            e = res["diff"][b]
            est, lo, hi = e["est"][j], e["lo"][j], e["hi"][j]
            excl = bool(lo > 0 or hi < 0)
            row[f"diff_{b}"], row[f"diff_{b}_lo"], row[f"diff_{b}_hi"] = est, lo, hi
            row[f"perm_p_{b}"], row[f"perm_method_{b}"] = e["p"][j], e["method"]
            row[f"ci_excl0_{b}"] = excl
            row[f"robust_{b}"] = (excl and abs(est) > nf[name]) if name in nf else None
        row["noise_p90"] = nf.get(name, np.nan)
        rows.append(row)
    return rows


def fmt_ci(v, lo, hi, d=1):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main() -> int:
    FIG.mkdir(exist_ok=True)
    nf = noise_p90()
    cells = pd.read_csv(HERE / "cells.csv")
    null = pd.read_csv(HERE / "null_replicate_cells.csv")
    out: dict = {"noise_floor_json_p90": nf}

    # ------------------------------------------------------------------ 1. pooled rows of cells.csv
    keep = ["scope", "terrain", "difficulty", "speed_bin", "metric", "n_robots", "n_lanes", "n_cells"]
    for p in POL:
        keep += [p, f"{p}_lo", f"{p}_hi", f"{p}_n"]
    for b in BASE:
        keep += [f"diff_{b}", f"diff_{b}_lo", f"diff_{b}_hi", f"ci_excl0_{b}", f"perm_p_{b}", f"perm_p_holm_{b}",
                 f"holm_m_{b}", f"robust_{b}", f"outcome_{b}"]
    keep += ["floor_ceiling", "noise_metric", "noise_p90"]
    pooled = cells[cells.scope.isin(["all", "difficulty", "terrain", "speed_bin"]) & cells.metric.isin(REPORT)][keep]
    pooled.to_csv(HERE / "lens_overall_groups.csv", index=False, float_format="%.6g")

    def show(scope, metrics, label_col):
        sub = pooled[(pooled.scope == scope) & pooled.metric.isin(metrics)]
        for m in metrics:
            d = 3 if m in ("cot", "time_ratio") else 1
            print(f"\n[{scope}] {m}")
            for _, r in sub[sub.metric == m].iterrows():
                s = " | ".join(fmt_ci(r[p], r[f"{p}_lo"], r[f"{p}_hi"], d) for p in POL)
                dd = " | ".join(f"{b}: {fmt_ci(r[f'diff_{b}'], r[f'diff_{b}_lo'], r[f'diff_{b}_hi'], d)} "
                                f"p={r[f'perm_p_{b}']:.2g} holm={r[f'perm_p_holm_{b}']:.2g} {r[f'outcome_{b}']}"
                                for b in BASE)
                print(f"  {str(r[label_col]):22s} {s} || {dd} ({r.floor_ceiling})")

    show("all", REPORT, "scope")
    show("difficulty", PRIMARY + ("mode_success_pct", "step_rough_pct", "roll_flat_out_pct"), "difficulty")
    show("terrain", PRIMARY + ("mode_success_pct",), "terrain")
    show("speed_bin", ("switch_success_pct", "completion_pct", "cot", "step_rough_pct", "roll_flat_out_pct",
                       "flat_lift_rate_pct", "mode_success_pct", "switch_success_dur_pct"), "speed_bin")

    # robust win / loss counts per scope and metric, with Holm and a chance expectation
    counts = {}
    for scope in ("difficulty", "terrain", "speed_bin"):
        for m in PRIMARY + ("mode_success_pct", "step_rough_pct", "roll_flat_out_pct", "failure_pct"):
            sub = pooled[(pooled.scope == scope) & (pooled.metric == m)]
            for b in BASE:
                oc = sub[f"outcome_{b}"]
                nullsub = null[(null.scope == "cell") & (null.metric == m)]
                null_rate = (float(nullsub.robust_Ours_rep.astype(str).eq("True").mean())
                             if nullsub.noise_metric.notna().all() else None)
                counts[f"{scope}|{m}|{b}"] = {
                    "m": int(len(sub)), "ours_better": int((oc == "Ours better").sum()),
                    "ours_worse": int((oc == "Ours worse").sum()), "not_robust": int((oc == "not robust").sum()),
                    "no_noise_floor": int((oc == "no_noise_floor").sum()),
                    "ci_excl0": int(sub[f"ci_excl0_{b}"].sum()),
                    "holm_lt_0.05": int((sub[f"perm_p_holm_{b}"] < 0.05).sum()),
                    "null_cell_robust_rate": null_rate,
                    "expected_false_robust": None if null_rate is None else null_rate * len(sub),
                    "expected_ci_excl0_by_chance(5%)": 0.05 * len(sub),
                }
    out["robust_counts"] = counts
    print("\nrobust counts (scope|metric|baseline): better / worse / not robust | Holm<0.05 | expected false robust")
    for k, v in counts.items():
        print(f"  {k:45s} {v['ours_better']:2d} / {v['ours_worse']:2d} / {v['not_robust']:2d} (m={v['m']}) "
              f"| holm<.05 {v['holm_lt_0.05']} | exp.false {v['expected_false_robust']}")

    # ------------------------------------------------------------------ 2. per-trajectory statistics
    df = load(HERE / "per_traj.csv.gz")
    data = T.Data(df)
    ref = df[df.ckpt == "Ours"].sort_values(["rough", "env"]).reset_index(drop=True)
    assert np.array_equal(ref.env.to_numpy(), data.meta.env.to_numpy())
    assert np.array_equal(ref.rough.to_numpy(), data.meta.rough.to_numpy())
    v, t, dlev = ref.cmd_speed.to_numpy(), ref.rough.to_numpy(), ref.difficulty.to_numpy()

    # self-check: the 'all' group recomputed here equals cells.csv (same key -> same seed)
    chk = T.group_stats(data, np.arange(len(ref)), T.gkey("all", "ALL", "ALL", "ALL"))
    dev = []
    for name in REPORT:
        row = cells[(cells.scope == "all") & (cells.metric == name)].iloc[0]
        j = IDX[name]
        for p in POL:
            lo, hi = T.pct(chk["boot"][p][:, [j]])
            dev += [abs(chk["est"][p][j] - row[p]), abs(lo[0] - row[f"{p}_lo"]), abs(hi[0] - row[f"{p}_hi"])]
        for b in BASE:
            e = chk["diff"][b]
            dev += [abs(e["est"][j] - row[f"diff_{b}"]), abs(e["lo"][j] - row[f"diff_{b}_lo"]),
                    abs(e["hi"][j] - row[f"diff_{b}_hi"])]
    out["selfcheck_all_group_vs_cells_csv_max_abs_dev"] = float(np.nanmax(dev))
    print("\nself-check vs cells.csv (all group) max |dev|:", out["selfcheck_all_group_vs_cells_csv_max_abs_dev"])

    # ------------------------------------------------------------------ 3. previous experiment
    uni = json.loads(UNIFIED.read_text())
    prev_sum = json.loads((PREV / "summary.json").read_text())
    prev = {"unified": {k: uni[k]["mean"] for k in ("switch", "completion", "cot")},
            "unified_p_vs_Ours": {k: uni[k]["p_vs_Ours"] for k in ("switch", "completion", "cot")}}
    per_t = {}
    for m in ("switch_success_pct", "completion_pct", "cot", "roll_flat_in_pct", "step_rough_pct",
              "roll_flat_out_pct", "flat_lift_rate_pct", "peak_tilt_deg"):
        per_t[m] = {p: {tt: prev_sum[f"{tt}|forward|{p}"][m] for tt in OLD3} for p in POL}
    prev["summary_per_terrain"] = per_t
    prev["summary_terrain_mean"] = {m: {p: float(np.mean(list(per_t[m][p].values()))) for p in POL} for m in per_t}
    out["previous"] = prev
    print("\nprevious (unified_table.json means):", json.dumps(prev["unified"], indent=None))
    print("previous (summary.json, mean of 3 terrains):",
          {m: {p: round(x, 3) for p, x in d.items()} for m, d in prev["summary_terrain_mean"].items()})

    # ------------------------------------------------------------------ 4. v2 subsets
    vm = (v >= PREV_V[0]) & (v <= PREV_V[1])
    old = np.isin(t, OLD3)
    d1 = dlev == 1.0
    stairs = np.isin(t, ("pyramid_stair", "pyramid_stair_inv", "random_stairs"))
    slopes = np.isin(t, ("hf_pyramid_slope", "hf_pyramid_slope_inv"))
    subsets = [
        # (id, description, mask)
        ("B", "v2 matched: 3 old terrains, d=1, v in [0.6,1.0]", old & d1 & vm),
        ("C", "v2: 3 old terrains, d=1, all v", old & d1),
        ("D", "v2: 3 old terrains, all d, all v", old),
        ("E", "v2: all 10 terrains, all d, all v (overall)", np.ones(len(t), bool)),
        ("S1", "v2: all terrains, all d, v in [0.6,1.0]", vm),
        ("S2", "v2: all terrains, d=1, all v", d1),
        ("S3", "v2: 7 new terrains, all d, all v", ~old),
        ("S4", "v2: 3 old terrains, all d, v in [0.6,1.0]", old & vm),
        ("S5", "v2: 7 new terrains, d=1, v in [0.6,1.0]", ~old & d1 & vm),
        ("G1", "v2: stairs (pyramid_stair, _inv, random_stairs), all d, all v", stairs),
        ("G2", "v2: hf slopes (hf_pyramid_slope, _inv), all d, all v", slopes),
        ("G3", "v2: random_rough + stepping_stones, all d, all v", np.isin(t, ("random_rough", "stepping_stones"))),
    ]
    for tt in OLD3:
        subsets.append((f"B_{tt}", f"v2 matched: {tt}, d=1, v in [0.6,1.0]", (t == tt) & d1 & vm))
    rows = []
    for sid, desc, mask in subsets:
        rows += stats_rows(data, np.flatnonzero(mask), f"lens_overall|{sid}", {"subset": sid, "description": desc}, nf)

    # run-to-run difference inside the same subsets (Ours - Ours rerun), supplementary
    rep = load(HERE / "per_traj_replicate.csv.gz")
    both = pd.concat([df[df.ckpt == "Ours"].assign(ckpt="Ours"), rep.assign(ckpt="Ours_rep")], ignore_index=True)
    ndata = T.Data(both, policies=("Ours", "Ours_rep"))
    assert np.array_equal(ndata.meta.env.to_numpy(), data.meta.env.to_numpy())
    assert np.array_equal(ndata.meta.rough.to_numpy(), data.meta.rough.to_numpy())
    rep_by_subset = {}
    for sid, desc, mask in subsets:
        rr = stats_rows(ndata, np.flatnonzero(mask), f"lens_overall|{sid}|rep", {"subset": sid}, nf,
                        pols=("Ours", "Ours_rep"))
        rep_by_subset[sid] = {r["metric"]: r for r in rr}
    for r in rows:
        rr = rep_by_subset[r["subset"]][r["metric"]]
        r["rep_diff_Ours_minus_rerun"] = rr["diff_Ours_rep"]
        r["rep_diff_lo"], r["rep_diff_hi"] = rr["diff_Ours_rep_lo"], rr["diff_Ours_rep_hi"]
    comp = pd.DataFrame(rows)
    # previous experiment rows (point estimates only: the old rollouts store no lane / cluster ids)
    prow = []
    for name, key in (("switch_success_pct", "switch"), ("completion_pct", "completion"), ("cot", "cot")):
        r = {"subset": "A", "description": "previous (unified_table.json): 3 old terrains, d=1, v in [0.6,1.0], "
                                          "4.4 m exit flat, 1024 per terrain", "metric": name, "n_robots": 3072}
        for p in POL:
            r[p] = prev["unified"][key][p]
        for b in BASE:
            r[f"diff_{b}"] = r["Ours"] - r[b]
            r[f"perm_p_{b}"] = prev["unified_p_vs_Ours"][key][b]
            r[f"perm_method_{b}"] = "paired test of unified_table (McNemar / Wilcoxon, robot level)"
        prow.append(r)
    for name in ("roll_flat_in_pct", "step_rough_pct", "roll_flat_out_pct", "flat_lift_rate_pct", "peak_tilt_deg"):
        r = {"subset": "A", "description": "previous (summary.json, mean of 3 terrains)", "metric": name,
             "n_robots": 3072}
        for p in POL:
            r[p] = prev["summary_terrain_mean"][name][p]
        for b in BASE:
            r[f"diff_{b}"] = r["Ours"] - r[b]
        prow.append(r)
    comp = pd.concat([pd.DataFrame(prow), comp], ignore_index=True)
    comp.to_csv(HERE / "lens_overall_prev_compare.csv", index=False, float_format="%.6g")

    print("\nprevious vs v2 subsets (Ours | w/o RG | w/o LP ; Ours-RGGP ; Ours-LPGP ; rerun diff)")
    for m in ("switch_success_pct", "completion_pct", "cot", "step_rough_pct", "roll_flat_out_pct",
              "flat_lift_rate_pct"):
        print(f"\n  {m}")
        d = 3 if m == "cot" else 1
        for _, r in comp[comp.metric == m].iterrows():
            if r.subset == "A":
                print(f"   {r.subset:6s} n={r.n_robots:5d} {r.Ours:.{d}f} | {r.RGGP:.{d}f} | {r.LPGP:.{d}f} ; "
                      f"{r.diff_RGGP:+.{d}f} ; {r.diff_LPGP:+.{d}f}   {r.description}")
                continue
            print(f"   {r.subset:6s} n={int(r.n_robots):5d} "
                  + " | ".join(fmt_ci(r[p], r[f"{p}_lo"], r[f"{p}_hi"], d) for p in POL)
                  + " ; " + " ; ".join(f"{fmt_ci(r[f'diff_{b}'], r[f'diff_{b}_lo'], r[f'diff_{b}_hi'], d)} "
                                        f"p={r[f'perm_p_{b}']:.2g} R={r[f'robust_{b}']}" for b in BASE)
                  + f" ; rerun {r.rep_diff_Ours_minus_rerun:+.{d}f}   {r.description}")

    # ------------------------------------------------------------------ 5. pooled run-to-run differences
    rep_rows = []
    groups = [("all", "ALL", "ALL", "ALL", np.ones(len(t), bool))]
    groups += [("difficulty", "ALL", dd, "ALL", dlev == dd) for dd in sorted(set(dlev))]
    groups += [("terrain", tt, "ALL", "ALL", t == tt) for tt in sorted(set(t))]
    groups += [("speed_bin", "ALL", "ALL", sb, ref.speed_bin.to_numpy() == sb) for sb in T.SPEED_LABELS]
    for scope, tt, dd, sb, mask in groups:
        rep_rows += stats_rows(ndata, np.flatnonzero(mask), f"lens_overall|rep|{scope}|{tt}|{dd}|{sb}",
                               {"scope": scope, "terrain": tt, "difficulty": dd, "speed_bin": sb}, nf,
                               pols=("Ours", "Ours_rep"))
    repdf = pd.DataFrame(rep_rows)
    repdf.to_csv(HERE / "lens_overall_replicate_pooled.csv", index=False, float_format="%.6g")
    rep_summary = {}
    for scope in ("all", "difficulty", "terrain", "speed_bin"):
        for m in PRIMARY + ("failure_pct", "step_rough_pct", "roll_flat_out_pct", "mode_success_pct"):
            s = repdf[(repdf.scope == scope) & (repdf.metric == m)]
            rep_summary[f"{scope}|{m}"] = {"n_groups": int(len(s)),
                                          "max_abs_diff": float(s.diff_Ours_rep.abs().max()),
                                          "ci_excl0": int(s.ci_excl0_Ours_rep.sum()),
                                          "robust_rule": int(s.robust_Ours_rep.fillna(False).astype(bool).sum())}
    out["replicate_pooled"] = rep_summary
    print("\nOurs - Ours rerun at pooled scopes: max |diff| (CI excl 0 count / robust count)")
    for k, s in rep_summary.items():
        print(f"  {k:36s} max|d|={s['max_abs_diff']:.4g} (n={s['n_groups']}, CI excl0 {s['ci_excl0']}, "
              f"robust {s['robust_rule']})")

    # ------------------------------------------------------------------ 6. figures
    forest(pooled, nf)
    prev_fig(comp, pooled, prev)
    (HERE / "lens_overall.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote lens_overall_groups.csv, lens_overall_prev_compare.csv, lens_overall_replicate_pooled.csv, "
          "lens_overall.json, figures/lens_overall_forest.png, figures/lens_overall_prev_vs_v2.png")
    return 0


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8.5, width=0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def forest(pooled: pd.DataFrame, nf: dict) -> None:
    """Paired differences Ours - baseline (95 % CI) for the primary metrics at the pooled scopes."""
    order = [("all", "ALL", "all 40 cells")]
    order += [None] + [("difficulty", f"{d}", f"d = {d:g}") for d in (0.25, 0.5, 0.75, 1.0)]
    terr = ["tilted_grid", "random_spread", "discrete_obstacles", "random_rough", "stepping_stones", "pyramid_stair",
            "pyramid_stair_inv", "random_stairs", "hf_pyramid_slope", "hf_pyramid_slope_inv"]
    order += [None] + [("terrain", tt, tt + (" *" if tt in OLD3 else "")) for tt in terr]
    order += [None] + [("speed_bin", sb, f"v {sb}") for sb in T.SPEED_LABELS]
    titles = {"switch_success_pct": "Switch success, Ours - baseline (pp)\n(> 0: Ours better)",
              "completion_pct": "Completion, Ours - baseline (pp)\n(> 0: Ours better)",
              "cot": "CoT, Ours - baseline\n(< 0: Ours better)"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 8.6), sharey=True)
    ypos, labels, y = [], [], 0.0
    for item in order:
        if item is None:
            y += 0.6
            continue
        ypos.append(y)
        labels.append(item[2])
        y += 1.0
    for ax, m in zip(axes, PRIMARY):
        style(ax)
        band = nf[m]
        ax.axvspan(-band, band, color=BAND, zorder=0, lw=0)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        k = 0
        for item in order:
            if item is None:
                continue
            scope, key, _ = item
            sub = pooled[(pooled.scope == scope) & (pooled.metric == m)]
            if scope == "difficulty":
                sub = sub[sub.difficulty.astype(str).astype(float) == float(key)]
            elif scope == "terrain":
                sub = sub[sub.terrain == key]
            elif scope == "speed_bin":
                sub = sub[sub.speed_bin == key]
            r = sub.iloc[0]
            for off, b, marker in ((-0.17, "RGGP", "o"), (0.17, "LPGP", "s")):
                yy = ypos[k] + off
                ax.plot([r[f"diff_{b}_lo"], r[f"diff_{b}_hi"]], [yy, yy], color=COLORS[b], lw=1.6, zorder=2,
                        solid_capstyle="round")
                robust = str(r[f"robust_{b}"]) == "True"
                ax.plot(r[f"diff_{b}"], yy, marker=marker, ms=6.5, mec=COLORS[b], mew=1.4,
                        mfc=COLORS[b] if robust else "white", zorder=3, ls="none")
            k += 1
        ax.set_title(titles[m], fontsize=10, color=INK, loc="left")
        lo, hi = ax.get_xlim()
        ax.text(0.99, 0.004, f"grey band: ±p90 cell-level run-to-run noise = {band:.3g}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=INK2)
    axes[0].set_yticks(ypos, labels, fontsize=8.5)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS["RGGP"], marker="o", mfc=COLORS["RGGP"], lw=1.6, label="vs Ours w/o RG"),
               Line2D([], [], color=COLORS["LPGP"], marker="s", mfc=COLORS["LPGP"], lw=1.6, label="vs Ours w/o LP"),
               Line2D([], [], color=INK2, marker="o", mfc=INK2, lw=0, label="robust (CI excl. 0 and |diff| > p90)"),
               Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, label="not robust")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("mode_switch_v2: paired differences with 95 % lane-bootstrap CI (n = 1 training seed per policy; "
                 "* = terrain of the previous experiment)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / "lens_overall_forest.png", dpi=150)
    plt.close(fig)


def prev_fig(comp: pd.DataFrame, pooled: pd.DataFrame, prev: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.45, 1, 1]})
    # (a) switch success along the chain previous -> v2 overall
    chain = [("A", "previous\n3 terr., d=1\nv 0.6-1.0"), ("B", "v2 matched\n3 terr., d=1\nv 0.6-1.0"),
             ("C", "v2\n3 terr., d=1\nv 0.5-2.0"), ("D", "v2\n3 terr., d 0.25-1\nv 0.5-2.0"),
             ("E", "v2 overall\n10 terr., d 0.25-1\nv 0.5-2.0")]
    ax = axes[0]
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    sw = comp[comp.metric == "switch_success_pct"].set_index("subset")
    for k, (sid, _) in enumerate(chain):
        r = sw.loc[sid]
        for off, p in zip((-0.2, 0.0, 0.2), POL):
            x = k + off
            if sid != "A":
                ax.plot([x, x], [r[f"{p}_lo"], r[f"{p}_hi"]], color=COLORS[p], lw=1.6, solid_capstyle="round")
            ax.plot(x, r[p], "o", color=COLORS[p], ms=7, mec="white", mew=1.0)
        ax.text(k, 104, f"-RG {r['diff_RGGP']:+.1f}\n-LP {r['diff_LPGP']:+.1f}", ha="center",
                va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(chain)), [c[1] for c in chain], fontsize=8)
    ax.set_ylabel("switch success (%)", fontsize=9, color=INK2)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("(a) switch success: previous experiment -> v2 subsets\n(top: Ours - w/o RG / Ours - w/o LP, pp)",
                 fontsize=10, loc="left", color=INK)
    ax.axvline(0.5, color=INK2, lw=0.6)
    # (b), (c): speed bins over all 40 cells
    for ax, m, title in ((axes[1], "switch_success_pct", "(b) switch success by commanded speed (40 cells)"),
                         (axes[2], "roll_flat_out_pct", "(c) roll_flat_out (no lift in flat-out), by speed")):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        sub = pooled[(pooled.scope == "speed_bin") & (pooled.metric == m)].set_index("speed_bin").loc[
            list(T.SPEED_LABELS)]
        xs = np.arange(len(sub))
        for off, p in zip((-0.05, 0.0, 0.05), POL):
            ax.errorbar(xs + off, sub[p], yerr=[sub[p] - sub[f"{p}_lo"], sub[f"{p}_hi"] - sub[p]], color=COLORS[p],
                        lw=1.8, marker="o", ms=7, mec="white", mew=1.0, capsize=0, label=LABEL[p])
            if m == "switch_success_pct":
                ax.annotate(LABEL[p], (xs[-1], sub[p].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
        ax.set_xticks(xs, [s.replace(",", " - ") for s in sub.index], fontsize=8.5)
        ax.set_xlabel("commanded v_x (m/s)", fontsize=9, color=INK2)
        ax.set_xlim(-0.3, len(xs) - 0.3 + (0.9 if m == "switch_success_pct" else 0))
        ax.set_ylim(0, 105)
        ax.set_ylabel("% of robots" if m == "switch_success_pct" else "% of completing robots without a lift event",
                      fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    handles = [plt.Line2D([], [], color=COLORS[p], marker="o", lw=1.8, label=LABEL[p]) for p in POL]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Why the switch-success gap differs from the previous experiment (bars: 95 % lane-bootstrap CI; "
                 "previous run: point estimates, no lane ids stored)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(FIG / "lens_overall_prev_vs_v2.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
