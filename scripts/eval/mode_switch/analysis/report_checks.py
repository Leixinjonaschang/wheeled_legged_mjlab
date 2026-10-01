"""Consistency checks for REPORT.md (reads existing analysis outputs only; no simulation, no new bootstrap).

1. lens_mode.py uses its own bootstrap seed (20261001); cells.csv (tidy.py) uses 20260930. Point estimates and
   permutation / Holm p agree, CI bounds differ by Monte-Carlo error. This prints the maximum bound difference
   and the cells.csv values of every cell-level quantity the mode section of REPORT.md quotes, so that the
   report uses one source (cells.csv) wherever the same quantity appears in several sections.
2. Duration-only switch success vs w/o RG / w/o LP over the 40 cells with the cells.csv intervals (CI-based
   counts, Holm counts, 3 cm -> duration-only verdict flips).

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/report_checks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script


def dkey(v) -> str:
    return f"{float(v):g}"


def main() -> None:
    cells = pd.read_csv(HERE / "cells.csv")
    cells = cells[cells.scope == "cell"].copy()
    cells["d"] = cells.difficulty.map(dkey)
    cm = cells.set_index(["terrain", "d", "metric"])

    lm = pd.read_csv(HERE / "lens_mode_groups.csv")
    lm = lm[lm.scope == "cell"].copy()
    lm[["terrain", "d"]] = lm.group.str.split("|", expand=True)
    lm["d"] = lm.d.map(dkey)
    both = lm.merge(cells, on=["terrain", "d", "metric"], suffixes=("_lm", "_cells"))
    print("== 1. lens_mode_groups.csv vs cells.csv (cell scope) ==")
    for metric, g in both.groupby("metric"):
        pt = max((g[f"diff_{b}_lm"] - g[f"diff_{b}_cells"]).abs().max() for b in ("RGGP", "LPGP"))
        ci = max((g[f"diff_{b}_{e}_lm"] - g[f"diff_{b}_{e}_cells"]).abs().max()
                 for b in ("RGGP", "LPGP") for e in ("lo", "hi"))
        hp = max((g[f"perm_p_holm_{b}_lm"] - g[f"perm_p_holm_{b}_cells"]).abs().max() for b in ("RGGP", "LPGP"))
        print(f"  {metric:24s} max|point diff| {pt:.2g}  max|CI bound diff| {ci:.3f}  max|Holm p diff| {hp:.2g}")

    def show(t, d, metric, cols=("RGGP", "LPGP")):
        r = cm.loc[(t, d, metric)]
        s = f"  {t} d={d} {metric}: O/R/L {r.Ours:.2f}/{r.RGGP:.2f}/{r.LPGP:.2f}"
        for b in cols:
            s += (f" | Ours-{b} {r[f'diff_{b}']:+.2f} [{r[f'diff_{b}_lo']:.2f},{r[f'diff_{b}_hi']:.2f}]"
                  f" Holm {r[f'perm_p_holm_{b}']:.3g} robust {r[f'robust_{b}']} {r.floor_ceiling}")
        print(s)

    print("\n== cells.csv values of quantities quoted in the mode section ==")
    for t, d in (("pyramid_stair", "0.25"), ("pyramid_stair_inv", "0.25"), ("hf_pyramid_slope", "0.75"),
                 ("hf_pyramid_slope", "0.5")):
        show(t, d, "switch_success_pct")
        show(t, d, "mode_success_pct")
    show("pyramid_stair", "1", "roll_flat_in_pct")
    for t, d in (("hf_pyramid_slope_inv", "1"), ("tilted_grid", "0.25")):
        show(t, d, "mode_success_pct")

    print("\n== 2. duration-only switch success (cells.csv, 40 cells) ==")
    sw = cells[cells.metric == "switch_success_pct"].set_index(["terrain", "d"])
    du = cells[cells.metric == "switch_success_dur_pct"].set_index(["terrain", "d"])
    for b in ("RGGP", "LPGP"):
        inf = du.floor_ceiling.eq("informative")
        excl = du[f"ci_excl0_{b}"].astype(str).eq("True")
        win, loss = inf & excl & (du[f"diff_{b}"] > 0), inf & excl & (du[f"diff_{b}"] < 0)
        h = du[f"perm_p_holm_{b}"] < 0.05
        print(f"  vs {b}: informative {int(inf.sum())}, CI wins {int(win.sum())}, CI losses {int(loss.sum())}, "
              f"Holm<0.05 wins {int((win & h).sum())}, losses {int((loss & h).sum())}")
        s_win = sw[f"ci_excl0_{b}"].astype(str).eq("True") & (sw[f"diff_{b}"] > 0)
        s_rob = sw[f"robust_{b}"].astype(str).eq("True") & (sw[f"diff_{b}"] > 0)
        flips = du[(s_rob | s_win) & loss.reindex(sw.index)]
        print(f"    3 cm CI-win or robust-win cells that are CI losses duration-only: {len(flips)}")
        for (t, d), r in flips.iterrows():
            s3 = sw.loc[(t, d)]
            print(f"      {t} d={d}: 3cm {s3[f'diff_{b}']:+.1f} (robust {s3[f'robust_{b}']}) -> dur "
                  f"{r[f'diff_{b}']:+.1f} [{r[f'diff_{b}_lo']:.1f},{r[f'diff_{b}_hi']:.1f}] Holm {r[f'perm_p_holm_{b}']:.3g}")
        border = du[inf & ~excl & (du[f"diff_{b}"] < 0) & ((du[f"diff_{b}_hi"] == 0) | (du[f"diff_{b}_lo"] == 0))]
        for (t, d), r in border.iterrows():
            print(f"    borderline (CI bound exactly 0): {t} d={d} dur {r[f'diff_{b}']:+.1f} "
                  f"[{r[f'diff_{b}_lo']:.1f},{r[f'diff_{b}_hi']:.1f}] Holm {r[f'perm_p_holm_{b}']:.3g}")

    allrow = pd.read_csv(HERE / "cells.csv")
    allrow = allrow[allrow.scope == "all"].set_index("metric")
    for m in ("mode_success_pct", "switch_success_dur_pct"):
        r = allrow.loc[m]
        print(f"  all 40 cells {m}: {r.Ours:.1f}/{r.RGGP:.1f}/{r.LPGP:.1f}, Ours-RG {r.diff_RGGP:+.1f} "
              f"[{r.diff_RGGP_lo:.1f},{r.diff_RGGP_hi:.1f}], Ours-LP {r.diff_LPGP:+.1f} [{r.diff_LPGP_lo:.1f},{r.diff_LPGP_hi:.1f}]")
    _ = np  # numpy imported for parity with the other analysis scripts


if __name__ == "__main__":
    main()
