"""Split-half (held-out lanes) check of the winner's curse in the favourable-settings ranking (rank_settings.py).

The ranking score is sw_min = min(Ours - w/o RG, Ours - w/o LP) on switch success, maximised over 220 overlapping
candidates. The rerun check in rank_settings.py replaces only Ours by its rerun, so selection noise in the two
baselines is not removed. Here the 16 lanes of every cell (independent terrain instances) are split into even and
odd lanes. The ranking is done on one half and the selected settings are evaluated on the other half, for all
three policies at once. No simulation, no bootstrap; point estimates only.

Selection sets: (a) all rankable candidates with informative switch success (random_rough cell / cell_speed rows are
replicate splits and not ranked, as in rank_settings.py); (b) only the 70 rows that are eligible on the full data
(favourable_settings.csv status). Top k = 5 and 10.

Reads analysis/per_traj.csv.gz, favourable_settings.csv. Writes analysis/split_half.json, analysis/split_half_rows.csv.

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/split_half.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
SPEEDS = ("[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0]")


def main() -> None:
    cols = ["terrain", "difficulty", "lane", "env", "policy", "speed_bin", "switch_success", "completed"]
    pt = pd.read_csv(HERE / "per_traj.csv.gz", usecols=cols)
    pt["dkey"] = pt.difficulty.map(lambda v: f"{float(v):g}")
    for c in ("switch_success", "completed"):
        pt[c] = pt[c].astype(str).str.lower().eq("true").astype(float) * 100.0
    pt["half"] = np.where(pt.lane % 2 == 0, "even", "odd")
    fav = pd.read_csv(HERE / "favourable_settings.csv", dtype={"difficulty": str})
    fav["difficulty"] = fav.difficulty.map(lambda v: v if v == "ALL" else f"{float(v):g}")
    key = ["scope", "terrain", "difficulty", "speed_bin"]

    recs = []
    for r in fav.itertuples():
        m = np.ones(len(pt), bool)
        if r.terrain != "ALL":
            m &= pt.terrain.to_numpy() == r.terrain
        if r.difficulty != "ALL":
            m &= pt.dkey.to_numpy() == r.difficulty
        if r.speed_bin != "ALL":
            m &= pt.speed_bin.to_numpy() == r.speed_bin
        sub = pt[m]
        rec = {k: getattr(r, k) for k in key}
        rec.update({"status": r.status, "rank": r.rank, "sw_min_full": r.sw_min,
                    "sw_informative": r.sw_floor_ceiling == "informative", "replicate_split":
                    r.status == "not ranked (replicate split)"})
        for h in ("even", "odd"):
            est = sub[sub.half == h].groupby("policy")[["switch_success", "completed"]].mean()
            rec[f"n_{h}"] = int((sub.half == h).sum() / 3)
            for b in ("RGGP", "LPGP"):
                rec[f"sw_diff_{b}_{h}"] = est.loc["Ours", "switch_success"] - est.loc[b, "switch_success"]
                rec[f"comp_diff_{b}_{h}"] = est.loc["Ours", "completed"] - est.loc[b, "completed"]
            rec[f"sw_min_{h}"] = min(rec[f"sw_diff_RGGP_{h}"], rec[f"sw_diff_LPGP_{h}"])
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(HERE / "split_half_rows.csv", index=False, float_format="%.6g")

    out = {"definition": "sw_min = min(Ours - w/o RG, Ours - w/o LP) switch success, percentage points; "
                         "halves = even / odd lane index within every cell"}
    pools = {"rankable_informative": df[~df.replicate_split & df.sw_informative],
             "eligible_full_data": df[df.status == "eligible"]}
    for pname, pool in pools.items():
        out[pname] = {"n_candidates": int(len(pool))}
        for k in (5, 10):
            res = {}
            for sel, ev in (("even", "odd"), ("odd", "even")):
                top = pool.sort_values(f"sw_min_{sel}", ascending=False).head(k)
                top_ev = set(pool.sort_values(f"sw_min_{ev}", ascending=False).head(k).index)
                res[f"select_{sel}_evaluate_{ev}"] = {
                    "mean_sw_min_selection_half": float(top[f"sw_min_{sel}"].mean()),
                    "mean_sw_min_heldout_half": float(top[f"sw_min_{ev}"].mean()),
                    "mean_shrinkage": float((top[f"sw_min_{sel}"] - top[f"sw_min_{ev}"]).mean()),
                    "max_shrinkage": float((top[f"sw_min_{sel}"] - top[f"sw_min_{ev}"]).max()),
                    "min_heldout_sw_min": float(top[f"sw_min_{ev}"].min()),
                    "overlap_with_heldout_topk": len(set(top.index) & top_ev),
                    "selected": [f"{r.scope}|{r.terrain}|{r.difficulty}|{r.speed_bin}" for r in top.itertuples()],
                }
            out[pname][f"top{k}"] = res
    # The published Table A top 10 (full-data ranking): values in each half.
    t10 = df[df.status == "eligible"].sort_values("rank").head(10)
    out["table_A_top10_by_half"] = [
        {"rank": int(r.rank), "row": f"{r.scope}|{r.terrain}|{r.difficulty}|{r.speed_bin}",
         "sw_min_full": round(float(r.sw_min_full), 2), "sw_min_even": round(float(r.sw_min_even), 2),
         "sw_min_odd": round(float(r.sw_min_odd), 2), "n_even": int(r.n_even), "n_odd": int(r.n_odd)}
        for r in t10.itertuples()]
    (HERE / "split_half.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
