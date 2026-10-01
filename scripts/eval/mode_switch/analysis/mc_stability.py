"""Monte-Carlo stability of the robust / Holm verdicts used in REPORT.md (switch success, completion).

Why: the robust flag (95 % lane-bootstrap CI excludes 0 AND |diff| > noise_floor.json p90) depends on the bootstrap
draw when a CI bound sits at or near 0 (binary outcomes put atoms on 0), and the pooled rows (> 16 lanes) used
10000 Monte-Carlo sign flips, so p >= 1/10001 and Holm-220 multiplies that resolution limit by up to 220.

What (no simulation; reads analysis/per_traj.csv.gz, per_traj_replicate.csv.gz, favourable_settings.csv,
unfavourable_settings.csv, noise_floor.json; writes analysis/mc_stability.json, analysis/mc_stability_rows.csv):
  1. The same 220 candidate groups as rank_settings.py. For K = 10 fresh bootstrap seeds (2000 lane-resampling
     replicates each, lanes resampled within each (terrain, difficulty) cell, paired across policies, the tidy.py
     definition) the robust flags of Ours - w/o RG, Ours - w/o LP and of the same-policy null Ours - Ours rerun.
     Reported: count ranges over the seeds and the rows whose flag is not the same in all seeds ("MC-borderline").
  2. Lane sign-flip permutation p: exact (2^L) for <= 16 lanes (as tidy.py); for pooled rows 1,000,000 random
     flips (p = (1 + #extreme) / (1 + N), floor 1e-6) instead of 10000. Holm over all 220 candidates per
     (metric, comparison), as rank_settings.py. Compared with the favourable_settings.csv Holm-220 values.
  3. The seed-0 point estimates are checked against favourable_settings.csv (must agree to rounding).

Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/mc_stability.py
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
RUN = HERE.parent
BASE = ("RGGP", "LPGP")
METRICS = {"sw": "switch_success", "comp": "completed"}
NF_NAME = {"sw": "switch_success_pct", "comp": "completion_pct"}
SPEEDS = ("[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0]")
K_SEEDS = 10
REPS = 2000
SEED0 = 20261010
N_PERM = 1_000_000
PERM_CHUNK = 50_000
EXACT_MAX_LANES = 16


def holm(p: np.ndarray) -> np.ndarray:  # identical to tidy.holm / rank_settings.holm
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    m = len(order)
    out[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return out


def as_bool(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.strip().str.lower().eq("true").to_numpy()


def load() -> tuple[pd.DataFrame, dict]:
    cols = ["terrain", "difficulty", "lane", "env", "policy", "speed_bin", "completed", "switch_success"]
    pt = pd.read_csv(HERE / "per_traj.csv.gz", usecols=cols)
    rep = pd.read_csv(HERE / "per_traj_replicate.csv.gz", usecols=cols)
    rep["policy"] = "Ours_rep"
    pt = pd.concat([pt, rep], ignore_index=True)
    pt["dkey"] = pt.difficulty.map(lambda v: f"{float(v):g}")
    vals, ref = {}, None
    for p in ("Ours", "RGGP", "LPGP", "Ours_rep"):
        s = pt[pt.policy == p].sort_values(["terrain", "env"]).reset_index(drop=True)
        if ref is None:
            ref = s
        else:
            for k in ("terrain", "env", "lane", "dkey", "speed_bin"):
                assert np.array_equal(s[k].to_numpy(), ref[k].to_numpy()), (p, k)
        vals[p] = {k: as_bool(s[c]).astype(float) for k, c in METRICS.items()}
    meta = ref[["terrain", "dkey", "lane", "speed_bin"]].copy()
    meta["cell"] = meta.terrain + "|" + meta.dkey
    meta["cluster"] = meta.cell + "|" + meta.lane.astype(str)
    return meta, vals


def groups(meta: pd.DataFrame) -> list[tuple]:
    terr = sorted(meta.terrain.unique())
    diffs = sorted(meta.dkey.unique(), key=float)
    out = [("all", "ALL", "ALL", "ALL")]
    out += [("difficulty", "ALL", d, "ALL") for d in diffs]
    out += [("terrain", t, "ALL", "ALL") for t in terr]
    out += [("speed_bin", "ALL", "ALL", s) for s in SPEEDS]
    out += [("difficulty_speed", "ALL", d, s) for d in diffs for s in SPEEDS]
    out += [("terrain_speed", t, "ALL", s) for t in terr for s in SPEEDS]
    out += [("cell", t, d, "ALL") for t in terr for d in diffs]
    out += [("cell_speed", t, d, s) for t in terr for d in diffs for s in SPEEDS]
    assert len(out) == 220
    return out


def mask(meta: pd.DataFrame, t: str, d: str, s: str) -> np.ndarray:
    m = np.ones(len(meta), bool)
    if t != "ALL":
        m &= meta.terrain.to_numpy() == t
    if d != "ALL":
        m &= meta.dkey.to_numpy() == d
    if s != "ALL":
        m &= meta.speed_bin.to_numpy() == s
    return m


def lane_sums(meta, vals, idx):
    cl = meta.cluster.to_numpy()[idx]
    uniq, inv = np.unique(cl, return_inverse=True)
    strata = np.array([u.rsplit("|", 1)[0] for u in uniq])
    den = np.bincount(inv, minlength=len(uniq)).astype(float)
    num = {p: {k: np.bincount(inv, weights=v[k][idx], minlength=len(uniq)) for k in METRICS} for p, v in vals.items()}
    return uniq, strata, den, num


def boot_weights(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.zeros((REPS, len(strata)))
    for s in np.unique(strata):
        j = np.flatnonzero(strata == s)
        w[:, j] = rng.multinomial(len(j), np.full(len(j), 1 / len(j)), size=REPS)
    return w


def signflip_p(nA, nB, den, rng, key) -> tuple[float, str]:
    """Two-sided lane label-swap p of 100 * (sum nA / sum den - sum nB / sum den); den is shared (same robots)."""
    L = len(den)
    tA, tB, tD = nA.sum(), nB.sum(), den.sum()
    obs = 100.0 * (tA - tB) / tD
    tol = 1e-9 * max(1.0, abs(obs))
    if L <= EXACT_MAX_LANES:
        S = ((np.arange(2 ** L)[:, None] >> np.arange(L)) & 1).astype(float)
        sA, sB = S @ nA, S @ nB
        T = 100.0 * ((sA + tB - sB) - (sB + tA - sA)) / tD
        return float((np.abs(T) >= abs(obs) - tol).sum() / len(T)), f"exact({2 ** L})"
    extreme = 0
    for start in range(0, N_PERM, PERM_CHUNK):
        n = min(PERM_CHUNK, N_PERM - start)
        bits = np.unpackbits(rng.integers(0, 256, size=(n, (L + 7) // 8), dtype=np.uint8), axis=1)[:, :L]
        S = bits.astype(np.float32)
        sA, sB = S @ nA.astype(np.float32), S @ nB.astype(np.float32)
        T = 100.0 * ((sA + tB - sB) - (sB + tA - sA)) / tD
        extreme += int((np.abs(T) >= abs(obs) - tol).sum())
    return (1 + extreme) / (1 + N_PERM), f"mc({N_PERM})"


def main() -> None:
    meta, vals = load()
    nf = json.loads((RUN / "noise_floor.json").read_text())
    fav = pd.read_csv(HERE / "favourable_settings.csv", dtype={"difficulty": str})
    fav["difficulty"] = fav.difficulty.map(lambda v: v if v == "ALL" else f"{float(v):g}")
    fav = fav.set_index(["scope", "terrain", "difficulty", "speed_bin"])
    p90 = {}
    for s, name in NF_NAME.items():
        p90[s] = float(fav[f"{s}_noise_p90"].dropna().iloc[0])
    rows = []
    max_dev = 0.0
    for gi, (scope, t, d, s) in enumerate(groups(meta)):
        idx = np.flatnonzero(mask(meta, t, d, s))
        uniq, strata, den, num = lane_sums(meta, vals, idx)
        ref = fav.loc[(scope, t, d, s)]
        rec = {"scope": scope, "terrain": t, "difficulty": d, "speed_bin": s, "n_robots": len(idx), "n_lanes": len(uniq)}
        est = {p: {k: 100.0 * num[p][k].sum() / den.sum() for k in METRICS} for p in num}
        for k in METRICS:
            max_dev = max(max_dev, abs(est["Ours"][k] - ref[f"{k}_Ours"]))
            for b in BASE:
                max_dev = max(max_dev, abs(est[b][k] - ref[f"{k}_{b}"]))
            vs = [est[p][k] for p in ("Ours", "RGGP", "LPGP")]
            rec[f"{k}_informative"] = not (all(v < 5 for v in vs) or all(v > 95 for v in vs))
        # 1. bootstrap robust flags over K seeds
        comps = [(b, b) for b in BASE] + [("Ours_rep", "null")]
        flags = {(k, lab): [] for k in METRICS for _, lab in comps}
        bounds = {(k, lab): [] for k in METRICS for _, lab in comps}
        for j in range(K_SEEDS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED0 + j, zlib.crc32(f"{scope}|{t}|{d}|{s}".encode())]))
            w = boot_weights(strata, rng)
            wd = w @ den
            for k in METRICS:
                bo = 100.0 * (w @ num["Ours"][k]) / wd
                for pol, lab in comps:
                    diff = bo - 100.0 * (w @ num[pol][k]) / wd
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    e = est["Ours"][k] - est[pol][k]
                    flags[(k, lab)].append(bool((lo > 0 or hi < 0) and abs(e) > p90[k]))
                    bounds[(k, lab)].append((lo, hi))
        for (k, lab), f in flags.items():
            e = est["Ours"][k] - est["Ours_rep" if lab == "null" else lab][k]
            rec[f"{k}_diff_{lab}"] = e
            rec[f"{k}_robust_frac_{lab}"] = float(np.mean(f))
            lo = np.array([b[0] for b in bounds[(k, lab)]])
            hi = np.array([b[1] for b in bounds[(k, lab)]])
            rec[f"{k}_lo_min_{lab}"], rec[f"{k}_lo_max_{lab}"] = lo.min(), lo.max()
            rec[f"{k}_hi_min_{lab}"], rec[f"{k}_hi_max_{lab}"] = hi.min(), hi.max()
            for j in range(K_SEEDS):
                rec[f"{k}_robust_{lab}_s{j}"] = f[j]
            if lab != "null":
                rec[f"{k}_robust_csv_{lab}"] = str(ref[f"{k}_robust_{lab}"]) == "True"
        # 2. high-resolution permutation p
        for k in METRICS:
            for b in BASE:
                rng = np.random.default_rng(np.random.SeedSequence([SEED0 + 99, zlib.crc32(f"{scope}|{t}|{d}|{s}|{k}|{b}".encode())]))
                p, meth = signflip_p(num["Ours"][k], num[b][k], den, rng, None)
                rec[f"{k}_perm_p_hi_{b}"] = p
                rec[f"{k}_perm_method_hi"] = meth
                rec[f"{k}_perm_p_csv_{b}"] = float(ref[f"{k}_perm_p_{b}"])
                rec[f"{k}_holm220_csv_{b}"] = float(ref[f"{k}_holm220_{b}"])
        rows.append(rec)
        if gi % 20 == 0:
            print(f"{gi + 1}/220 {scope} {t} {d} {s}", flush=True)
    df = pd.DataFrame(rows)
    for k in METRICS:
        for b in BASE:
            df[f"{k}_holm220_hi_{b}"] = holm(df[f"{k}_perm_p_hi_{b}"].to_numpy(float))

    # ---------------- summaries ----------------
    out = {"k_seeds": K_SEEDS, "reps_per_seed": REPS, "seed0": SEED0, "n_perm_pooled": N_PERM,
           "noise_p90": p90, "max_abs_dev_point_estimates_vs_favourable_settings_csv": max_dev}
    exact = df.sw_perm_method_hi.str.startswith("exact")
    out["exact_perm_rows_max_abs_dev_vs_csv"] = float(max(
        (df.loc[exact, f"{k}_perm_p_hi_{b}"] - df.loc[exact, f"{k}_perm_p_csv_{b}"]).abs().max()
        for k in METRICS for b in BASE))
    counts, borderline = {}, []
    for k in METRICS:
        inf = df[f"{k}_informative"]
        for lab in BASE + ("null",):
            d = df[f"{k}_diff_{lab}"]
            per_seed_w, per_seed_l = [], []
            for j in range(K_SEEDS):
                r = df[f"{k}_robust_{lab}_s{j}"].astype(bool)
                sel = r if lab == "null" else (r & inf)
                per_seed_w.append(int((sel & (d > 0)).sum()))
                per_seed_l.append(int((sel & (d < 0)).sum()))
            frac = df[f"{k}_robust_frac_{lab}"]
            sel_inf = inf if lab != "null" else pd.Series(True, index=df.index)
            entry = {"wins_per_seed": per_seed_w, "losses_per_seed": per_seed_l,
                     "wins_range": [min(per_seed_w), max(per_seed_w)], "losses_range": [min(per_seed_l), max(per_seed_l)],
                     "wins_all_seeds": int((sel_inf & (frac == 1) & (d > 0)).sum()),
                     "losses_all_seeds": int((sel_inf & (frac == 1) & (d < 0)).sum()),
                     "rows_mixed": int((sel_inf & (frac > 0) & (frac < 1)).sum())}
            if lab != "null":
                csvr = df[f"{k}_robust_csv_{lab}"] & inf
                entry["csv_wins"] = int((csvr & (d > 0)).sum())
                entry["csv_losses"] = int((csvr & (d < 0)).sum())
                h_csv = df[f"{k}_holm220_csv_{lab}"] < 0.05
                h_hi = df[f"{k}_holm220_hi_{lab}"] < 0.05
                stable = sel_inf & (frac == 1)
                entry["holm_lt05_csv_robustcsv"] = [int((csvr & h_csv & (d > 0)).sum()), int((csvr & h_csv & (d < 0)).sum())]
                entry["holm_lt05_hires_robustcsv"] = [int((csvr & h_hi & (d > 0)).sum()), int((csvr & h_hi & (d < 0)).sum())]
                entry["holm_lt05_hires_robust_all_seeds"] = [int((stable & h_hi & (d > 0)).sum()),
                                                             int((stable & h_hi & (d < 0)).sum())]
                entry["holm_verdict_changes_csv_to_hires"] = [
                    {"row": f"{r.scope}|{r.terrain}|{r.difficulty}|{r.speed_bin}", "diff": round(float(getattr(r, f'{k}_diff_{lab}')), 3),
                     "holm_csv": round(float(getattr(r, f'{k}_holm220_csv_{lab}')), 4),
                     "holm_hires": round(float(getattr(r, f'{k}_holm220_hi_{lab}')), 4),
                     "robust_csv": bool(getattr(r, f'{k}_robust_csv_{lab}'))}
                    for r in df[(h_csv != h_hi)].itertuples()]
            counts[f"{NF_NAME[k]} vs {lab}"] = entry
            for r in df[sel_inf & (frac > 0) & (frac < 1)].itertuples():
                borderline.append({"metric": NF_NAME[k], "vs": lab, "row": f"{r.scope}|{r.terrain}|{r.difficulty}|{r.speed_bin}",
                                   "diff": round(float(getattr(r, f"{k}_diff_{lab}")), 3),
                                   "robust_frac": float(getattr(r, f"{k}_robust_frac_{lab}")),
                                   "lo_range": [round(float(getattr(r, f"{k}_lo_min_{lab}")), 3), round(float(getattr(r, f"{k}_lo_max_{lab}")), 3)],
                                   "hi_range": [round(float(getattr(r, f"{k}_hi_min_{lab}")), 3), round(float(getattr(r, f"{k}_hi_max_{lab}")), 3)]})
    out["counts"] = counts
    out["borderline_rows"] = borderline
    pooled = ~exact
    out["pooled_rows_min_p_csv_vs_hires"] = {
        f"{k}|{b}": [float(df.loc[pooled, f"{k}_perm_p_csv_{b}"].min()), float(df.loc[pooled, f"{k}_perm_p_hi_{b}"].min())]
        for k in METRICS for b in BASE}
    # Table C rows (losses with Holm-220 < 0.05 in the CSV) and Table A rows, re-evaluated.
    unf = pd.read_csv(HERE / "unfavourable_settings.csv", dtype={"difficulty": str})
    unf["difficulty"] = unf.difficulty.map(lambda v: v if v == "ALL" else f"{float(v):g}")
    dfi = df.set_index(["scope", "terrain", "difficulty", "speed_bin"])
    lab_of = {"w/o RG": "RGGP", "w/o LP": "LPGP"}
    kk = {"switch_success_pct": "sw", "completion_pct": "comp"}
    tc = []
    for r in unf.itertuples():
        g = dfi.loc[(r.scope, r.terrain, r.difficulty, r.speed_bin)]
        k, b = kk[r.metric], lab_of[r.vs]
        tc.append({"loss_rank": int(r.loss_rank), "row": f"{r.scope}|{r.terrain}|{r.difficulty}|{r.speed_bin}",
                   "metric": r.metric, "vs": r.vs, "diff": round(float(r.diff), 3), "holm_csv": float(r.holm220),
                   "holm_hires": float(g[f"{k}_holm220_hi_{b}"]), "robust_frac": float(g[f"{k}_robust_frac_{b}"])})
    out["loss_rows"] = tc
    (HERE / "mc_stability.json").write_text(json.dumps(out, indent=1))
    keep = [c for c in df.columns if not re.search(r"_s\d+$", c)]
    df[keep].to_csv(HERE / "mc_stability_rows.csv", index=False, float_format="%.6g")
    print(json.dumps({k: v for k, v in out.items() if k not in ("borderline_rows", "loss_rows")}, indent=1))
    print("borderline rows:", len(borderline))
    for b in borderline:
        print(" ", b)
    print("loss rows whose Holm<0.05 verdict changes:")
    for r in tc:
        if (r["holm_csv"] < 0.05) != (r["holm_hires"] < 0.05) or r["holm_csv"] < 0.05:
            print(" ", r)


if __name__ == "__main__":
    main()
