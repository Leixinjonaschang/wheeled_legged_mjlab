"""Ours vs LPGP scorecard over every metric of the evaluation (paired, Holm-corrected).

Binary outcomes: exact McNemar test on identical trajectories. Continuous outcomes:
paired Wilcoxon signed-rank (taken from benefit_metrics.csv / energy_contact data).
Holm correction over terrains within each metric; alpha = 0.05.

Run from the repo root (after benefit_metrics.py and energy_contact.py):
  .venv/bin/python scripts/eval/lipm_diagnostics/ours_vs_lpgp_scorecard.py
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

OUT = Path("logs/lipm_eval/full")
TERRAINS = (
    "flat", "discrete_obstacles", "random_rough", "hf_pyramid_slope",
    "hf_pyramid_slope_inv", "pyramid_stair", "pyramid_stair_inv", "random_stairs",
    "random_spread", "stepping_stones", "tilted_grid",
)
A, B = "Ours", "LPGP"
NON_FINITE, FELL, ILLEGAL = 0, 1, 2


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def load(terrain, run):
    run_dir = OUT / "raw" / terrain / run
    data = np.load(run_dir / "timeseries.npz")
    with (run_dir / "trajectories.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return data, rows


def classes(data):
    cmd = data["command_w"][0, :, :2]
    heading = data["heading_target"][0]
    vx = np.cos(heading) * cmd[:, 0] + np.sin(heading) * cmd[:, 1]
    vy = -np.sin(heading) * cmd[:, 0] + np.cos(heading) * cmd[:, 1]
    return np.where(
        data["is_standing"][0], "standing",
        np.where(np.abs(vx) >= np.abs(vy), np.where(vx > 0, "forward", "backward"), "lateral"),
    )


results = []  # (metric, terrain, ours, lpgp, p, higher_is_better)


def binary(metric, terrain, a, b, higher_is_better, mask=None):
    mask = np.ones_like(a, dtype=bool) if mask is None else mask
    a, b = a[mask], b[mask]
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    results.append((metric, terrain, 100 * a.mean(), 100 * b.mean(), p, higher_is_better))


for terrain in TERRAINS:
    (da, ra), (db, rb) = load(terrain, A), load(terrain, B)
    ok_a, ok_b = da["fail_step"] < 0, db["fail_step"] < 0
    binary("strict success (%)", terrain, ok_a, ok_b, True)
    cls = classes(da)
    for name in ("forward", "backward", "lateral", "standing"):
        binary(f"strict success, {name} cmd (%)", terrain, ok_a, ok_b, True, cls == name)
    fall_a = ~(da["failure_flags"][..., FELL] | da["failure_flags"][..., NON_FINITE]).any(0)
    fall_b = ~(db["failure_flags"][..., FELL] | db["failure_flags"][..., NON_FINITE]).any(0)
    binary("fall-only survival (%)", terrain, fall_a, fall_b, True)
    ill_a = np.array([r["fail_illegal_contact"] == "True" for r in ra])
    ill_b = np.array([r["fail_illegal_contact"] == "True" for r in rb])
    binary("non-wheel contact termination (%)", terrain, ill_a, ill_b, False)
    track_a = ok_a & np.array([int(r["tracking_failure_shadow_step"]) < 0 for r in ra])
    track_b = ok_b & np.array([int(r["tracking_failure_shadow_step"]) < 0 for r in rb])
    binary("success incl. tracking-failure check (%)", terrain, track_a, track_b, True)
    surv_a = np.array([float(r["survival_time_s"]) for r in ra])
    surv_b = np.array([float(r["survival_time_s"]) for r in rb])
    diff = surv_a - surv_b
    p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    results.append(("survival time (s)", terrain, surv_a.mean(), surv_b.mean(), p, True))

# Continuous behaviour metrics (paired over the common alive window / both-success subset).
with (OUT / "results" / "benefit_metrics.csv").open() as stream:
    for r in csv.DictReader(stream):
        if r["other"] != B or r["view"] not in ("common_window", "both_success"):
            continue
        results.append((f"{r['metric']} [{r['view']}]", r["terrain"], float(r["ours_mean"]),
                        float(r["other_mean"]), float(r["p"]), False))

# Energy and lift selectivity (5 terrains re-simulated in energy_contact.py).
DIAG = OUT / "diagnostics" / "energy_contact"
for terrain in ("flat", "random_rough", "pyramid_stair", "stepping_stones", "discrete_obstacles"):
    da, db = np.load(DIAG / f"{terrain}__{A}.npz"), np.load(DIAG / f"{terrain}__{B}.npz")
    window = da["valid"] & db["valid"]
    dt = float(da["step_dt"])

    def per_traj(d):
        power = d["power_leg"] + d["power_wheel"]
        with np.errstate(all="ignore"):
            dist = np.where(window, d["speed"], 0.0).sum(0) * dt
            energy = np.where(window, power, 0.0).sum(0) * dt
            speed = np.where(window, d["speed"], 0.0).sum(0) / window.sum(0)
            cot = np.where((speed >= 0.2) & (dist > 0), energy / dist / (d["mass_kg"] * 9.81), np.nan)
            air = ~d["wheels_in_contact"].all(-1)
            flat = window & (d["relief"] < 0.02)
            rough = window & (d["relief"] >= 0.05)
            sel = np.where((flat.sum(0) >= 25) & (rough.sum(0) >= 25),
                           (air & rough).sum(0) / rough.sum(0) - (air & flat).sum(0) / flat.sum(0),
                           np.nan)
        return cot, sel

    (cot_a, sel_a), (cot_b, sel_b) = per_traj(da), per_traj(db)
    for metric, a, b, higher in (("CoT", cot_a, cot_b, False),
                                 ("lift selectivity: rough - flat airborne (pp)", sel_a, sel_b, True)):
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < 20:
            continue
        diff = a[good] - b[good]
        scale = 100.0 if "pp" in metric else 1.0
        results.append((metric, terrain, scale * a[good].mean(), scale * b[good].mean(),
                        wilcoxon(diff).pvalue, higher))

# Holm per metric, then verdicts.
metrics = sorted({r[0] for r in results}, key=[r[0] for r in results].index)
rows = []
for metric in metrics:
    family = [r for r in results if r[0] == metric]
    for (m, terrain, a, b, p, higher), ph in zip(family, holm([r[4] for r in family])):
        better = a > b if higher else a < b
        verdict = "=" if ph >= 0.05 else ("win" if better else "loss")
        rows.append({"metric": m, "terrain": terrain, "ours": a, "lpgp": b,
                     "p": p, "p_holm": ph, "verdict": verdict})

with (OUT / "results" / "ours_vs_lpgp_scorecard.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = ["# Ours vs LPGP scorecard (paired, Holm over terrains, alpha 0.05)", "",
         "| Metric | wins | losses | ties | terrains won (Ours vs LPGP) |", "|---|---:|---:|---:|---|"]
for metric in metrics:
    fam = [r for r in rows if r["metric"] == metric]
    wins = [r for r in fam if r["verdict"] == "win"]
    losses = [r for r in fam if r["verdict"] == "loss"]
    won = "; ".join(f"{r['terrain']} {r['ours']:.3g} vs {r['lpgp']:.3g}" for r in wins)
    lines.append(f"| {metric} | {len(wins)} | {len(losses)} | {len(fam) - len(wins) - len(losses)} | {won} |")
text = "\n".join(lines) + "\n"
(OUT / "results" / "ours_vs_lpgp_scorecard.md").write_text(text)
print(text)
