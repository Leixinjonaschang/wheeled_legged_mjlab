"""Fall-only survival: Ours vs RGGP / LPGP, paired over identical trajectories.

Failure = fell_over (tilt > 85 deg) or non_finite_physics at any of the 500 steps;
illegal_contact is ignored. Terminated robots were never reset during evaluation,
so their motion after an illegal contact is the exact counterfactual without that
criterion. All runs of a terrain share terrain, initial states and command
sequences, so trajectory i is paired across checkpoints (exact McNemar test).

Run from the repo root:
  .venv/bin/python scripts/eval/lipm_diagnostics/fall_only_comparison.py
"""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

OUT = Path("logs/lipm_eval/full")
TERRAINS = (
    "flat", "discrete_obstacles", "random_rough", "hf_pyramid_slope",
    "hf_pyramid_slope_inv", "pyramid_stair", "pyramid_stair_inv", "random_stairs",
    "random_spread", "stepping_stones", "tilted_grid",
)
RUNS = ("Ours", "RGGP", "LPGP")
# Column order of failure_flags: non_finite, fell_over, illegal_contact.
NON_FINITE, FELL, ILLEGAL = 0, 1, 2


def first_step(mask: np.ndarray) -> np.ndarray:
    """First step index (0-based) at which mask is set, or a large sentinel."""
    return np.where(mask.any(0), mask.argmax(0), 10**9)


def load(terrain: str, run: str) -> tuple[np.ndarray, np.ndarray]:
    """Fall-only survival and the fall step relative to the first illegal contact."""
    run_dir = OUT / "raw" / terrain / run
    data = np.load(run_dir / "timeseries.npz")
    with (run_dir / "trajectories.csv").open() as stream:
        evaluated = np.array([r["evaluated"] == "True" for r in csv.DictReader(stream)])
    assert evaluated.all()
    flags = data["failure_flags"]
    fall = first_step(flags[..., FELL] | flags[..., NON_FINITE])
    contact = first_step(flags[..., ILLEGAL])
    alive = fall >= 10**9
    return alive, ~alive & (contact < fall)


def survived(terrain: str, run: str) -> np.ndarray:
    return load(terrain, run)[0]


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.96
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (center - half), 100 * (center + half)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


rows = []
totals = {run: [0, 0] for run in RUNS}
for terrain in TERRAINS:
    loaded = {run: load(terrain, run) for run in RUNS}
    alive = {run: loaded[run][0] for run in RUNS}
    row = {"terrain": terrain, "n": len(alive["Ours"])}
    for run in RUNS:
        k, n = int(alive[run].sum()), len(alive[run])
        low, high = wilson(k, n)
        row[f"{run}_pct"] = 100 * k / n
        row[f"{run}_ci95"] = f"[{low:.1f}, {high:.1f}]"
        after_contact = int(loaded[run][1].sum())
        row[f"{run}_falls_direct"] = n - k - after_contact
        row[f"{run}_falls_after_contact"] = after_contact
        if terrain != "flat":
            totals[run][0] += k
            totals[run][1] += n
    for other in ("RGGP", "LPGP"):
        ours_only = int((alive["Ours"] & ~alive[other]).sum())
        other_only = int((~alive["Ours"] & alive[other]).sum())
        discordant = ours_only + other_only
        row[f"vs_{other}_ours_only"] = ours_only
        row[f"vs_{other}_other_only"] = other_only
        row[f"vs_{other}_p"] = binomtest(ours_only, discordant, 0.5).pvalue if discordant else 1.0
    rows.append(row)

for other in ("RGGP", "LPGP"):
    for row, p in zip(rows, holm([r[f"vs_{other}_p"] for r in rows])):
        row[f"vs_{other}_p_holm"] = p

with (OUT / "results" / "fall_only_comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Fall-only survival (illegal_contact ignored), paired over identical trajectories",
    "",
    "| Terrain | Ours % | RGGP % | LPGP % | Ours vs RGGP: Ours-only / RGGP-only, p (Holm) | "
    "Ours vs LPGP: Ours-only / LPGP-only, p (Holm) |",
    "|---|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['terrain']} | {r['Ours_pct']:.1f} | {r['RGGP_pct']:.1f} | {r['LPGP_pct']:.1f} | "
        f"{r['vs_RGGP_ours_only']} / {r['vs_RGGP_other_only']}, {r['vs_RGGP_p_holm']:.2g} | "
        f"{r['vs_LPGP_ours_only']} / {r['vs_LPGP_other_only']}, {r['vs_LPGP_p_holm']:.2g} |"
    )
lines += [
    "",
    "Falls split by whether a non-wheel (illegal) contact happened earlier in the trajectory "
    "(training would have terminated there, so post-contact states are unseen in training):",
    "",
    "| Terrain | Ours: no prior contact / after contact | RGGP: no prior contact / after contact | "
    "LPGP: no prior contact / after contact |",
    "|---|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['terrain']} | "
        + " | ".join(f"{r[f'{run}_falls_direct']} / {r[f'{run}_falls_after_contact']}" for run in RUNS)
        + " |"
    )
lines += [
    "",
    "Pooled over the 10 non-flat terrains: "
    + ", ".join(f"{run} {100 * k / n:.1f}% ({k}/{n})" for run, (k, n) in totals.items()),
    "",
    "Single training seed per method: p-values reflect evaluation sampling only, not "
    "training-seed variability.",
    "",
]
(OUT / "results" / "fall_only_comparison.md").write_text("\n".join(lines))
print("\n".join(lines))
