"""Fall-only completion on the v2 course: illegal (non-wheel) contact is ignored.

Rollouts run with auto_reset=False, so a robot keeps being simulated (policy still acting)
after its first termination; positions and orientation are recorded for every step.
Fall-only completion = the base reaches the course end within the time limit while the
tilt never exceeds 85 deg (the fell_over limit) and the state stays finite.
Paired differences use a cluster bootstrap over (terrain, difficulty, lane) instances.

  .venv/bin/python scripts/eval/mode_switch/analysis/fall_only_completion.py
"""

import glob
from pathlib import Path

import numpy as np

V2 = Path("logs/lipm_eval/mode_switch_v2")
LIMIT = np.radians(85.0)
CKPTS = ("Ours", "RGGP", "LPGP")


def per_traj(path):
    d = np.load(path)
    valid, x = d["valid"], d["course_x"]
    g = d["gravity_b"].astype(np.float64)
    end = float(d["course_end"])
    tilt = np.arccos(np.clip(-g[..., 2], -1, 1))
    bad = (tilt > LIMIT) | ~np.isfinite(g).all(-1) | ~np.isfinite(x)
    at_end = x >= end
    T = len(x)
    first_end = np.where(at_end.any(0), at_end.argmax(0), T)
    first_bad = np.where(bad.any(0), bad.argmax(0), T)
    fall_only = first_end < first_bad
    first_end_valid = np.where((at_end & valid).any(0), (at_end & valid).argmax(0), T)
    standard = first_end_valid < T
    lane = d["lane"] if "lane" in d.files else np.arange(x.shape[1]) % 16
    difficulty = d["difficulty"] if "difficulty" in d.files else np.ones(x.shape[1])
    return standard, fall_only, lane, difficulty


rows = {}
for ckpt in CKPTS:
    for path in sorted(glob.glob(str(V2 / f"*__forward_wide__{ckpt}.npz"))):
        terrain = Path(path).name.split("__")[0]
        rows[(terrain, ckpt)] = per_traj(path)

terrains = sorted({t for t, _ in rows})
lines = ["| terrain | completion O / RG / LP | fall-only completion O / RG / LP | gain O / RG / LP |", "|---|---|---|---|"]
pooled = {c: ([], []) for c in CKPTS}
clusters = []
for terrain in terrains:
    std = {c: rows[(terrain, c)][0] for c in CKPTS}
    fo = {c: rows[(terrain, c)][1] for c in CKPTS}
    lane, diff = rows[(terrain, "Ours")][2], rows[(terrain, "Ours")][3]
    for c in CKPTS:
        pooled[c][0].append(std[c]); pooled[c][1].append(fo[c])
    for dv in np.unique(diff):
        for ln in np.unique(lane):
            m = (diff == dv) & (lane == ln)
            clusters.append({c: (std[c][m].sum(), fo[c][m].sum(), m.sum()) for c in CKPTS})
    s = " / ".join(f"{100 * std[c].mean():.1f}" for c in CKPTS)
    f = " / ".join(f"{100 * fo[c].mean():.1f}" for c in CKPTS)
    g = " / ".join(f"{100 * (fo[c].mean() - std[c].mean()):+.1f}" for c in CKPTS)
    lines.append(f"| {terrain} | {s} | {f} | {g} |")
s = " / ".join(f"{100 * np.concatenate(pooled[c][0]).mean():.1f}" for c in CKPTS)
f = " / ".join(f"{100 * np.concatenate(pooled[c][1]).mean():.1f}" for c in CKPTS)
g = " / ".join(f"{100 * (np.concatenate(pooled[c][1]).mean() - np.concatenate(pooled[c][0]).mean()):+.1f}" for c in CKPTS)
lines.append(f"| **all 40 cells** | {s} | {f} | {g} |")

# Paired Ours - ablation differences, cluster bootstrap over terrain instances (640 clusters).
rng = np.random.default_rng(20260930)
arr = {c: np.array([[cl[c][0], cl[c][1], cl[c][2]] for cl in clusters], float) for c in CKPTS}
n_cl = len(clusters)
lines += ["", "| paired difference (pp), 95% cluster-bootstrap CI | standard completion | fall-only completion |", "|---|---|---|"]
for abl in ("RGGP", "LPGP"):
    out = []
    for col in (0, 1):
        est = 100 * (arr["Ours"][:, col].sum() - arr[abl][:, col].sum()) / arr["Ours"][:, 2].sum()
        idx = rng.integers(0, n_cl, size=(2000, n_cl))
        boot = 100 * (arr["Ours"][idx, col].sum(1) - arr[abl][idx, col].sum(1)) / arr["Ours"][idx, 2].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(f"{est:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    lines.append(f"| Ours - {abl} | {out[0]} | {out[1]} |")

text = "\n".join(lines)
(V2 / "analysis" / "fall_only_completion.md").write_text(text + "\n")
print(text)
