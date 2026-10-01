"""Sensitivity of the flat-rough-flat course metrics to the lift-event clearance threshold.

Recomputes switch success and the per-window verdicts from the raw course rollouts
(forward commands, 3 rough types x 1024) for several STEP_MIN_CLEARANCE values, and
summarizes the airborne phases (>= 0.06 s) inside the rough window.

  uv run --no-project --python 3.13 --with numpy --with scipy \
      python scripts/eval/lipm_diagnostics/clearance_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/eval")
import mode_switch_eval as m  # noqa: E402

COURSE = Path("logs/lipm_eval/mode_switch")
OUT = Path("logs/lipm_eval/full/results/clearance_sensitivity.md")
CKPTS = {"Ours": "Ours", "RGGP": "Ours w/o RG", "LPGP": "Ours w/o LP"}
THRESHOLDS = [-np.inf, 0.01, 0.02, 0.03, 0.05]
KEYS = ["switch_success", "roll_flat_in", "step_rough", "roll_flat_out", "completed"]

data = {}
for path in sorted(COURSE.glob("*__forward__*.npz")):
    rough, _, ckpt = path.stem.split("__")
    data[(rough, ckpt)] = dict(np.load(path))

lines = [
    "# Lift-event clearance threshold sensitivity (flat-rough-flat course, forward, 3 x 1024 pooled)",
    "",
    "Lift event = one wheel airborne for >= 0.06 s whose peak clearance above the highest terrain point in the",
    "0.4 m x 0.4 m wheel scan reaches the threshold. `-inf` = duration only (recorded clearance is clipped at 0).",
    "Window verdicts are percentages of the robots that traversed that window; switch success and completion use all 3072.",
    "",
    "| threshold | method | switch success | roll flat-in | step rough (both wheels) | roll flat-out | completion |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for thr in THRESHOLDS:
    m.STEP_MIN_CLEARANCE = thr
    for ckpt, name in CKPTS.items():
        agg = {k: [] for k in KEYS}
        for (_, c), d in data.items():
            if c != ckpt:
                continue
            for i in range(d["valid"].shape[1]):
                r = m.trajectory_metrics(d, i)
                for k in KEYS:
                    if r[k] is not None:
                        agg[k].append(r[k])
        cells = " | ".join(f"{100 * np.mean(agg[k]):.1f} (n={len(agg[k])})" for k in KEYS)
        label = "duration only" if thr == -np.inf else f"{100 * thr:.0f} cm"
        lines.append(f"| {label} | {name} | {cells} |")

lines += [
    "",
    "## Airborne phases >= 0.06 s inside the rough window (x in [4.2, 7.4) m)",
    "",
    "| method | phases | median duration | median peak clearance | share with peak >= 3 cm |",
    "|---|---:|---:|---:|---:|",
]
for ckpt, name in CKPTS.items():
    durations, peaks = [], []
    for (_, c), d in data.items():
        if c != ckpt:
            continue
        for i in range(d["valid"].shape[1]):
            x = d["course_x"][:, i]
            window = d["valid"][:, i] & (x >= m.WINDOWS["rough"][0]) & (x < m.WINDOWS["rough"][1])
            airborne = ~d["wheel_contact"][:, i]
            clearance = d["wheel_clearance"][:, i].astype(float)
            for w in range(2):
                edges = np.diff(np.concatenate([[False], airborne[:, w] & window, [False]]).astype(np.int8))
                for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if e - s >= m.STEP_MIN_DURATION:
                        durations.append(e - s)
                        peaks.append(clearance[s:e, w].max())
    durations, peaks = np.array(durations), np.array(peaks)
    dt = float(next(iter(data.values()))["step_dt"])
    lines.append(
        f"| {name} | {len(durations)} | {np.median(durations):.0f} steps ({np.median(durations) * dt:.2f} s) | "
        f"{100 * np.median(peaks):.1f} cm | {100 * np.mean(peaks >= 0.03):.1f}% |"
    )

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
