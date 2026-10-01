"""Run-to-run noise floor of the mode_switch_v2 rollouts.

Two rollouts of the same (rough, command, ckpt) with identical terrain, DR, reset states and
commands (checked here) differ only by nondeterministic GPU physics. Per (rough, command,
ckpt, difficulty) cell this reports the difference of the summary statistics between the
two runs and the share of robots whose per-trajectory verdict flips; if DIR_A holds a
stats.json, also the bootstrap interval half-width of the same cells for comparison.
Checkpoint differences inside this floor should be read as ties.
Run (repo root), e.g. after run_all.sh (which reruns Ours into replicate/):
  uv run --no-project --python 3.13 --with numpy --with scipy python \
    scripts/eval/mode_switch/tools/noise_floor.py logs/lipm_eval/mode_switch_v2 \
    logs/lipm_eval/mode_switch_v2/replicate
Writes DIR_A/noise_floor.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import mode_switch_eval as M  # noqa: E402

METRICS = ("completion_pct", "failure_pct", "switch_success_pct", "roll_flat_in_pct", "step_rough_pct", "roll_flat_out_pct",
           "peak_tilt_deg", "cot")
VERDICTS = ("completed", "failed", "switch_success", "roll_flat_in", "step_rough", "roll_flat_out")


def rows_of(d) -> list[dict]:
    return [{"difficulty": float(d["difficulty"][i]), **M.trajectory_metrics(d, i)} for i in range(d["valid"].shape[1])]


def main(dir_a: str, dir_b: str) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)  # Empty / all-NaN cells (e.g. nothing completed).
    a_dir, b_dir = Path(dir_a), Path(dir_b)
    stats = json.loads((a_dir / "stats.json").read_text())["cells"] if (a_dir / "stats.json").exists() else {}
    cells, flips, bad = {}, {}, []
    for path_b in sorted(b_dir.glob("*__*__*.npz")):
        path_a = a_dir / path_b.name
        if not path_a.exists():
            continue
        a, b = dict(np.load(path_a)), dict(np.load(path_b))
        same = {k: bool(np.array_equal(a[k], b[k])) for k in (*M.IDENTITY_KEYS, "time_limit_s") if k in a}
        if not all(same.values()):
            bad.append(path_b.name)
            print(f"{path_b.name}: configurations differ {[k for k, ok in same.items() if not ok]}, skipped")
            continue
        ra, rb = rows_of(a), rows_of(b)
        moved = np.flatnonzero(np.any(a["course_x"] != b["course_x"], axis=1))
        first = int(moved[0]) if len(moved) else None
        flips[path_b.stem] = {
            "first_step_with_differing_course_x": first,
            **{k: float(np.mean([x[k] != y[k] for x, y in zip(ra, rb) if x[k] is not None and y[k] is not None]))
               for k in VERDICTS},
        }
        for level in sorted({r["difficulty"] for r in ra}):
            sa, sb = ({**M.summarize(sel), "failure_pct": 100 * float(np.mean([r["failed"] for r in sel]))}
                      for sel in ([r for r in rows if r["difficulty"] == level] for rows in (ra, rb)))
            key = f"{path_b.stem.replace('__', '|')}|d={level:g}"
            ci = stats.get(key, {})
            cells[key] = {m: {"run_a": sa[m], "run_b": sb[m], "diff": sb[m] - sa[m],
                              "ci_half_width_a": (ci[m][2] - ci[m][1]) / 2 if m in ci else None} for m in METRICS}
    summary = {}
    for m in METRICS:
        diffs = np.abs([c[m]["diff"] for c in cells.values()])
        halves = [c[m]["ci_half_width_a"] for c in cells.values() if c[m]["ci_half_width_a"] is not None]
        summary[m] = {"median_abs_diff": float(np.nanmedian(diffs)), "p90_abs_diff": float(np.nanpercentile(diffs, 90)),
                      "max_abs_diff": float(np.nanmax(diffs)),
                      "median_ci_half_width": float(np.nanmedian(halves)) if halves else None}
        print(f"{m:20s} |run A - run B| median {summary[m]['median_abs_diff']:.2f}, p90 {summary[m]['p90_abs_diff']:.2f}, "
              f"max {summary[m]['max_abs_diff']:.2f}; median 95 % CI half-width {summary[m]['median_ci_half_width']}")
    for name, f in flips.items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()})
    (a_dir / "noise_floor.json").write_text(json.dumps(
        {"run_a": str(a_dir), "run_b": str(b_dir), "summary_over_cells": summary, "verdict_flip_share": flips,
         "cells": cells, "skipped_config_mismatch": bad}, indent=1))
    return int(bool(bad) or not cells)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
