"""Legacy regression: trajectory_metrics() on the old tilted_grid/forward rollouts must
reproduce logs/lipm_eval/mode_switch/trajectories.csv exactly.

CSV conventions: None -> '', bools -> 'True'/'False', floats compared to 1e-6 relative.
Run (repo root):
  uv run --no-project --python 3.13 --with numpy --with scipy python \
    scripts/eval/mode_switch/tools/legacy_regression.py
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import mode_switch_eval as M  # noqa: E402

LEGACY = ROOT / "logs/lipm_eval/mode_switch"


def same(value, text: str) -> bool:
    if value is None:
        return text == ""
    if isinstance(value, (bool, np.bool_)):
        return text == str(bool(value))
    if isinstance(value, (int, np.integer)):
        return text == str(int(value))
    ref = float(text)
    if math.isnan(value) or math.isnan(ref):
        return math.isnan(value) and math.isnan(ref)
    return math.isclose(value, ref, rel_tol=1e-6, abs_tol=0.0)


def main() -> int:
    with (LEGACY / "trajectories.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    total = 0
    for ckpt in ("Ours", "RGGP", "LPGP"):
        ref = {int(r["env"]): r for r in rows
               if (r["rough"], r["command"], r["ckpt"]) == ("tilted_grid", "forward", ckpt)}
        d = dict(np.load(LEGACY / f"tilted_grid__forward__{ckpt}.npz"))
        n = d["valid"].shape[1]
        mismatches = []
        for i in range(n):
            metrics = M.trajectory_metrics(d, i)
            for key, value in metrics.items():
                if not same(value, ref[i][key]):
                    mismatches.append((i, key, value, ref[i][key]))
        missing = len(ref) != n
        print(f"{ckpt}: envs={n} csv_rows={len(ref)} mismatches={len(mismatches)}"
              + (" ROW COUNT DIFFERS" if missing else ""))
        for item in mismatches[:10]:
            print("   ", item)
        total += len(mismatches) + int(missing)
    print("TOTAL mismatches:", total)
    return int(total > 0)


if __name__ == "__main__":
    sys.exit(main())
