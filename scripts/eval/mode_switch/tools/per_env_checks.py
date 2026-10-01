"""Checks on mode_switch_v2 rollout npz files (smoke test or full run).

Per file: terrain fingerprint equals the CPU audit's (so every terrain row has the audited
difficulty), every env starts inside the course of its recorded (level, lane) and the level's
difficulty is the one of that row, commanded heading-frame v_x within the preset range and
v_y = 0, and the policy-frame command right after reset vs after the first step. Reported
without failing: robots whose lateral offset from the lane centre reaches LANE_BAND before the
course end (they leave their lane instance; the policy does not correct lateral drift).
Run (repo root):
  uv run --no-project --python 3.13 --with numpy python scripts/eval/mode_switch/tools/per_env_checks.py DIR
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/eval"))
import mode_switch_eval as M  # noqa: E402

LEVELS = 4
LANE_BAND = 1.5  # m from the lane centre (lane half-width 1.8 m).


def main(directory: str) -> int:
    audit = json.loads((ROOT / "logs/lipm_eval/mode_switch_v2/terrain_audit/terrain_audit.json").read_text())
    difficulties = [(k + 1) / LEVELS for k in range(LEVELS)]  # level_difficulties(4) without lipm_eval
    bad = 0
    for path in sorted(Path(directory).glob("*__*__*.npz")):
        rough, command, ckpt = path.stem.split("__")
        d = np.load(path)
        pos = d["init_root_state"][:, :2]
        row = np.floor((pos[:, 0] + LEVELS * M.COURSE_END / 2) / M.COURSE_END).astype(int)
        lane = np.floor((pos[:, 1] + M.LANES * M.WIDTH / 2) / M.WIDTH).astype(int)
        local = pos - np.stack([row * M.COURSE_END - LEVELS * M.COURSE_END / 2, lane * M.WIDTH - M.LANES * M.WIDTH / 2], -1)
        lo, hi = M.COMMANDS[command]["lin_vel_x"]
        vx, vy = d["cmd_vel_h"][:, 0], d["cmd_vel_h"][:, 1]
        checks = {
            "terrain_sha == CPU audit": str(d["terrain_sha"]) == audit[rough]["terrain_fingerprint_4x16"],
            "spawn row == level": bool(np.array_equal(row, d["level"])),
            "spawn lane == lane": bool(np.array_equal(lane, d["lane"])),
            "difficulty == row difficulty": bool(np.array_equal(d["difficulty"], np.array(difficulties)[row])),
            "spawn within 0.3 m of (1.0, 1.8)": bool(np.all(np.abs(local - [1.0, 1.8]) <= 0.3 + 1e-4)),
            f"v_x in [{lo}, {hi}]": bool(np.all((vx >= lo) & (vx <= hi))),
            "v_y == 0": bool(np.all(vy == 0)),
            "|command_b| after reset == v_x": bool(np.allclose(np.linalg.norm(d["init_command_b"][:, :2], axis=-1), vx, atol=1e-5)),
        }
        diff = np.abs(d["command_b"][0].astype(float) - d["init_command_b"])
        counts = np.unique(d["level"] * M.LANES + d["lane"], return_counts=True)[1]
        print(f"{path.name}: envs {len(vx)}, per (level, lane) {counts.min()}-{counts.max()}, v_x {vx.min():.3f}-{vx.max():.3f}, "
              f"max |command_b[step 0] - command_b[reset]| = {diff.max():.4f}")
        for name, ok in checks.items():
            if not ok:
                bad += 1
                print("   FAIL", name)
        print("   all checks passed" if all(checks.values()) else "")
        if "course_y" in d:
            data = {k: d[k] for k in ("valid", "course_x", "course_y", "course_end", "windows")}
            dy = np.array([M.lateral_metrics(data, i)["max_abs_dy"] for i in range(len(vx))])
            out = int(np.sum(dy >= LANE_BAND))
            print(f"   {'WARN' if out else 'ok  '} {out} robots reach |y - lane centre| >= {LANE_BAND} m before the "
                  f"course end; max |y - centre| median {np.nanmedian(dy):.3f}, p99 {np.nanpercentile(dy, 99):.3f}, "
                  f"max {np.nanmax(dy):.3f} m")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
