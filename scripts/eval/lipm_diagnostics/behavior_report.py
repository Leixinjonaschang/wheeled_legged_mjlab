"""Multi-terrain gait / stability / efficiency metrics from behavior_record.py outputs.

Per trajectory (alive window of the task criterion, moving commands only where noted),
pooled over the 10 complex terrain classes (flat reported separately):
  rough_lift   % of steps with >= 1 wheel airborne while the roughness gate is ON
               (same gate as the reward: lambda > 0.65), moving command (|c| > 0.05)
  smooth_lift  % of steps with >= 1 wheel airborne while the gate is OFF, under
               rollable commands (|v_y cmd| < 0.15 m/s, |yaw cmd| < 0.1 rad/s,
               |v_x cmd| > 0.3 m/s): lateral motion needs stepping on any terrain
  selectivity  rough_lift - smooth_lift (percentage points), trajectories with both
  cot          sum |P| dt / (m g distance), trajectories with mean speed >= 0.2 m/s
  peak_tilt    max base tilt (deg); roll_pitch_rate mean ||(w_x, w_y)|| (rad/s)
Paired comparisons with Ours on identical trajectories (Wilcoxon signed-rank).

Run from the repo root:
  uv run --no-project --python 3.13 --with numpy --with scipy \
      python scripts/eval/lipm_diagnostics/behavior_report.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("logs/lipm_eval/full")
DIAG = OUT / "diagnostics" / "behavior"
TERRAINS = ("flat", "discrete_obstacles", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv",
            "pyramid_stair", "pyramid_stair_inv", "random_stairs", "random_spread", "stepping_stones",
            "tilted_grid")
RUNS = ("Ours", "RGGP", "LPGP")


def per_trajectory(d) -> dict[str, np.ndarray]:
    valid = d["valid"]
    dt = float(d["step_dt"])
    cmd = d["command_b"].astype(np.float32)
    moving = np.linalg.norm(cmd, axis=-1) > 0.05
    rollable = (np.abs(cmd[..., 1]) < 0.15) & (np.abs(cmd[..., 2]) < 0.1) & (np.abs(cmd[..., 0]) > 0.3)
    air = ~d["wheel_contact"].all(-1)
    rough = d["rough"]

    def rate(mask, min_count=25):
        count = mask.sum(0)
        with np.errstate(all="ignore"):
            return np.where(count >= min_count, 100.0 * (air & mask).sum(0) / count, np.nan)

    rough_lift = rate(valid & rough & moving)
    smooth_lift = rate(valid & ~rough & rollable)
    g = d["gravity_b"].astype(np.float32)
    tilt = np.degrees(np.arccos(np.clip(-g[..., 2], -1, 1)))
    ang = np.linalg.norm(d["ang_vel_b"][..., :2].astype(np.float32), axis=-1)
    power = d["power_leg"].astype(np.float32) + d["power_wheel"].astype(np.float32)
    speed = d["speed"].astype(np.float32)
    n = valid.sum(0)
    with np.errstate(all="ignore"):
        distance = np.where(valid, speed, 0).sum(0) * dt
        mean_speed = np.where(valid, speed, 0).sum(0) / n
        cot = np.where((mean_speed >= 0.2) & (distance > 0.5),
                       np.where(valid, power, 0).sum(0) * dt / distance / (d["mass_kg"] * 9.81), np.nan)
        return {
            "rough_lift": rough_lift,
            "smooth_lift": smooth_lift,
            "selectivity": rough_lift - smooth_lift,
            "cot": cot,
            "peak_tilt": np.where(n > 0, np.where(valid, tilt, -np.inf).max(0), np.nan),
            "roll_pitch_rate": np.where(n > 0, np.where(valid, ang, 0).sum(0) / n, np.nan),
        }


def main():
    rows = {}
    for terrain in TERRAINS:
        rows[terrain] = {r: per_trajectory(dict(np.load(DIAG / f"{terrain}__{r}.npz"))) for r in RUNS}
    report = {}
    groups = {"complex (10 classes)": TERRAINS[1:], "flat": ("flat",),
              **{t: (t,) for t in TERRAINS[1:]}}
    for group, terrains in groups.items():
        entry = {}
        for metric in rows["flat"]["Ours"]:
            vals = {r: np.concatenate([rows[t][r][metric] for t in terrains]) for r in RUNS}
            entry[metric] = {r: float(np.nanmean(vals[r])) for r in RUNS}
            for other in ("RGGP", "LPGP"):
                a, b = vals["Ours"], vals[other]
                ok = np.isfinite(a) & np.isfinite(b)
                diff = a[ok] - b[ok]
                entry[metric][f"p_vs_{other}"] = float(wilcoxon(diff).pvalue) if ok.sum() > 20 and np.any(diff) else float("nan")
                entry[metric][f"n_vs_{other}"] = int(ok.sum())
        report[group] = entry
    (OUT / "results" / "behavior_metrics.json").write_text(json.dumps(report, indent=1))
    for group in ("complex (10 classes)", "flat", "stepping_stones", "pyramid_stair_inv", "discrete_obstacles"):
        print("==", group)
        for metric, v in report[group].items():
            print(f"   {metric:16s} Ours {v['Ours']:8.3f} RGGP {v['RGGP']:8.3f} LPGP {v['LPGP']:8.3f} "
                  f"p(RGGP) {v['p_vs_RGGP']:.1e} p(LPGP) {v['p_vs_LPGP']:.1e}")


if __name__ == "__main__":
    main()
