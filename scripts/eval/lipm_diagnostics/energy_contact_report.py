"""Summarise energy_contact.py outputs: Ours vs RGGP / LPGP.

Per trajectory, over the common window where both compared policies are still alive:
  power (W)        mean sum_j |tau_j qd_j| (legs, wheels)
  energy per metre sum |P| dt / sum v dt (moving trajectories: mean speed >= 0.2 m/s)
  CoT              energy per metre / (m g), m = per-env robot mass after DR
  wheel saturation fraction of wheel-actuator samples at >= 99% of the 40 N m limit
  airborne         fraction of steps with at least one wheel off the ground,
                   split by local terrain relief under the base (< 2 cm: locally
                   flat, >= 5 cm: locally rough) and by first-command direction.
Paired Wilcoxon signed-rank tests over identical trajectories.

Run from the repo root:
  .venv/bin/python scripts/eval/lipm_diagnostics/energy_contact_report.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("logs/lipm_eval/full")
DIAG = OUT / "diagnostics" / "energy_contact"
TERRAINS = ("flat", "random_rough", "pyramid_stair", "stepping_stones", "discrete_obstacles")
RUNS = ("Ours", "RGGP", "LPGP")
G = 9.81


def direction_classes(terrain: str) -> np.ndarray:
    data = np.load(OUT / "raw" / terrain / "Ours" / "timeseries.npz")
    cmd = data["command_w"][0, :, :2]
    heading = data["heading_target"][0]
    vx = np.cos(heading) * cmd[:, 0] + np.sin(heading) * cmd[:, 1]
    vy = -np.sin(heading) * cmd[:, 0] + np.cos(heading) * cmd[:, 1]
    return np.where(
        data["is_standing"][0], "standing",
        np.where(np.abs(vx) >= np.abs(vy), "fore_aft", "lateral"),
    )


def masked_mean(values, mask):
    with np.errstate(all="ignore"):
        return np.where(mask.any(0), np.where(mask, values, 0.0).sum(0) / mask.sum(0), np.nan)


def trajectory_metrics(d, window) -> dict[str, np.ndarray]:
    dt = float(d["step_dt"])
    power = d["power_leg"] + d["power_wheel"]
    airborne = ~d["wheels_in_contact"].all(-1)
    relief = d["relief"]
    distance = np.where(window, d["speed"], 0.0).sum(0) * dt
    energy = np.where(window, power, 0.0).sum(0) * dt
    mean_speed = masked_mean(d["speed"], window)
    moving = mean_speed >= 0.2
    with np.errstate(all="ignore"):
        per_metre = np.where(moving & (distance > 0), energy / distance, np.nan)
    return {
        "power_total_W": masked_mean(power, window),
        "power_leg_W": masked_mean(d["power_leg"], window),
        "power_wheel_W": masked_mean(d["power_wheel"], window),
        "energy_per_m_J": per_metre,
        "CoT": per_metre / (d["mass_kg"] * G),
        "wheel_torque_saturation": masked_mean(d["wheel_saturated_frac"], window),
        "airborne": masked_mean(airborne, window),
        "airborne_locally_flat": masked_mean(airborne, window & (relief < 0.02)),
        "airborne_locally_rough": masked_mean(airborne, window & (relief >= 0.05)),
        "mean_speed": mean_speed,
    }


lines = ["# Energy and wheel-contact diagnostics (common alive window, paired)", ""]
for terrain in TERRAINS:
    runs = {r: np.load(DIAG / f"{terrain}__{r}.npz") for r in RUNS}
    classes = direction_classes(terrain)
    lines += [f"## {terrain}", "",
              "| Metric | Ours | RGGP | LPGP | Ours vs RGGP (Ours-better frac, p) | "
              "Ours vs LPGP (Ours-better frac, p) |",
              "|---|---:|---:|---:|---:|---:|"]
    window = runs["Ours"]["valid"] & runs["RGGP"]["valid"] & runs["LPGP"]["valid"]
    metrics = {r: trajectory_metrics(runs[r], window) for r in RUNS}
    names = list(metrics["Ours"]) + ["airborne_fore_aft_cmd", "airborne_lateral_cmd"]
    for r in RUNS:
        for cls in ("fore_aft", "lateral"):
            metrics[r][f"airborne_{cls}_cmd"] = np.where(classes == cls, metrics[r]["airborne"], np.nan)
    for name in names:
        cells = []
        for r in RUNS:
            v = metrics[r][name]
            cells.append(f"{np.nanmean(v):.3g}")
        tests = []
        for other in ("RGGP", "LPGP"):
            a, b = metrics["Ours"][name], metrics[other][name]
            good = np.isfinite(a) & np.isfinite(b)
            diff = a[good] - b[good]
            p = wilcoxon(diff).pvalue if good.sum() > 10 and np.any(diff != 0) else float("nan")
            tests.append(f"{np.mean(diff < 0):.2f}, {p:.1e} (n={good.sum()})")
        lines.append(f"| {name} | " + " | ".join(cells) + " | " + " | ".join(tests) + " |")
    lines.append("")
lines.append("Ours-better frac = fraction of paired trajectories where Ours has the lower value "
             "(lower is better for power, energy, CoT, saturation; airborne rates are descriptive).")

# Flat ground, split by the instantaneous policy-frame command of each run: a wheeled
# biped must step for lateral motion, so unnecessary lifting is judged on straight
# fore/aft driving (|yaw cmd| < 0.1, |lateral cmd| < 0.15, |forward cmd| > 0.3) and standing.
lines += ["", "## flat: wheel lift by command phase (per-run policy-frame command)", "",
          "| Run | straight fore/aft: >=1 wheel airborne (%) | standing (%) | lateral (%) | "
          "straight samples |", "|---|---:|---:|---:|---:|"]
straight_rate = {}
for r in RUNS:
    main = np.load(OUT / "raw" / "flat" / r / "timeseries.npz")
    d = np.load(DIAG / f"flat__{r}.npz")
    cmd = main["command_b"]
    valid = d["valid"] & main["valid"]
    air = ~d["wheels_in_contact"].all(-1)
    straight = valid & (np.abs(cmd[..., 2]) < 0.1) & (np.abs(cmd[..., 1]) < 0.15) & (np.abs(cmd[..., 0]) > 0.3)
    still = valid & (np.linalg.norm(cmd, axis=-1) < 0.05)
    lateral = valid & (np.abs(cmd[..., 1]) > 0.5) & (np.abs(cmd[..., 2]) < 0.1)
    with np.errstate(all="ignore"):
        straight_rate[r] = np.where(
            straight.sum(0) >= 25, (air & straight).sum(0) / straight.sum(0), np.nan
        )
    lines.append(f"| {r} | {100 * air[straight].mean():.1f} | {100 * air[still].mean():.1f} | "
                 f"{100 * air[lateral].mean():.1f} | {int(straight.sum())} |")
for other in ("RGGP", "LPGP"):
    a, b = straight_rate["Ours"], straight_rate[other]
    good = np.isfinite(a) & np.isfinite(b)
    lines.append(
        f"\nPaired per-trajectory straight-driving lift rate, Ours vs {other}: "
        f"{100 * a[good].mean():.1f}% vs {100 * b[good].mean():.1f}% "
        f"(Ours lower in {np.mean(a[good] < b[good]):.2f}, higher in {np.mean(a[good] > b[good]):.2f}; "
        f"Wilcoxon p = {wilcoxon(a[good] - b[good]).pvalue:.1e}, n = {int(good.sum())})"
    )
text = "\n".join(lines) + "\n"
(OUT / "results" / "energy_contact.md").write_text(text)
print(text)
