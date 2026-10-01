"""Paired behaviour metrics: Ours vs RGGP / LPGP on identical trajectories.

All runs of a terrain share terrain, initial state and command sequence, so trajectory
i is paired across checkpoints. Continuous metrics are averaged over the common valid
window (steps where BOTH policies are still alive under the task criterion), which
removes survival-selection bias; a second view uses only trajectories that both
policies complete. Paired Wilcoxon signed-rank tests, Holm-corrected over terrains.

Run from the repo root:
  .venv/bin/python scripts/eval/lipm_diagnostics/benefit_metrics.py
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
OTHERS = ("RGGP", "LPGP")
LEG, WHEEL = slice(0, 6), slice(6, 8)


def per_step(data) -> dict[str, np.ndarray]:
    g = data["projected_gravity_b"].astype(np.float64)
    ang = data["ang_vel_b"].astype(np.float64)
    cmd = data["command_b"].astype(np.float64)
    act = data["actions"].astype(np.float64)
    prev = np.concatenate([np.zeros_like(act[:1]), act[:-1]])
    heading = np.abs(np.angle(np.exp(1j * (data["heading_target"] - data["heading_w"]))))
    return {
        "orientation": np.linalg.norm(g - [0.0, 0.0, -1.0], axis=-1),
        "tilt_deg": np.degrees(np.arccos(np.clip(-g[..., 2], -1.0, 1.0))),
        "roll_pitch_rate": np.linalg.norm(ang[..., :2], axis=-1),
        "lin_vel_err": np.linalg.norm(cmd[..., :2] - data["lin_vel_b"][..., :2], axis=-1),
        "yaw_rate_err": np.abs(cmd[..., 2] - ang[..., 2]),
        # Heading only matters for moving commands (standing commands zero the yaw rate).
        "heading_err_deg": np.where(data["is_standing"], np.nan, np.degrees(heading)),
        "leg_action_rate": np.sum(np.square(act[..., LEG] - prev[..., LEG]), axis=-1),
        "wheel_action_rate": np.sum(np.square(act[..., WHEEL] - prev[..., WHEEL]), axis=-1),
    }


METRICS = (
    "orientation", "tilt_deg", "roll_pitch_rate", "lin_vel_err", "yaw_rate_err",
    "heading_err_deg", "leg_action_rate", "wheel_action_rate",
)


def load(terrain: str, run: str):
    data = np.load(OUT / "raw" / terrain / run / "timeseries.npz")
    steps = per_step(data)
    steps["tilt_max_deg"] = steps["tilt_deg"]  # Aggregated with max instead of mean.
    success = data["fail_step"] < 0
    return data["valid"], steps, success, data


def window_mean(values: np.ndarray, mask: np.ndarray, reduce: str = "mean") -> np.ndarray:
    values = np.where(mask, values, np.nan)
    with np.errstate(all="ignore"):
        if reduce == "max":
            return np.nanmax(np.where(mask.any(0), values, 0.0), axis=0)
        return np.nanmean(values, axis=0)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def direction_classes(data) -> np.ndarray:
    cmd = data["command_w"][0, :, :2]
    heading = data["heading_target"][0]
    vx = np.cos(heading) * cmd[:, 0] + np.sin(heading) * cmd[:, 1]
    vy = -np.sin(heading) * cmd[:, 0] + np.cos(heading) * cmd[:, 1]
    return np.where(
        data["is_standing"][0], "standing",
        np.where(np.abs(vx) >= np.abs(vy), np.where(vx > 0, "forward", "backward"), "lateral"),
    )


rows = []
for terrain in TERRAINS:
    v_o, s_o, ok_o, d_o = load(terrain, "Ours")
    classes = direction_classes(d_o)
    for other in OTHERS:
        v_x, s_x, ok_x, _ = load(terrain, other)
        for view, mask, keep in (
            ("common_window", v_o & v_x, (v_o & v_x).any(0)),
            ("both_success", v_o & v_x, ok_o & ok_x),
        ):
            for metric in METRICS + ("tilt_max_deg",):
                reduce = "max" if metric == "tilt_max_deg" else "mean"
                a = window_mean(s_o[metric], mask, reduce)[keep]
                b = window_mean(s_x[metric], mask, reduce)[keep]
                good = np.isfinite(a) & np.isfinite(b)
                a, b = a[good], b[good]
                diff = a - b
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                rows.append({
                    "terrain": terrain, "other": other, "view": view, "metric": metric,
                    "n_pairs": int(good.sum()), "ours_mean": a.mean(), "other_mean": b.mean(),
                    "rel_diff_pct": 100 * (a.mean() - b.mean()) / b.mean(),
                    "ours_better_frac": float(np.mean(diff < 0)), "p": p,
                })
        for subset in ("all", "forward"):
            sel = np.ones_like(ok_o) if subset == "all" else classes == "forward"
            ours_only = int((ok_o & ~ok_x & sel).sum())
            other_only = int((~ok_o & ok_x & sel).sum())
            n = ours_only + other_only
            rows.append({
                "terrain": terrain, "other": other, "view": f"success_{subset}",
                "metric": "strict_success", "n_pairs": int(sel.sum()),
                "ours_mean": 100 * ok_o[sel].mean(), "other_mean": 100 * ok_x[sel].mean(),
                "rel_diff_pct": np.nan, "ours_better_frac": np.nan,
                "p": binomtest(ours_only, n, 0.5).pvalue if n else 1.0,
            })

# Holm correction across the 11 terrains for every (other, view, metric) family.
families = {}
for row in rows:
    families.setdefault((row["other"], row["view"], row["metric"]), []).append(row)
for family in families.values():
    for row, p in zip(family, holm([r["p"] for r in family])):
        row["p_holm"] = p

with (OUT / "results" / "benefit_metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def verdict(row) -> str:
    if row["p_holm"] >= 0.05:
        return "="
    better = row["ours_mean"] > row["other_mean"] if row["metric"] == "strict_success" else (
        row["ours_mean"] < row["other_mean"]
    )
    return "Ours better" if better else "Ours worse"


lines = ["# Paired behaviour metrics: Ours vs RGGP / LPGP", ""]
for view in ("common_window", "both_success", "success_all", "success_forward"):
    metrics = [m for m in METRICS + ("tilt_max_deg",)] if not view.startswith("success") else ["strict_success"]
    for other in OTHERS:
        lines += [f"## {view}: Ours vs {other}", "",
                  "| Metric | " + " | ".join(TERRAINS) + " |",
                  "|---|" + "---|" * len(TERRAINS)]
        for metric in metrics:
            cells = []
            for terrain in TERRAINS:
                r = next(x for x in rows if (x["terrain"], x["other"], x["view"], x["metric"])
                         == (terrain, other, view, metric))
                mark = {"Ours better": "+", "Ours worse": "-", "=": "~"}[verdict(r)]
                cells.append(f"{r['ours_mean']:.3g} vs {r['other_mean']:.3g} {mark}")
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")
lines.append("+ Ours significantly better, - significantly worse, ~ not significant "
             "(paired test, Holm over 11 terrains, alpha 0.05).")
(OUT / "results" / "benefit_metrics.md").write_text("\n".join(lines) + "\n")

# Compact console summary: count of terrains where Ours is significantly better/worse.
for view in ("common_window", "both_success", "success_all", "success_forward"):
    for other in OTHERS:
        metrics = METRICS + ("tilt_max_deg",) if not view.startswith("success") else ("strict_success",)
        for metric in metrics:
            fam = [r for r in rows if (r["other"], r["view"], r["metric"]) == (other, view, metric)]
            verdicts = [verdict(r) for r in fam]
            better = [r["terrain"] for r, v in zip(fam, verdicts) if v == "Ours better"]
            worse = [r["terrain"] for r, v in zip(fam, verdicts) if v == "Ours worse"]
            print(f"{view:15s} vs {other}: {metric:18s} better {len(better):2d} worse {len(worse):2d}"
                  f"  better={better}  worse={worse}")
