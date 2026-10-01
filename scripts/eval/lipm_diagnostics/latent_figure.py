"""Figures: effect of latent dynamics prediction on the learned latent space.

Main figure (latent_separability.pdf/png), Ours (with latent prediction) vs LPGP (without),
all encoders evaluated on identical observations:
  (a) t-SNE of teacher latents, small multiples: one column per terrain family (family
      highlighted, all other samples grey), one row per encoder;
  (b) linear-probe and (c) kNN balanced accuracy for four label sets;
  (d) linear predictability (R^2) of future teacher latents from latent + actions.
Supplementary figure (latent_separability_all_metrics.pdf/png): every pre-specified
cluster metric for the teacher latent on the 8 terrain families, including the
compactness metrics that do not favour Ours.

Colours: validated two-slot categorical palette (blue = with latent prediction, orange =
without); grey is context only. Legends plus direct labels carry identity.

Run from the repo root (after latent_analysis.py):
  uv run --no-project --python 3.13 --with numpy --with matplotlib \
      python scripts/eval/lipm_diagnostics/latent_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RESULTS = Path("logs/lipm_eval/full/results")
FAMILIES = ("flat", "rough", "slope", "stairs", "discrete\nobstacles", "boxes",
            "stepping\nstones", "tilted\ngrid")
ENCODERS = (("LPGP", "w/o LP", "#eb6834"), ("Ours", "with LP (Ours)", "#2a78d6"))
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LABEL_SETS = (
    ("teacher_families", "terrain family (8), teacher", 1 / 8),
    ("teacher_classes", "terrain class (11), teacher", 1 / 11),
    ("teacher_mode", "rolling vs. lifted, teacher", 1 / 2),
    ("student_families", "terrain family (8), student", 1 / 8),
)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

report = json.loads((RESULTS / "latent_separability.json").read_text())
tsne = np.load(RESULTS / "latent_tsne.npz")
family = tsne["family"]


def style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, color=AXIS, labelsize=6)


def tsne_block(fig, spec):
    grid = spec.subgridspec(2, len(FAMILIES), wspace=0.06, hspace=0.08)
    for r, (encoder, label, color) in enumerate(ENCODERS):
        xy = tsne[f"tsne_{encoder}"]
        for c, fam in enumerate(FAMILIES):
            ax = fig.add_subplot(grid[r, c])
            mask = family == fam.replace("\n", " ")
            ax.scatter(xy[~mask, 0], xy[~mask, 1], s=0.6, c=GRID, linewidths=0, rasterized=True)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=1.2, c=color, linewidths=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if r == 0:
                ax.set_title(fam, fontsize=6.3, color=INK2, pad=3)
            if c == 0:
                ax.set_ylabel(label, fontsize=6.5, color=INK, labelpad=3)


def dot_whisker(ax, entries, fmt, chance=None):
    """entries: list of (row label, {encoder: (mean, low, high)})."""
    for i, (row, values) in enumerate(entries):
        for k, (encoder, _, color) in enumerate(ENCODERS):
            mean, low, high = values[encoder]
            y = i + (0.18 if k == 1 else -0.18)
            ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(mean, y, "o", color=color, markersize=4.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.0)
            ax.annotate(fmt.format(mean), (high, y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=5.8, color=INK2)
        if chance is not None and chance[i] is not None:
            ax.plot([chance[i]] * 2, [i - 0.4, i + 0.4], color=AXIS, linewidth=0.8)
    ax.set_yticks(range(len(entries)))
    ax.set_ylim(len(entries) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)


def main_figure():
    fig = plt.figure(figsize=(7.16, 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.9, 1.35), hspace=0.38)
    tsne_block(fig, outer[0])
    bottom = outer[1].subgridspec(1, 3, width_ratios=(1.25, 1.0, 1.0), wspace=0.12)

    for col, (metric, title) in enumerate((("linear_probe", "(b) Linear-probe balanced accuracy ↑"),
                                           ("knn", "(c) kNN balanced accuracy ↑"))):
        ax = fig.add_subplot(bottom[0, col])
        entries = [(label, {e: (report[view][e][metric]["mean"], *report[view][e][metric]["ci95"])
                            for e, _, _ in ENCODERS}) for view, label, _ in LABEL_SETS]
        dot_whisker(ax, entries, "{:.2f}")
        ax.set_yticklabels([label for _, label, _ in LABEL_SETS] if col == 0 else [], color=INK2,
                           fontsize=6.2)
        ax.set_xlim(0.40, 1.0)
        style(ax)
        ax.set_title(title, fontsize=7, color=INK, loc="left", pad=4)

    ax = fig.add_subplot(bottom[0, 2])
    horizons = (1, 5, 10)
    pred = report["teacher_predictability_r2"]
    for encoder, label, color in ENCODERS:
        means = [pred[encoder][str(h)]["mean"] for h in horizons]
        lows = [min(pred[encoder][str(h)]["values"]) for h in horizons]
        highs = [max(pred[encoder][str(h)]["values"]) for h in horizons]
        ax.plot(horizons, means, color=color, linewidth=2, solid_capstyle="round")
        ax.vlines(horizons, lows, highs, color=color, linewidth=2)
        ax.plot(horizons, means, "o", color=color, markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.0)
        ax.annotate(f"{means[-1]:.2f}", (horizons[-1], means[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=5.8, color=INK2)
    ax.set_xticks(horizons)
    ax.set_xticklabels(["1", "5", "10"])
    ax.set_xlabel("prediction horizon (policy steps, 0.02 s)", fontsize=6.2)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.6, 1.0)
    style(ax, "y")
    ax.set_title("(d) Latent predictability, R² ↑", fontsize=7, color=INK, loc="left", pad=4)
    ax.yaxis.tick_right()

    handles = [Line2D([], [], color=c, marker="o", linewidth=2, markersize=4.5, label=l)
               for _, l, c in ENCODERS]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2,
               frameon=False, fontsize=6.5)
    fig.text(0.01, 0.995, "(a) t-SNE of teacher latents on identical observations; each column "
             "highlights one terrain family", fontsize=7, color=INK, va="top")
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"latent_separability.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def all_metrics_figure():
    metrics = (("silhouette", "Silhouette (cosine) ↑", "{:.3f}"),
               ("linear_probe", "Linear-probe bal. acc. ↑", "{:.3f}"),
               ("knn", "kNN bal. acc. ↑", "{:.3f}"),
               ("fisher", "Fisher ratio ↑", "{:.3f}"),
               ("davies_bouldin", "Davies–Bouldin ↓", "{:.2f}"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.16, 1.35), gridspec_kw={"wspace": 0.5})
    view = report["teacher_families"]
    for i, (key, title, fmt) in enumerate(metrics):
        ax = axes[i]
        dot_whisker(ax, [("", {e: (view[e][key]["mean"], *view[e][key]["ci95"])
                               for e, _, _ in ENCODERS})], fmt)
        ax.set_yticks([])
        lows = [view[e][key]["ci95"][0] for e, _, _ in ENCODERS]
        highs = [view[e][key]["ci95"][1] for e, _, _ in ENCODERS]
        span = max(highs) - min(lows)
        ax.set_xlim(min(lows) - 0.15 * span, max(highs) + 0.6 * span)
        ax.locator_params(axis="x", nbins=3)
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=6.8, color=INK, pad=4)
    handles = [Line2D([], [], color=c, marker="o", linewidth=2, markersize=4.5, label=l)
               for _, l, c in ENCODERS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2,
               frameon=False, fontsize=6.5)
    fig.suptitle("Teacher latent, 8 terrain families: all pre-specified separability metrics "
                 "(point: mean; line: 95% interval)", fontsize=7, y=1.12)
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"latent_separability_all_metrics.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


main_figure()
all_metrics_figure()
print("saved", RESULTS / "latent_separability.pdf", RESULTS / "latent_separability_all_metrics.pdf")
