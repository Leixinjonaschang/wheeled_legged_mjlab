"""Unified paper table: mode switching + multi-terrain adaptation (Ours vs ablations).

Columns (all paired against Ours on identical trajectories):
  Flat-rough-flat course, forward commands, rough section in {tilted grid, random boxes,
  discrete obstacles} at maximum training difficulty, 3 x 1024 trajectories:
    switch success (%)   rolls on flat-in, both wheels step on rough, rolls on flat-out,
                         and completes without failure              (exact McNemar)
    completion (%)       reaches the course end without failure     (exact McNemar)
    CoT                  sum |tau qd| dt / (m g distance)            (Wilcoxon)
  Ten complex terrain classes at maximum training difficulty, omnidirectional commands,
  10 x 1024 trajectories:
    rough lift (%)       >= 1 wheel airborne while the reward's roughness gate is on
                         (moving commands)                           (Wilcoxon)
    flat lift (%)        >= 1 wheel airborne on flat ground under straight-driving
                         commands (flat terrain class)               (Wilcoxon)
    orientation error    ||g_b - [0,0,-1]||, time-averaged per trajectory (Wilcoxon)
Also reported (notes): omnidirectional success on the ten classes and on stepping stones.

Run from the repo root:
  uv run --no-project --python 3.13 --with numpy --with scipy \
      python scripts/eval/lipm_diagnostics/unified_table.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path("logs/lipm_eval")
FULL = ROOT / "full"
RESULTS = FULL / "results"
COURSE = ROOT / "mode_switch"
ROUGH_TYPES = ("tilted_grid", "random_spread", "discrete_obstacles")
COMPLEX = ("discrete_obstacles", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv",
           "pyramid_stair", "pyramid_stair_inv", "random_stairs", "random_spread",
           "stepping_stones", "tilted_grid")
RUNS = ("RGGP", "LPGP", "Ours")
LABEL = {"RGGP": "Ours w/o RG", "LPGP": "Ours w/o LP", "Ours": "Ours"}


def course_rows():
    with (COURSE / "trajectories.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["command"] == "forward" and r["rough"] in ROUGH_TYPES]
    out = {}
    for run in RUNS:
        sel = sorted((r for r in rows if r["ckpt"] == run), key=lambda r: (r["rough"], int(r["env"])))
        out[run] = {
            "switch": np.array([r["switch_success"] == "True" for r in sel]),
            "completion": np.array([r["completed"] == "True" for r in sel]),
            "cot": np.array([float(r["cot"]) for r in sel]),
        }
    return out


def behavior_rows():
    import importlib.util

    spec = importlib.util.spec_from_file_location("behavior_report", Path(__file__).resolve().parent / "behavior_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = {run: {"rough_lift": [], "flat_lift": None} for run in RUNS}
    for run in RUNS:
        for terrain in COMPLEX:
            m = module.per_trajectory(dict(np.load(FULL / "diagnostics" / "behavior" / f"{terrain}__{run}.npz")))
            out[run]["rough_lift"].append(m["rough_lift"])
        out[run]["rough_lift"] = np.concatenate(out[run]["rough_lift"])
        flat = module.per_trajectory(dict(np.load(FULL / "diagnostics" / "behavior" / f"flat__{run}.npz")))
        out[run]["flat_lift"] = flat["smooth_lift"]
    return out


def main_eval_rows():
    out = {run: {"orientation": [], "success": [], "stepping_success": None} for run in RUNS}
    for run in RUNS:
        for terrain in COMPLEX:
            with (FULL / "raw" / terrain / run / "trajectories.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            out[run]["orientation"].append(np.array([
                float(r["orientation_error"]) if int(r["n_valid_samples"]) > 0 else np.nan for r in rows]))
            success = np.array([r["success"] == "True" for r in rows])
            out[run]["success"].append(success)
            if terrain == "stepping_stones":
                out[run]["stepping_success"] = success
        out[run]["orientation"] = np.concatenate(out[run]["orientation"])
        out[run]["success"] = np.concatenate(out[run]["success"])
    return out


def paired_p(a, b, binary):
    if binary:
        only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
        return binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    ok = np.isfinite(a) & np.isfinite(b)
    diff = a[ok] - b[ok]
    return float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0


COLUMNS = (  # key, source, header, unit, higher_is_better, binary, scale, digits
    ("switch", "course", "Switch success", "\\%", True, True, 100.0, 1),
    ("completion", "course", "Completion", "\\%", True, True, 100.0, 1),
    ("cot", "course", "CoT", "--", False, False, 1.0, 2),
    ("rough_lift", "behavior", "Rough lift", "\\%", True, False, 1.0, 1),
    ("flat_lift", "behavior", "Flat lift", "\\%", False, False, 1.0, 1),
    ("orientation", "main", "Orientation error", "--", False, False, 1.0, 3),
)
EXTRA = (("success", "main", "Success, 10 classes (\\%)", True, True, 100.0, 1),
         ("stepping_success", "main", "Success, stepping stones (\\%)", True, True, 100.0, 1))


def main():
    sources = {"course": course_rows(), "behavior": behavior_rows(), "main": main_eval_rows()}
    table = {}
    for key, source, header, _, higher, binary, scale, digits in COLUMNS + tuple(
            (k, s, h, "", hi, b, sc, d) for k, s, h, hi, b, sc, d in EXTRA):
        values = {run: sources[source][run][key] for run in RUNS}
        means = {run: scale * float(np.nanmean(values[run].astype(float))) for run in RUNS}
        shown = {run: round(means[run], digits) for run in RUNS}
        rank = {run: 1 + sum((shown[o] > shown[run]) if higher else (shown[o] < shown[run]) for o in RUNS)
                for run in RUNS}
        table[key] = {
            "header": header, "digits": digits, "higher_is_better": higher, "mean": means, "rank": rank,
            "p_vs_Ours": {run: paired_p(values["Ours"], values[run], binary) for run in ("RGGP", "LPGP")},
            "n": {run: int(np.isfinite(values[run].astype(float)).sum()) for run in RUNS},
        }
    (RESULTS / "unified_table.json").write_text(json.dumps(table, indent=1))

    def cell(key, run):
        entry = table[key]
        text = f"{entry['mean'][run]:.{entry['digits']}f}"
        if entry["rank"][run] == 1:
            return f"\\textbf{{{text}}}"
        if entry["rank"][run] == 2:
            return f"\\underline{{{text}}}"
        return text

    arrows = {True: "$\\uparrow$", False: "$\\downarrow$"}
    tex = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Terrain-adaptive locomotion and multi-terrain performance. Best and second-best "
        "results among the evaluated methods are marked in bold and underlined, respectively.}",
        "    \\label{tab:unified_results}",
        "    \\footnotesize",
        "    \\setlength{\\tabcolsep}{5.0pt}",
        "    \\renewcommand{\\arraystretch}{1.12}",
        "    \\begin{tabular}{lcccccc}",
        "        \\toprule",
        "        & \\multicolumn{3}{c}{\\textbf{Flat--rough--flat course}} "
        "& \\multicolumn{3}{c}{\\textbf{Ten terrain classes}} \\\\",
        "        \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "        \\textbf{Method} & " + " & ".join(
            f"\\textbf{{{c[2]}}}{arrows[c[4]]}" for c in COLUMNS) + " \\\\",
        "        & " + " & ".join(f"({c[3]})" for c in COLUMNS) + " \\\\",
        "        \\midrule",
        "        Blind & " + " & ".join("--" for _ in COLUMNS) + " \\\\",
        "        Vision-CTS & " + " & ".join("--" for _ in COLUMNS) + " \\\\",
    ]
    for run in RUNS:
        tex.append(f"        {LABEL[run]} & " + " & ".join(cell(c[0], run) for c in COLUMNS) + " \\\\")
    tex += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\par\\vspace{1mm}",
        "    \\parbox{\\textwidth}{\\scriptsize Single training seed per method. Course: 4~m flat, 3.6~m rough "
        "section (tilted grid, random boxes, or discrete obstacles at the maximum training difficulty), "
        "4.4~m flat; forward commands of 0.6--1.0~m/s; $3\\times1024$ trajectories. A switch is successful when "
        "the robot rolls without wheel lifting on the first flat segment, lifts both wheels on the rough "
        "section, returns to rolling on the last flat segment, and completes the course without falling or "
        "non-wheel contact; a lift is an airborne phase of at least 0.06~s reaching 3~cm clearance. "
        "Ten terrain classes: maximum training difficulty, randomized omnidirectional commands, "
        "$10\\times1024$ trajectories of 10~s. Rough lift: time share with a wheel airborne while the "
        "roughness gate of the reward is active; flat lift: the same on flat ground under straight-driving "
        "commands; orientation error: $\\lVert \\mathbf{g}_b-[0,0,-1]^{\\top}\\rVert_2$. "
        "--: not evaluated.}",
        "\\end{table*}",
    ]
    (RESULTS / "unified_table.tex").write_text("\n".join(tex) + "\n")

    md = ["| Method | " + " | ".join(f"{c[2]} {'↑' if c[4] else '↓'}" for c in COLUMNS)
          + " | " + " | ".join(e[2].replace("\\%", "%") for e in EXTRA) + " |",
          "|---|" + "---:|" * (len(COLUMNS) + len(EXTRA))]
    for run in RUNS:
        cells = []
        for key, *_ in COLUMNS + tuple((e[0],) for e in EXTRA):
            entry = table[key]
            text = f"{entry['mean'][run]:.{entry['digits']}f}"
            text = f"**{text}**" if entry["rank"][run] == 1 else text
            if run != "Ours":
                p = entry["p_vs_Ours"][run]
                text += " †" if p < 0.05 else " (n.s.)"
            cells.append(text)
        md.append(f"| {LABEL[run]} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("† difference to Ours significant (paired test, p < 0.05); n.s. not significant.")
    (RESULTS / "unified_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    for key, entry in table.items():
        print(key, {r: round(v, 4) for r, v in entry["mean"].items()},
              {r: f"{p:.1e}" for r, p in entry["p_vs_Ours"].items()}, entry["n"])


if __name__ == "__main__":
    main()
