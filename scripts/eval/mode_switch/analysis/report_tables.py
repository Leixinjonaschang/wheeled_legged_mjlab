"""Markdown tables for REPORT.md, generated from existing analysis outputs. No statistics are computed here;
values are only formatted.

Inputs: favourable_settings.csv / unfavourable_settings.csv / rank_settings.json (rank_settings.py),
mc_stability_rows.csv / mc_stability.json (mc_stability.py: 10 bootstrap seeds, 1e6-flip permutation p for pooled
rows, Holm over the 220 candidates), split_half_rows.csv / split_half.json (split_half.py).

Conventions used in every table:
  * p = raw lane sign-flip permutation p (exact for <= 16 lanes, 1e6 random flips otherwise; mc_stability.py).
  * H = Holm-adjusted lane sign-flip permutation p over all 220 candidates per (metric, comparison), recomputed by
    mc_stability.py with exact enumeration (<= 16 lanes) or 1,000,000 random flips (pooled rows). The 10,000-flip
    values in favourable_settings.csv give the same < 0.05 verdict for every robust row; their pooled values
    (0.015 - 0.022) are set by the 1/10001 resolution, not by the data.
  * ▲ / ▼ robust (pre-specified: CI excludes 0 and |diff| > noise_floor.json p90) Ours better / worse;
    △ / ▽ CI excludes 0 but |diff| <= the cell-level noise floor (not robust by the rule); · CI contains 0.
  * † the robust flag is not the same in all 10 bootstrap seeds of mc_stability.py (MC-borderline); (k/10) = number
    of seeds in which it is robust.

Run (repo root, after rank_settings.py, mc_stability.py and split_half.py):
  uv run --no-project --python 3.13 --with numpy --with scipy --with matplotlib --with pandas \
      python scripts/eval/mode_switch/analysis/report_tables.py \
      --splice logs/lipm_eval/mode_switch_v2/analysis/REPORT.md > logs/lipm_eval/mode_switch_v2/analysis/report_tables.md
(--splice PATH also rewrites the <!-- BEGIN TABLE X --> ... <!-- END TABLE X --> blocks of REPORT.md.)
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "logs/lipm_eval/mode_switch_v2/analysis"  # data dir, not this script
KEY = ["scope", "terrain", "difficulty", "speed_bin"]
SCOPE_ZH = {"all": "全部", "difficulty": "难度", "terrain": "地形", "speed_bin": "速度段", "difficulty_speed": "难度×速度",
            "terrain_speed": "地形×速度", "cell": "格", "cell_speed": "格×速度"}
LAB = {"RGGP": "w/o RG", "LPGP": "w/o LP"}
PYRAMIDS = {"pyramid_stair", "pyramid_stair_inv", "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_stairs"}


def dnorm(v) -> str:
    return v if str(v) == "ALL" else f"{float(v):g}"


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(HERE / name, dtype={"difficulty": str})
    df["difficulty"] = df.difficulty.map(dnorm)
    return df


def setting(r) -> str:
    parts = []
    if r.terrain != "ALL":
        parts.append(str(r.terrain))
    if str(r.difficulty) != "ALL":
        parts.append(f"d={r.difficulty}")
    if r.speed_bin != "ALL":
        parts.append(f"v∈{r.speed_bin}")
    return ", ".join(parts) if parts else "全部 40 格"


def mark(r, s, b) -> str:
    d = getattr(r, f"{s}_diff_{b}")
    lo, hi = getattr(r, f"{s}_diff_{b}_lo"), getattr(r, f"{s}_diff_{b}_hi")
    excl = lo > 0 or hi < 0
    if str(getattr(r, f"{s}_robust_{b}")) == "True":
        m = "▲" if d > 0 else "▼"
    elif excl:
        m = "△" if d > 0 else "▽"
    else:
        m = "·"
    frac = getattr(r, f"{s}_robust_frac_{b}", np.nan)
    if pd.notna(frac) and 0 < frac < 1:
        m += f"†({int(round(frac * 10))}/10)"
    return m


def hp(v) -> str:
    if pd.isna(v):
        return "–"
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"


def pp(v) -> str:
    if pd.isna(v):
        return "–"
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"


def diff_ci(r, s, b) -> str:
    d, lo, hi = getattr(r, f"{s}_diff_{b}"), getattr(r, f"{s}_diff_{b}_lo"), getattr(r, f"{s}_diff_{b}_hi")
    return f"{d:+.1f} [{lo:.1f}, {hi:.1f}] {mark(r, s, b)}"


def notes(r) -> str:
    n = []
    if r.comp_floor_ceiling != "informative":
        n.append(f"完成率 {r.comp_floor_ceiling}")
    if r.n_roll_cells_in_group:
        n.append(f"含 {int(r.n_roll_cells_in_group)} 个 roll 格")
    if r.terrain in PYRAMIDS:
        n.append("窄落脚面地形（看 dy 分层）")
    if r.terrain == "stepping_stones" and r.difficulty in ("0.75", "1"):
        n.append("stepping_stones d≥0.75 地板效应")
    if str(r.comp_ci_loss_any) == "True":
        n.append("完成率 CI 显著下降")
    return "；".join(n)


def fav_table(df: pd.DataFrame, rank_col: str = "rank") -> str:
    head = ("| 排名 | 层级 | 设置 | n | Switch O / RG / LP | Δ vs w/o RG [CI] (H) | Δ vs w/o LP [CI] (H) "
            "| 完成率 O / RG / LP | 完成率 Δ vs w/o RG (p; H) | 完成率 Δ vs w/o LP (p; H) | 严格合格 | 备注 |\n"
            "|---|---|---|---:|---|---|---|---|---|---|---|---|")
    rows = [head]
    for r in df.itertuples():
        rows.append(
            f"| {int(getattr(r, rank_col))} | {SCOPE_ZH[r.scope]} | {setting(r)} | {int(r.n_robots)} "
            f"| {r.sw_Ours:.1f} / {r.sw_RGGP:.1f} / {r.sw_LPGP:.1f} "
            f"| {diff_ci(r, 'sw', 'RGGP')} ({hp(r.sw_holm220_hi_RGGP)}) | {diff_ci(r, 'sw', 'LPGP')} ({hp(r.sw_holm220_hi_LPGP)}) "
            f"| {r.comp_Ours:.1f} / {r.comp_RGGP:.1f} / {r.comp_LPGP:.1f} "
            f"| {r.comp_diff_RGGP:+.1f} {mark(r, 'comp', 'RGGP')} ({pp(r.comp_perm_p_hi_RGGP)}; {hp(r.comp_holm220_hi_RGGP)}) "
            f"| {r.comp_diff_LPGP:+.1f} {mark(r, 'comp', 'LPGP')} ({pp(r.comp_perm_p_hi_LPGP)}; {hp(r.comp_holm220_hi_LPGP)}) "
            f"| {'是' if str(r.eligible_strict) == 'True' else '否'} | {notes(r)} |")
    return "\n".join(rows)


def sens_table(df: pd.DataFrame) -> str:
    def dc(r, s, b):
        d, lo, hi = getattr(r, f"{s}_diff_{b}"), getattr(r, f"{s}_diff_{b}_lo"), getattr(r, f"{s}_diff_{b}_hi")
        m = "+" if lo > 0 else ("−" if hi < 0 else "·")
        return f"{d:+.1f} [{lo:.1f}, {hi:.1f}] {m}"

    head = ("| 排名 | 设置 | n | Switch Δ vs w/o RG：3 cm → 仅时长 | Switch Δ vs w/o LP：3 cm → 仅时长 "
            "| 出口恢复滚动 Δ vs w/o RG | 出口恢复滚动 Δ vs w/o LP | min Δ switch：全部 / 偶数 lane / 奇数 lane / 重跑 Ours "
            "| 完成率 min Δ：全部 / 重跑 Ours |\n|---:|---|---:|---|---|---|---|---|---|")
    rows = [head]
    for r in df.itertuples():
        rows.append(
            f"| {int(r.rank)} | {setting(r)} | {int(r.n_robots)} "
            f"| {r.sw_diff_RGGP:+.1f} → {dc(r, 'swdur', 'RGGP')} | {r.sw_diff_LPGP:+.1f} → {dc(r, 'swdur', 'LPGP')} "
            f"| {dc(r, 'rfo', 'RGGP')} | {dc(r, 'rfo', 'LPGP')} "
            f"| {r.sw_min:.1f} / {r.sw_min_even:.1f} / {r.sw_min_odd:.1f} / {r.sw_rep_min:.1f} "
            f"| {r.comp_min:+.1f} / {r.comp_rep_min:+.1f} |")
    return "\n".join(rows)


def loss_table(df: pd.DataFrame) -> str:
    head = ("| # | 层级 | 设置 | 指标 | 对比 | n | Ours / 基线 | Δ [CI] | H (1e6) | H (1e4, csv) | robust 种子数 | 重跑 Δ "
            "| mode success Δ [CI]（见注） | 期望模式 / roll 格数 |\n"
            "|---:|---|---|---|---|---:|---|---|---|---|---:|---:|---|---|")
    rows = [head]
    zh = {"switch_success_pct": "switch", "completion_pct": "完成率"}
    for i, r in enumerate(df.itertuples(), 1):
        mode = (f"{r.mode_success_diff:+.1f} [{r.mode_success_diff_lo:.1f}, {r.mode_success_diff_hi:.1f}]"
                if pd.notna(r.mode_success_diff) else "–")
        em = r.expected_mode if isinstance(r.expected_mode, str) else "–"
        rows.append(
            f"| {i} | {SCOPE_ZH[r.scope]} | {setting(r)} | {zh[r.metric]} | {r.vs} | {int(r.n_robots)} "
            f"| {r.Ours:.1f} / {r.baseline:.1f} | {r.diff:+.1f} [{r.diff_lo:.1f}, {r.diff_hi:.1f}] | {hp(r.holm_hi)} "
            f"| {hp(r.holm220)} | {int(round(r.robust_frac * 10))}/10 | {r.replicate_diff:+.1f} | {mode} "
            f"| {em} / {int(r.n_roll_cells_in_group)} |")
    return "\n".join(rows)


ZH_METRIC = {"switch_success_pct": "切换成功率 (%)", "completion_pct": "完成率 (%)", "cot": "CoT",
             "failure_pct": "失败率 (%)", "roll_flat_in_pct": "入口平地滚动 (%)", "step_rough_pct": "粗糙段双轮抬起 (%)",
             "roll_flat_out_pct": "出口平地恢复滚动 (%)", "flat_lift_rate_pct": "平地离地步占比 (%)",
             "rough_lift_rate_pct": "粗糙段离地步占比 (%)", "peak_tilt_deg": "峰值倾角 (°)", "time_ratio": "time_ratio",
             "mode_success_pct": "门控感知 mode success (%)", "switch_success_dur_pct": "仅时长切换成功率 (%)"}


def overall_table(cells: pd.DataFrame, rep: pd.DataFrame, scope: str = "all", metrics=None) -> str:
    """Pooled rows of cells.csv (+ Ours - Ours rerun at the same scope for scope 'all')."""
    rows = ["| 分组 | 指标 | Ours | w/o RG | w/o LP | Ours − w/o RG [CI] | Ours − w/o LP [CI] | Ours − 重跑 [CI]（同尺度） | 格级噪声 p90 |",
            "|---|---|---:|---:|---:|---|---|---|---:|"]
    sub = cells[cells.scope == scope]
    if metrics is not None:
        sub = sub[sub.metric.isin(metrics)]
    rep = rep.set_index(KEY + ["metric"])
    for r in sub.itertuples():
        nd = 3 if r.metric in ("cot", "time_ratio") else (2 if r.metric == "peak_tilt_deg" else 1)

        def dc(b):
            d, lo, hi = getattr(r, f"diff_{b}"), getattr(r, f"diff_{b}_lo"), getattr(r, f"diff_{b}_hi")
            rob = getattr(r, f"robust_{b}")
            if str(rob) == "True":
                m = "robust"
            elif lo > 0 or hi < 0:
                m = "CI≠0"
            else:
                m = "n.s."
            return f"{d:+.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}] {m}"
        k = (r.scope, r.terrain, r.difficulty, r.speed_bin, r.metric)
        rr = (f"{rep.loc[k, 'diff_Ours_rep']:+.{nd}f} [{rep.loc[k, 'diff_Ours_rep_lo']:.{nd}f}, {rep.loc[k, 'diff_Ours_rep_hi']:.{nd}f}]"
              if k in rep.index else "–")
        grp = "全部 40 格" if scope == "all" else f"v∈{r.speed_bin}" if scope == "speed_bin" else f"{r.terrain} {r.difficulty}"
        nf = f"{r.noise_p90:.{nd + 1}f}" if pd.notna(r.noise_p90) else "–"
        rows.append(f"| {grp} | {ZH_METRIC[r.metric]} | {r.Ours:.{nd}f} | {r.RGGP:.{nd}f} | {r.LPGP:.{nd}f} | {dc('RGGP')} "
                    f"| {dc('LPGP')} | {rr} | {nf} |")
    return "\n".join(rows)


def main() -> None:
    cells = load("cells.csv")
    rep_pooled = load("lens_overall_replicate_pooled.csv")
    print("## Table 0: all 40 cells pooled (cells.csv scope all; 10240 robots per policy)\n")
    print(overall_table(cells, rep_pooled))
    print("\n(robust = pre-specified rule; CI≠0 = CI excludes 0 but |diff| <= noise floor or no noise_floor.json metric; "
          "all-scope permutation p: 1e-4 (the floor of 10000 flips, cells.csv) for every metric; switch and completion "
          "1/(1e6+1), i.e. no flip as extreme among 1e6 (mc_stability_rows.csv); Holm-220 of these two = 0.00022. "
          "Window verdicts are % of the robots that traversed the window.)")
    print("\n## Table S: by commanded speed bin (cells.csv scope speed_bin)\n")
    print(overall_table(cells, rep_pooled, "speed_bin", ["switch_success_pct", "completion_pct", "cot", "step_rough_pct",
                                                          "roll_flat_out_pct", "mode_success_pct", "switch_success_dur_pct"]))
    fav = load("favourable_settings.csv")
    mc = load("mc_stability_rows.csv")
    sh = load("split_half_rows.csv")
    mc_cols = [c for c in mc.columns if c.startswith(("sw_robust_frac", "comp_robust_frac", "sw_holm220_hi", "comp_holm220_hi",
                                                        "sw_perm_p_hi", "comp_perm_p_hi"))]
    fav = fav.merge(mc[KEY + mc_cols], on=KEY, how="left", validate="1:1")
    fav = fav.merge(sh[KEY + ["sw_min_even", "sw_min_odd", "n_even", "n_odd"]], on=KEY, how="left", validate="1:1")
    assert fav.sw_holm220_hi_RGGP.notna().all() and fav.sw_min_even.notna().all()

    el = fav[fav.status == "eligible"].sort_values("rank")
    print("\n## Table A: top 10 eligible settings (pre-specified rule, all scopes)\n")
    print(fav_table(el.head(10)))
    print("\n## Table A2: sensitivity of the Table A rows (duration-only lift events, exit-flat rolling, held-out lanes, rerun)\n")
    print(sens_table(el.head(10)))
    print("\n(+ / − / · : 95 % CI above 0 / below 0 / contains 0. 仅时长 = lift event without the 3 cm clearance "
          "condition; 出口恢复滚动 = roll_flat_out_pct; even / odd lanes are disjoint halves of the 16 terrain instances "
          "of every cell; 重跑 Ours = the Ours rerun in place of Ours, baselines unchanged.)")

    print("\n## Table B: best eligible per coarser scope (rank_within_scope <= k)\n")
    k = {"all": 1, "difficulty": 2, "terrain": 2, "speed_bin": 2, "difficulty_speed": 2, "terrain_speed": 3, "cell": 6}
    sub = pd.concat([el[(el.scope == s) & (el.rank_within_scope <= n)] for s, n in k.items()])
    print(fav_table(sub))
    print("\n## Table B2: sensitivity of the Table B rows (same columns as Table A2)\n")
    print(sens_table(sub))
    st = fav[fav.eligible_strict.astype(str) == "True"].sort_values("rank_strict")
    js = json.loads((HERE / "rank_settings.json").read_text())
    print(f"\n(eligible {js['n_eligible']}; of these {js['n_eligible_with_ci_completion_loss']} have a completion loss whose "
          f"CI excludes 0 but that is below the cell-level noise floor, so 严格合格 = 否; strict set {js['n_eligible_strict']} rows. "
          f"The strict top 10 equals the Table A top 10: "
          f"{list(st.head(10)['rank'].astype(int)) == list(range(1, 11))}.)")

    unf = load("unfavourable_settings.csv")
    lab_b = {"w/o RG": "RGGP", "w/o LP": "LPGP"}
    kk = {"switch_success_pct": "sw", "completion_pct": "comp"}
    mci = mc.set_index(KEY)
    unf["holm_hi"] = [mci.loc[(r.scope, r.terrain, r.difficulty, r.speed_bin), f"{kk[r.metric]}_holm220_hi_{lab_b[r.vs]}"]
                      for r in unf.itertuples()]
    unf["robust_frac"] = [mci.loc[(r.scope, r.terrain, r.difficulty, r.speed_bin), f"{kk[r.metric]}_robust_frac_{lab_b[r.vs]}"]
                          for r in unf.itertuples()]
    sel = unf[(unf.holm_hi < 0.05) | (unf.holm220 < 0.05)].sort_values(["holm_hi", "diff"])
    print("\n## Table C: robust informative losses with Holm-220 < 0.05\n")
    print(loss_table(sel))
    print(f"\n(robust informative losses in total: {len(unf)}; with H (1e6) < 0.05: {int((unf.holm_hi < 0.05).sum())}; "
          f"with H (1e4, csv) < 0.05: {int((unf.holm220 < 0.05).sum())}; rows whose verdict differs: "
          f"{int(((unf.holm_hi < 0.05) != (unf.holm220 < 0.05)).sum())}.)")
    print("\n注：mode success（门控感知）的期望模式来自奖励粗糙度门控 λ > 0.65。Ours 和 w/o LP 的奖励使用这个门控，w/o RG 的奖励不用，"
          "所以对 w/o RG 的比较按构造偏向 Ours；mode success 还包含“完成”这一条件。它不是完成率损失的抵消证据，只作为次要指标列出。")

    mcj = json.loads((HERE / "mc_stability.json").read_text())
    print("\n## Table D: counts over the 220 candidates (informative rows)\n")
    print("| 指标 | 对比 | 有信息量 | robust 胜 (csv) | robust 负 (csv) | 胜：10 个种子范围 / 全部种子都 robust | 负：10 个种子范围 / 全部种子都 robust "
          "| H(1e6)<0.05 胜 / 负（csv robust 行） | 名义机会上限/方向 |")
    print("|---|---|---:|---:|---:|---|---|---|---:|")
    for key, c in js["counts_over_220"].items():
        m, b = key.split(" vs ")
        bb = lab_b[b]
        s = mcj["counts"][f"{m} vs {bb}"]
        print(f"| {m} | vs {b} | {c['informative']} | {c['robust_wins']} | {c['robust_losses']} "
              f"| {s['wins_range'][0]}–{s['wins_range'][1]} / {s['wins_all_seeds']} "
              f"| {s['losses_range'][0]}–{s['losses_range'][1]} / {s['losses_all_seeds']} "
              f"| {s['holm_lt05_hires_robustcsv'][0]} / {s['holm_lt05_hires_robustcsv'][1]} "
              f"| {c['nominal_chance_bound_per_direction']:.2f} |")
    print("\nSame-policy null (Ours vs Ours rerun, identical 220 groups; rows flagged robust = chance level):\n")
    print("| 指标 | csv：robust（Ours 好 / 差） | 10 个种子：Ours 好 范围 | Ours 差 范围 | 合计范围 |")
    print("|---|---|---|---|---|")
    for m, c in js["null_same_policy_over_220"].items():
        s = mcj["counts"][f"{m} vs null"]
        tot = [w + l for w, l in zip(s["wins_per_seed"], s["losses_per_seed"])]
        print(f"| {m} | {c['robust_any_direction']} ({c['robust_Ours_better']} / {c['robust_Ours_worse']}) "
              f"| {s['wins_range'][0]}–{s['wins_range'][1]} | {s['losses_range'][0]}–{s['losses_range'][1]} "
              f"| {min(tot)}–{max(tot)} |")
    print(f"\nstatus counts: {js['status_counts']}; eligible {js['n_eligible']}, "
          f"eligible with H<0.05 (1e4) on both switch comparisons {js['n_eligible_holm_both_lt05']}, "
          f"eligible whose replicate min switch diff > p90 {js['n_eligible_rep_sw_min_gt_p90']}")

    shj = json.loads((HERE / "split_half.json").read_text())
    print("\n## Table E: held-out-lane check of the ranking score (split_half.py)\n")
    print("| 候选集 | top k | 选择用 lane → 评估用 lane | 选择半的平均 min Δ switch | 留出半的平均 | 平均缩水 | 留出半最小值 | 与留出半 top k 重合 |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for pool, pname in (("rankable_informative", "全部可排名（switch 有信息量）"), ("eligible_full_data", "全数据合格的 70 行")):
        for kk_ in ("top5", "top10"):
            for dirn, v in shj[pool][kk_].items():
                a, b = dirn.replace("select_", "").split("_evaluate_")
                print(f"| {pname} ({shj[pool]['n_candidates']}) | {kk_[3:]} | {a} → {b} | {v['mean_sw_min_selection_half']:.1f} "
                      f"| {v['mean_sw_min_heldout_half']:.1f} | {v['mean_shrinkage']:.1f} | {v['min_heldout_sw_min']:.1f} "
                      f"| {v['overlap_with_heldout_topk']}/{kk_[3:]} |")


def splice(md: str, target: Path) -> int:
    """Replace every <!-- BEGIN TABLE X --> ... <!-- END TABLE X --> block of target with Table X of md
    (its body, without the '## Table X' heading). Returns the number of blocks replaced."""
    parts = re.split(r"^## Table (\S+?):[^\n]*\n", md, flags=re.M)
    body = {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}
    if not target.exists():
        print(f"splice target {target} does not exist; nothing spliced", file=sys.stderr)
        return 0
    text = target.read_text()
    n = 0
    for name, content in body.items():
        pat = re.compile(rf"(<!-- BEGIN TABLE {re.escape(name)} -->\n).*?(\n<!-- END TABLE {re.escape(name)} -->)", re.S)
        text, k = pat.subn(lambda m: m.group(1) + content + m.group(2), text)
        n += k
    target.write_text(text)
    return n


if __name__ == "__main__":
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    md = buf.getvalue()
    sys.stdout.write(md)
    if len(sys.argv) > 2 and sys.argv[1] == "--splice":
        n = splice(md, Path(sys.argv[2]))
        print(f"spliced {n} table blocks into {sys.argv[2]}", file=sys.stderr)
