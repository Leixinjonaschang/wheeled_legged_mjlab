# Ablation evaluation scripts

Evaluation of the depth ablation checkpoints (`logs/exp_ckpt/model_29999_{Ours,RGGP,LPGP}.pt`).
Run everything from the repo root; results are written under `logs/lipm_eval/` (not tracked).
Exact commands, protocols and metric definitions are in each script's docstring.

```
scripts/eval/
  lipm_eval.py            multi-terrain evaluation (audit / rollout / aggregate / run) -> logs/lipm_eval/full
  mode_switch_eval.py     flat -> rough -> flat course (rollout / aggregate)
  lipm_diagnostics/       follow-up diagnostics of logs/lipm_eval/full -> full/diagnostics, full/results
  mode_switch/            mode_switch_v2 batch -> logs/lipm_eval/mode_switch_v2
    run_all.sh            30 rollouts + Ours replicate, aggregate, per-env checks, noise floor
    tools/                terrain audit, per-env checks, noise floor, legacy regression, box density
    analysis/             statistics and report tables (data in mode_switch_v2/analysis)
```

Order:

1. `lipm_eval.py run --out logs/lipm_eval/full`
2. `lipm_diagnostics/` (simulation scripts first, then their reports):
   - `behavior_record.py` -> `behavior_report.py`
   - `energy_contact.py` -> `energy_contact_report.py`
   - `latent_collect.py` -> `latent_analysis.py` -> `latent_figure.py`
   - `benefit_metrics.py`, `fall_only_comparison.py`, then `ours_vs_lpgp_scorecard.py`
   - `unified_table.py` and `clearance_sensitivity.py` also read the first course experiment
     `logs/lipm_eval/mode_switch` (forward commands, `mode_switch_eval.py rollout --rough
     {tilted_grid,random_spread,discrete_obstacles}`, then `aggregate`)
   - `illegal_contact_geoms.py`, `reset_variant_check.py`: one-off protocol checks
3. `mode_switch/tools/terrain_audit.py`, then `mode_switch/run_all.sh`
4. `mode_switch/analysis/`: `tidy.py` -> `lens_*.py` -> `rank_settings.py` -> `mc_stability.py`,
   `split_half.py` -> `report_tables.py`, `report_checks.py`; `fall_only_completion.py` reads the rollouts.

Rollouts are not bit-reproducible (nondeterministic GPU physics); compare against the noise floor
(`mode_switch/tools/noise_floor.py`), not point estimates.
