# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MJLab (MuJoCo Warp) RL environment for the WF-TRON1B wheeled-legged robot, trained with a
repository-bundled, customized fork of RSL-RL (`rsl_rl/`, installed editable). The research focus is
teacher–student velocity-tracking policies with a learned latent representation, optional depth
perception, latent-dynamics prediction, and roughness-conditioned rewards.

Pinned stack (enforced by `tests/test_dependency_versions.py`): Python 3.13, mjlab 1.6.0,
MuJoCo / mujoco-warp 3.11.0, warp-lang 1.14.0, rsl-rl-lib 5.3.0 (from `rsl_rl/`).

## Commands

```shell
uv sync --locked                      # install; do NOT use `pip install .` (skips the uv overrides)

uv run python -m pytest tests         # project tests (`python -m` puts the repo root on sys.path for `scripts.*` imports; GPU-only tests skip without CUDA)
uv run python -m pytest rsl_rl/tests  # bundled RSL-RL tests
uv run python -m pytest tests/test_velocity_command.py::<test_name>   # single test

uv run ruff check <path>              # rsl_rl/ has its own ruff.toml (line length 120)

# Train / play (task configs are tyro dataclasses: override with --env.x.y / --agent.x.y)
uv run python scripts/rsl_rl/train.py <TASK> --help
uv run python scripts/rsl_rl/train.py <TASK> --gpu-ids [0] --env.scene.num-envs 512 --agent.max-iterations 1000
uv run python scripts/rsl_rl/train.py <TASK> --gpu-ids [0,1]      # multi-GPU via torchrunx
uv run python scripts/rsl_rl/play.py <TASK> --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_N.pt
uv run python scripts/rsl_rl/play.py <TASK> --agent zero --num-envs 1 --no-terminations True

# Ablation batch: LPGP, RGGP, OursGP × seeds 42/44/46, then BlindGP × 3 seeds; GPU pairs (0,1),(2,3)
uv run python scripts/rsl_rl/run_ablation.py --dry-run
```

Outputs go to `logs/rsl_rl/<experiment_name>/` (checkpoints + ONNX) and `logs/ablation/<timestamp>/`
(launcher logs, `status.json`). Default logger is W&B. See README.md for the full CLI option list.

## Registered tasks

Only the tasks in `src/wheeled_legged_mjlab/__init__.py` exist (the package is discovered through the
`mjlab.tasks` entry point in `pyproject.toml`):

- `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-BlindGP` — blind, proprio-only representation policy
- `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-LPGP` — depth, no latent predictor
- `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-OursGP` — depth + latent-dynamics predictor (full method)
- `Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-RGGP` — OursGP with roughness-conditioned rewards disabled

The README and `tests/test_mjlab160_integration.py` still reference `Mjlab-Velocity-Flat-WF-Tron1B` /
`Mjlab-Velocity-Rough-WF-Tron1B`, which are no longer registered (their cfg factories still exist).
`run_ablation.py` hard-codes these task IDs (`TASKS`, `TAIL_TASKS`), so renaming tasks breaks it.

## Architecture

A task = env cfg + RL runner cfg + runner class, wired together by `register_mjlab_task`.

**Env config** — `src/wheeled_legged_mjlab/tasks/velocity/config/wf_tron1b/env_cfgs.py`. A single
`make_env_cfg(rough, play, depth, lin_vel_representation, async_depth, roughness_conditioned_rewards)`
composes `make_scene/observations/actions/commands/events/rewards/terminations/curriculum/...`;
the public `wf_tron1b_*_env_cfg` functions are thin flag combinations. `play=True` disables action delay
and applies `apply_play_overrides` (e.g. turns off observation corruption). Terrain is in
`terrain_cfg.py`; the robot asset in `assets/WF_TRON1B/`. MDP term implementations
(rewards, observations, custom actions with delay, commands, curriculums, events, terminations)
live in `tasks/velocity/mdp/`.

**Observation groups are the contract between env and algorithm.** With `lin_vel_representation`
the env emits `proprio_history` (noisy student input), `actor_command`, `lin_vel_target`,
`critic`, `privileged_encoder`, `wheel_roughness`, `dynamics_context`, and (with depth) `depth_camera`.
The runner cfg's `obs_groups` dict maps the algorithm's named inputs to one or more env groups
(e.g. `"critic": ("critic", "dynamics_context")`). Changing observation groups requires updating
`obs_groups` in `rl_cfg.py`, the model's expected inputs, and the ONNX metadata in `rl/runner.py`.

**RL config** — `config/wf_tron1b/rl_cfg.py`. Custom cfg dataclasses subclass mjlab's
`RslRlModelCfg` / `RslRlPpoAlgorithmCfg` and set `class_name`, which `rsl_rl` resolves via
`resolve_callable`; it is either a name exported from `rsl_rl.models`/`rsl_rl.algorithms` or a full
`"module.path:ClassName"` string (used for the predictor classes, which are not exported).
Variants are built by calling a base factory and mutating fields (e.g. the RGGP runner cfg reuses
OursGP and only changes `experiment_name`).

**Bundled RSL-RL fork** (`rsl_rl/rsl_rl/`) — the custom pieces are in `algorithms/representation_*`
(teacher–student PPO: teacher uses privileged encoder, student learns to match the latent and
predict linear velocity from proprio history; the `predictor` variant adds multi-horizon latent/velocity
dynamics prediction and latent rollout losses with a separate optimizer) and matching
`models/*representation*` actor-critics (the depth models add a CNN+GRU depth encoder with a recurrent
hidden state). Edits here change training behavior directly; there is no upstream package.

**Runtime glue** — `src/wheeled_legged_mjlab/rl/`:
- `WheeledLeggedRslRlVecEnvWrapper` adds `extras["applied_actions"]` (the delayed/actually applied
  action, used by the dynamics predictor); `scripts/rsl_rl/train.py` uses it instead of mjlab's wrapper.
- `WheeledLeggedVelocityOnPolicyRunner` exports ONNX on every save and attaches deployment metadata
  (mixed position/velocity leg+wheel actions, per-policy-type input/output names). ONNX export
  failures are logged as warnings and do not stop training.
- `tasks/velocity/env.py` adds play-time debug visualizers (predicted lin vel, roughness gate).

## Conventions and gotchas

- `tests/` and `rsl_rl/tests/` are gitignored except for an explicit allowlist in `.gitignore`;
  a new test file must be added there with a `!path` entry or it will not be committed.
- Indentation differs by file (mjlab-derived scripts/`env.py` use 2 spaces, most other code 4);
  match the file you are editing.
- Resuming: loading a checkpoint's state dict does not imply strict-resume compatibility across
  config changes (see README).
- `docs/` holds design notes and plans (depth sim2real DR, roughness-conditioned rewards, training-input
  migration, NaN debugging) — check them before changing those subsystems.
