# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/wheeled_legged_mjlab/`. Robot models and meshes are under `assets/WF_TRON1B/`; sensors, RL wrappers, and velocity-task logic are separated into `sensors/`, `rl/`, and `tasks/velocity/`. Task-specific environment, terrain, and agent settings live in `tasks/velocity/config/wf_tron1b/`, while reusable MDP terms live in `tasks/velocity/mdp/`. Use `scripts/rsl_rl/train.py` and `play.py` as operational entry points. Root tests are in `tests/`; the editable, repository-bundled RSL-RL fork has independent code and tests under `rsl_rl/`. Design notes and experiment audits belong in `docs/`.

## Build, Test, and Development Commands

- `uv sync --locked` installs the validated Python 3.13/MJLab stack and editable bundled RSL-RL. Do not replace this with `pip install .`.
- `uv run pytest tests` runs project regression and integration tests.
- `uv run pytest rsl_rl/tests` validates changes to the bundled learner.
- `uv run ruff check src scripts tests` checks imports and Python lint issues; use `uv run ruff format --check src scripts tests` to verify formatting.
- `uv run python scripts/rsl_rl/train.py Mjlab-Velocity-Flat-WF-Tron1B` starts training. Substitute the `Rough` task when appropriate.
- `uv run python scripts/rsl_rl/play.py Mjlab-Velocity-Flat-WF-Tron1B --checkpoint-file <path>` evaluates a checkpoint.

## Coding Style & Naming Conventions

Use type annotations and concise Google-style docstrings. Follow the indentation already established in the file: task modules and scripts commonly use two-space blocks, while tests and bundled RSL-RL generally use four. Avoid unrelated reindentation. Name functions and modules `snake_case`, classes/config dataclasses `PascalCase`, and constants `UPPER_SNAKE_CASE`. Keep configuration construction in config modules and reusable reward, observation, action, event, curriculum, and termination logic in the matching `mdp/` module.

## Testing Guidelines

Pytest discovers `test_*.py` files and `test_*` functions. Add focused regression tests beside related root tests; changes inside `rsl_rl/` need tests in its own suite. Prefer deterministic CPU-sized fixtures, and clearly isolate GPU/MJLab integration assumptions. No coverage threshold is enforced, but new behavior and bug fixes should exercise both configuration and runtime behavior where practical.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Split leg and wheel motion penalties.`; scoped Conventional Commit forms such as `feat(depth): ...` and `chore(rewards): ...` are also accepted. Keep each commit focused. PRs should explain affected tasks/configs, list exact validation commands, and link relevant issues. Include videos or screenshots for viewer/visualization changes and training metrics plus checkpoint-compatibility notes for reward, observation, or policy changes.
