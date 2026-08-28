from __future__ import annotations

import importlib.metadata
from pathlib import Path

import rsl_rl.algorithms


def test_migration_dependency_versions_and_bundled_rsl_rl_source() -> None:
    assert importlib.metadata.version("mjlab") == "1.6.0"
    assert importlib.metadata.version("mujoco") == "3.11.0"
    assert importlib.metadata.version("mujoco-warp") == "3.11.0"
    assert importlib.metadata.version("warp-lang") == "1.14.0"
    assert importlib.metadata.version("rsl-rl-lib") == "5.3.0"

    repository_root = Path(__file__).resolve().parents[1]
    algorithms_file = Path(rsl_rl.algorithms.__file__).resolve()
    assert algorithms_file.is_relative_to(repository_root / "rsl_rl" / "rsl_rl")
