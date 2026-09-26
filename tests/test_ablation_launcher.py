"""Check experiment ordering and GPU isolation without starting training."""

import json
import subprocess
from argparse import Namespace

import pytest

from scripts.rsl_rl import run_ablation as launcher


def options(tmp_path, max_concurrent=2):
    return Namespace(
        output_dir=tmp_path / "batch",
        num_envs_per_gpu=2048,
        max_iterations=30_000,
        logger="tensorboard",
        max_concurrent=max_concurrent,
    )


@pytest.mark.parametrize("max_concurrent", [1, 2])
def test_seed_barrier_and_exclusive_gpu_pairs(tmp_path, monkeypatch, max_concurrent):
    active = {}
    launched = []
    tick = [0]

    class Process:
        def __init__(self, cmd, *, cwd, env, stdout, stderr, start_new_session):
            self.name = cmd[cmd.index("--agent.run-name") + 1]
            self.seed = int(cmd[cmd.index("--agent.seed") + 1])
            assert self.seed == int(self.name.rsplit("_seed", 1)[1])
            assert self.seed in (42, 44, 46)
            self.gpus = env["CUDA_VISIBLE_DEVICES"]
            assert self.gpus not in active
            assert all(process.seed == self.seed for process in active.values())
            assert len(active) < max_concurrent
            assert cmd[cmd.index("--env.scene.num-envs") + 1] == "2048"
            assert cmd[cmd.index("--gpu-ids") + 1] == "[0,1]"
            assert start_new_session
            assert self.name in env["TORCHRUNX_LOG_DIR"]
            self.finish = tick[0] + (3 if self.name.startswith("LPGP") else 1)
            self.pid = 100 + len(launched)
            active[self.gpus] = self
            launched.append(self.name)

        def poll(self):
            if tick[0] < self.finish:
                return None
            del active[self.gpus]
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(
        launcher.time, "sleep", lambda _: tick.__setitem__(0, tick[0] + 1)
    )
    args = options(tmp_path, max_concurrent)
    assert launcher.run(args, ["0,1", "2,3"]) == 0
    assert launched == [
        f"{name}_seed{seed}"
        for seed in (42, 44, 46)
        for name in ("LPGP", "RGGP", "OursGP")
    ]
    assert not active
    records = json.loads((args.output_dir / "status.json").read_text())
    assert all(record["status"] == "completed" for record in records)
    expected_seeds = {0: [42, 43], 1: [44, 45], 2: [46, 47]}
    for record in records:
        assert record["worker_seeds"] == expected_seeds[record["seed"]]
        assert record["base_seed"] == record["worker_seeds"][0]
        assert record["run_name"].endswith(f"_seed{record['base_seed']}")
        cmd = record["command"]
        assert int(cmd[cmd.index("--agent.seed") + 1]) == record["base_seed"]


def test_failure_stops_queue_and_cleans_up_active_jobs(tmp_path, monkeypatch):
    launched = []
    stopped = []

    class Process:
        def __init__(self, cmd, **kwargs):
            self.name = cmd[cmd.index("--agent.run-name") + 1]
            self.pid = 100 + len(launched)
            launched.append(self.name)

        def poll(self):
            return 7 if self.name.startswith("RGGP") else None

    def stop(running):
        for process, stream, _ in running.values():
            stopped.append(process.name)
            stream.close()

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    monkeypatch.setattr(launcher, "stop_processes", stop)
    args = options(tmp_path)
    assert launcher.run(args, ["0,1", "2,3"]) == 1
    assert launched == stopped == ["LPGP_seed42", "RGGP_seed42"]
    records = json.loads((args.output_dir / "status.json").read_text())
    assert [record["status"] for record in records] == ["interrupted", "failed"] + [
        "pending"
    ] * 7


def test_gpu_mapping_respects_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6,1,3")
    assert launcher.visible_groups(launcher.gpu_groups(["0,1", "2,3"])) == [
        "4,6",
        "1,3",
    ]
    with pytest.raises(ValueError, match="overlap"):
        launcher.gpu_groups(["0,1", "1,2"])
    with pytest.raises(ValueError, match="exactly two"):
        launcher.gpu_groups(["0"])
    with pytest.raises(ValueError, match="outside"):
        launcher.visible_groups([(0, 4)])


def test_dry_run_does_not_launch_or_create_output(tmp_path):
    output = tmp_path / "unused"
    result = subprocess.run(
        [
            launcher.sys.executable,
            str(launcher.Path(launcher.__file__)),
            "--dry-run",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={
            k: v for k, v in launcher.os.environ.items() if k != "CUDA_VISIBLE_DEVICES"
        },
    )
    assert result.stdout.count("--agent.run-name") == 9
    for trial_seed, base_seed in ((0, 42), (1, 44), (2, 46)):
        assert (
            f"Trial {trial_seed}; worker seeds ({base_seed}, {base_seed + 1})"
            in result.stdout
        )
        for name in ("LPGP", "RGGP", "OursGP"):
            assert (
                f"--agent.seed {base_seed} --agent.run-name {name}_seed{base_seed}"
                in result.stdout
            )
    assert not output.exists()
