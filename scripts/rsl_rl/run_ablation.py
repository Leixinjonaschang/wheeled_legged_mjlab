"""Run the three depth ablations, seed by seed, on disjoint two-GPU groups.

Preview: uv run python scripts/rsl_rl/run_ablation.py --dry-run
Launch:  uv run python scripts/rsl_rl/run_ablation.py
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth"
TASKS = (
    ("LPGP", PREFIX + "-LPGP"),
    ("RGGP", PREFIX + "-Predict-RGGP"),
    ("OursGP", PREFIX + "-Predict-OursGP"),
)
SEEDS = (0, 1, 2)  # Trial indices used in scheduling.


def worker_seeds(seed: int) -> tuple[int, int]:
    """Allocate disjoint worker seeds to each two-GPU experiment repeat."""
    return 42 + 2 * seed, 43 + 2 * seed


def gpu_groups(values: list[str]) -> list[tuple[int, int]]:
    groups = [tuple(int(value) for value in group.split(",")) for group in values]
    if any(len(group) != 2 for group in groups):
        raise ValueError("Each GPU group must contain exactly two indices, e.g. 0,1.")
    indices = [index for group in groups for index in group]
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError(
            "GPU indices must be non-negative and groups must not overlap."
        )
    return groups


def visible_groups(groups: list[tuple[int, int]]) -> list[str]:
    """Interpret indices relative to the parent's CUDA_VISIBLE_DEVICES, if set."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [",".join(map(str, group)) for group in groups]
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if any(index >= len(devices) for group in groups for index in group):
        raise ValueError(
            "GPU group index is outside the inherited CUDA_VISIBLE_DEVICES."
        )
    selected = [[devices[index] for index in group] for group in groups]
    flat = [device for group in selected for device in group]
    if len(set(flat)) != len(flat):
        raise ValueError("CUDA_VISIBLE_DEVICES maps the groups to overlapping devices.")
    return [",".join(group) for group in selected]


def command(task: str, name: str, seed: int, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/rsl_rl/train.py"),
        task,
        "--gpu-ids",
        "[0,1]",
        "--env.scene.num-envs",
        str(args.num_envs_per_gpu),
        "--agent.seed",
        str(worker_seeds(seed)[0]),
        "--agent.run-name",
        f"{name}_seed{worker_seeds(seed)[0]}",
        "--agent.max-iterations",
        str(args.max_iterations),
        "--agent.logger",
        args.logger,
        "--agent.resume",
        "False",
    ]


def stop_processes(running: dict) -> None:
    """Stop each training process group, including its distributed workers."""
    for process, _, _ in running.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for process, _, _ in running.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process, stream, _ in running.values():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        stream.close()


def run(args: argparse.Namespace, groups: list[str]) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = [
        {
            "run_name": f"{name}_seed{worker_seeds(seed)[0]}",
            "seed": seed,
            "base_seed": worker_seeds(seed)[0],
            "worker_seeds": worker_seeds(seed),
            "task": task,
            "command": command(task, name, seed, args),
            "status": "pending",
        }
        for seed in SEEDS
        for name, task in TASKS
    ]

    def save_status():
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(output / "status.json")

    running = {}
    save_status()
    print(f"Launcher logs and status: {output}", flush=True)
    try:
        for seed in SEEDS:
            print(f"=== Trial {seed}; worker seeds {worker_seeds(seed)} ===", flush=True)
            pending = deque(record for record in records if record["seed"] == seed)
            while pending or running:
                # Check every active job before assigning any newly freed slot.
                for slot, (process, stream, record) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    record.update(
                        exit_code=code, finished_at=datetime.now(UTC).isoformat()
                    )
                    record["status"] = "completed" if code == 0 else "failed"
                    save_status()
                    if code != 0:
                        print(
                            f"FAILED {record['run_name']}: see {record['stdout']}",
                            flush=True,
                        )
                        return 1
                    stream.close()
                    del running[slot]
                    print(f"DONE {record['run_name']}", flush=True)

                for slot, devices in enumerate(groups):
                    if not pending or len(running) >= args.max_concurrent:
                        break
                    if slot in running:
                        continue
                    record = pending.popleft()
                    stdout = output / f"{record['run_name']}.log"
                    child_env = os.environ.copy()
                    child_env["CUDA_VISIBLE_DEVICES"] = devices
                    child_env["PYTHONUNBUFFERED"] = "1"
                    # Keep torchrunx output local to this experiment even if the
                    # parent shell defines a shared logging directory.
                    child_env["TORCHRUNX_LOG_DIR"] = str(
                        output / f"{record['run_name']}_workers"
                    )
                    record.update(
                        gpus=devices,
                        stdout=str(stdout),
                        started_at=datetime.now(UTC).isoformat(),
                        status="starting",
                    )
                    save_status()
                    stream = stdout.open("w")
                    stream.write(
                        f"CUDA_VISIBLE_DEVICES={devices} {shlex.join(record['command'])}\n"
                    )
                    stream.flush()
                    try:
                        process = subprocess.Popen(
                            record["command"],
                            cwd=ROOT,
                            env=child_env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except OSError:
                        stream.close()
                        record["status"] = "failed"
                        save_status()
                        raise
                    running[slot] = (process, stream, record)
                    record.update(pid=process.pid, status="running")
                    save_status()
                    print(
                        f"START {record['run_name']} GPUs={devices} log={stdout}",
                        flush=True,
                    )
                if running:
                    time.sleep(1)
            # This barrier prevents seed N+1 from starting before all of seed N.
        return 0
    finally:
        stop_processes(running)
        for _, _, record in running.values():
            if record["status"] == "running":
                record.update(
                    status="interrupted", finished_at=datetime.now(UTC).isoformat()
                )
        save_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-groups",
        nargs="+",
        default=["0,1", "2,3"],
        help="Disjoint GPU pairs, relative to CUDA_VISIBLE_DEVICES if set.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        choices=(1, 2),
        default=2,
        help="Use 1 for strict sequential execution; default: 2.",
    )
    parser.add_argument("--num-envs-per-gpu", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=30_000)
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "logs/ablation"
        / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_%f"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all nine jobs without launching or writing files.",
    )
    args = parser.parse_args()
    try:
        groups = gpu_groups(args.gpu_groups)
        devices = visible_groups(groups)
    except ValueError as error:
        parser.error(str(error))
    if args.num_envs_per_gpu <= 0 or args.max_iterations <= 0:
        parser.error("Environment count and iteration count must be positive.")
    if args.max_concurrent > len(groups):
        parser.error("--max-concurrent cannot exceed the number of GPU groups.")
    print(f"GPU groups: {devices}; concurrent jobs: {args.max_concurrent}", flush=True)
    print(
        f"Each job: 2 GPUs x {args.num_envs_per_gpu} envs; {args.max_iterations} iterations",
        flush=True,
    )
    if args.dry_run:
        for seed in SEEDS:
            print(
                f"Trial {seed}; worker seeds {worker_seeds(seed)} "
                "(wait for all three jobs before the next seed):"
            )
            for name, task in TASKS:
                print(
                    f"  CUDA_VISIBLE_DEVICES=<free GPU pair> {shlex.join(command(task, name, seed, args))}"
                )
        return 0

    # Preflight only when launching, so dry-run works without CUDA or mjlab.
    import mjlab.tasks  # noqa: F401
    import torch
    from mjlab.tasks.registry import list_tasks

    count = torch.cuda.device_count()
    if any(index >= count for group in groups for index in group):
        parser.error(
            f"Requested GPU index is unavailable; only {count} CUDA devices are visible."
        )
    missing = [task for _, task in TASKS if task not in list_tasks()]
    if missing:
        parser.error(f"Tasks are not registered: {missing}")

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        return run(args, devices)
    except KeyboardInterrupt:
        print("Interrupted; active training process groups stopped.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
