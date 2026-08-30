#!/usr/bin/env python3
"""30-minute GPU-aware supervisor for the exploratory EfficientAD screen.

The supervisor never sends signals to an existing process. A GPU is considered startable only
when nvidia-smi reports no compute process, <=5% utilization, and >=4 GiB free memory. Each
started child is tagged in the state file and limited to one CPU thread and 30% GPU memory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(
    os.environ.get(
        "EVOINSPECT_EFFICIENTAD_PYTHON",
        "/home/CuiMinghao/envs/evoinspect-efficientad/bin/python",
    )
)
BATCH = REPO / "reports/experiments/parallel-screening-20260831"
SOURCE_BATCH = REPO / "reports/experiments/efficientad-s-frozen-20260830T004009Z-seed143-gpu0-3"
VARIANTS = {
    "efficientad_s_384": ("global_single_forward", 384),
    "efficientad_s_512": ("global_single_forward", 512),
    "static_tile_efficientad_s": ("static_tile_efficientad_s", 256),
}
CATEGORIES = ("cable", "capsule", "screw", "carpet", "transistor", "wood")


def now() -> str:
    return datetime.now(UTC).isoformat()


def gpu_snapshot() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        return []
    result: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpu = int(parts[0])
            used = int(parts[1])
            free = int(parts[2])
            util = int(parts[3])
        except ValueError:
            continue
        process_query = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [
            int(value.strip())
            for value in process_query.stdout.splitlines()
            if value.strip().isdigit()
        ]
        result.append(
            {
                "gpu": gpu,
                "memory_used_mb": used,
                "memory_free_mb": free,
                "utilization_gpu_percent": util,
                "compute_pids": pids,
                "startable": not pids and free >= 4096 and util <= 5,
            }
        )
    return result


def source_run(category: str) -> Path:
    matches = sorted(SOURCE_BATCH.glob(f"runs/efficientad-s-{category}-s143-*/"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source run for {category}, found {matches}")
    return matches[0]


@dataclass
class Task:
    task_id: str
    variant: str
    category: str
    output_dir: Path
    process: subprocess.Popen[bytes] | None = None
    gpu: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    returncode: int | None = None
    status: str = "queued"


def task_list() -> list[Task]:
    tasks: list[Task] = []
    for variant in VARIANTS:
        for category in CATEGORIES:
            task_id = f"{variant}-{category}-s143"
            tasks.append(
                Task(
                    task_id=task_id,
                    variant=variant,
                    category=category,
                    output_dir=BATCH / "runs" / task_id,
                )
            )
    return tasks


def json_task(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "variant": task.variant,
        "category": task.category,
        "status": task.status,
        "gpu": task.gpu,
        "pid": task.process.pid if task.process else None,
        "started_at": task.started_at,
        "ended_at": task.ended_at,
        "returncode": task.returncode,
        "output_dir": str(task.output_dir),
    }


def write_snapshot(tasks: list[Task], snapshots: Path, reason: str) -> None:
    payload = {
        "timestamp": now(),
        "reason": reason,
        "gpus": gpu_snapshot(),
        "tasks": [json_task(task) for task in tasks],
        "queue": {
            "queued": sum(task.status == "queued" for task in tasks),
            "running": sum(task.status == "running" for task in tasks),
            "completed": sum(task.status == "completed" for task in tasks),
            "failed": sum(task.status == "failed" for task in tasks),
        },
    }
    snapshots.parent.mkdir(parents=True, exist_ok=True)
    with snapshots.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    (BATCH / "state.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def start_task(task: Task, gpu: int) -> None:
    inference, resolution = VARIANTS[task.variant]
    source = source_run(task.category)
    if task.output_dir.exists():
        if (task.output_dir / "metrics.json").is_file():
            raise FileExistsError(f"completed output already exists: {task.output_dir}")
        # A prior launch failure may have left only a log/partial directory. Keep it
        # recoverable and give the evaluator its required fresh output directory.
        backup = task.output_dir.with_name(
            f"{task.output_dir.name}.stale-{int(time.time())}"
        )
        shutil.move(str(task.output_dir), str(backup))
    task.output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = task.output_dir.parent / f"{task.output_dir.name}.launcher.log"
    log = log_path.open("wb")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "EVOINSPECT_PHYSICAL_GPU": str(gpu),
            "EVOINSPECT_GPU_MEMORY_FRACTION": "0.30",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": f"{REPO / 'src'}:{REPO}",
        }
    )
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "2h",
        str(PYTHON),
        str(REPO / "scripts/evaluate_efficientad_screen_variant.py"),
        "--source-run",
        str(source),
        "--output-dir",
        str(task.output_dir),
        "--inference",
        inference,
        "--resolution",
        str(resolution),
        "--device",
        "cuda:0",
        "--run-id",
        f"parallel-screening-{task.task_id}",
    ]
    task.process = subprocess.Popen(
        command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    task.gpu = gpu
    task.started_at = now()
    task.status = "running"


def poll_tasks(tasks: list[Task]) -> None:
    for task in tasks:
        if task.status != "running" or task.process is None:
            continue
        returncode = task.process.poll()
        if returncode is None:
            continue
        task.returncode = int(returncode)
        task.ended_at = now()
        task.status = "completed" if returncode == 0 else "failed"


def launch(tasks: list[Task]) -> int:
    snapshots = BATCH / "monitor_snapshots.jsonl"
    launched = 0
    running_by_gpu: dict[int, int] = {}
    for task in tasks:
        if task.status == "running" and task.gpu is not None:
            running_by_gpu[task.gpu] = running_by_gpu.get(task.gpu, 0) + 1
    for record in gpu_snapshot():
        gpu = int(record["gpu"])
        if not bool(record["startable"]):
            continue
        slots = 2 - running_by_gpu.get(gpu, 0)
        for _ in range(max(0, slots)):
            pending = next((task for task in tasks if task.status == "queued"), None)
            if pending is None:
                break
            try:
                start_task(pending, gpu)
            except Exception as exc:  # keep the supervisor alive and preserve the error
                pending.status = "failed"
                pending.ended_at = now()
                pending.returncode = 99
                pending.output_dir.mkdir(parents=True, exist_ok=True)
                (pending.output_dir / "launch_error.txt").write_text(
                    str(exc) + "\n", encoding="utf-8"
                )
            else:
                launched += 1
                running_by_gpu[gpu] = running_by_gpu.get(gpu, 0) + 1
    write_snapshot(tasks, snapshots, f"launch_{launched}")
    return launched


def aggregate_if_done(tasks: list[Task]) -> None:
    if any(task.status in {"queued", "running"} for task in tasks):
        return
    output = BATCH / "screening-summary.json"
    subprocess.run(
        [
            str(PYTHON),
            str(REPO / "scripts/aggregate_parallel_screening.py"),
            "--batch-root",
            str(BATCH),
            "--output",
            str(output),
        ],
        cwd=REPO,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="queue polling interval in seconds; snapshots are written at each poll",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise SystemExit(f"EfficientAD Python not found: {PYTHON}")
    BATCH.mkdir(parents=True, exist_ok=True)
    tasks = task_list()
    # Resume only from durable metrics; failed directories are never overwritten.
    for task in tasks:
        metrics = task.output_dir / "metrics.json"
        if metrics.is_file():
            task.status = "completed"
    stop = False

    def handle_stop(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True
        for task in tasks:
            if task.status == "running" and task.process is not None:
                try:
                    os.killpg(task.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    while not stop:
        poll_tasks(tasks)
        launch(tasks)
        aggregate_if_done(tasks)
        if args.once or all(task.status in {"completed", "failed"} for task in tasks):
            break
        for _ in range(args.interval):
            if stop:
                break
            time.sleep(1)
    poll_tasks(tasks)
    write_snapshot(tasks, BATCH / "monitor_snapshots.jsonl", "exit")
    aggregate_if_done(tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
