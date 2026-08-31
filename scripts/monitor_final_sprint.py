#!/usr/bin/env python3
"""Conservatively schedule the two remaining bounded visual screens.

The supervisor only launches on a GPU with no entries in nvidia-smi's compute-app
table, at least 8 GiB free, and <=5% utilization. It never terminates a process.
Polling is frequent enough to catch a released GPU while preserving a half-hour
audit record in the log.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

POLL_SECONDS = 30
MIN_FREE_MIB = 8 * 1024
MAX_UTILIZATION = 5


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Task:
    method: str
    category: str
    config: Path
    script: Path
    output_root: Path

    @property
    def key(self) -> str:
        return f"{self.method}-{self.category}"


@dataclass
class Running:
    task: Task
    gpu: int
    process: subprocess.Popen[str]
    log_path: Path
    started_at: str


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(row[0].strip()),
                    "uuid": row[1].strip(),
                    "utilization_gpu": int(row[2].strip()),
                    "memory_free_mib": int(row[3].strip()),
                    "memory_used_mib": int(row[4].strip()),
                }
            )
        except ValueError:
            continue
    return rows


def query_compute_uuids() -> set[str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"DRIVER_UNAVAILABLE"}
    uuids: set[str] = set()
    for row in csv.reader(completed.stdout.splitlines()):
        if row and row[0].strip():
            uuids.add(row[0].strip())
    return uuids


def safe_gpus() -> tuple[list[int], list[dict[str, Any]]]:
    gpus = query_gpus()
    compute_uuids = query_compute_uuids()
    safe: list[int] = []
    for gpu in gpus:
        gpu["compute_process_present"] = gpu["uuid"] in compute_uuids
        gpu["startable"] = (
            not gpu["compute_process_present"]
            and gpu["memory_free_mib"] >= MIN_FREE_MIB
            and gpu["utilization_gpu"] <= MAX_UTILIZATION
        )
        if gpu["startable"]:
            safe.append(int(gpu["index"]))
    return safe, gpus


def write_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_tasks(repo: Path, output_root: Path) -> list[Task]:
    categories = ["cable", "capsule", "screw", "carpet", "transistor", "wood"]
    return [
        *[
            Task(
                method="anomalydino",
                category=category,
                config=repo / "configs/experiments/anomalydino_screen_20260901.yaml",
                script=repo / "scripts/evaluate_anomalydino_screen.py",
                output_root=output_root / "anomalydino",
            )
            for category in categories
        ],
        *[
            Task(
                method="dinomaly",
                category=category,
                config=repo / "configs/experiments/dinomaly_screen_20260901.yaml",
                script=repo / "scripts/evaluate_dinomaly_screen.py",
                output_root=output_root / "dinomaly",
            )
            for category in categories
        ],
    ]


def task_completed(task: Task) -> bool:
    return (task.output_root / task.category / "metrics.json").is_file()


def launch(task: Task, gpu: int, repo: Path, log_dir: Path) -> Running:
    task.output_root.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "EVOINSPECT_PHYSICAL_GPU": f"RTX3090:{gpu}",
            "PYTHONPATH": f"{repo / 'src'}:{repo}:{repo / 'third_party/anomalib-2.3.0/src'}",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "4"),
        }
    )
    python_path = repo.parents[2] / "envs/evoinspect-efficientad/bin/python"
    command = [
        str(python_path),
        str(task.script),
        "--config",
        str(task.config),
        "--category",
        task.category,
        "--device",
        "cuda:0",
        "--output-root",
        str(task.output_root),
    ]
    stream.write(f"[{utc_now()}] launch gpu={gpu} command={command}\n")
    stream.flush()
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return Running(task, gpu, process, log_path, utc_now())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/experiments/final-sprint-20260901"),
    )
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = (
        (repo / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root
    )
    log_path = args.log or (output_root / "monitor.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_root / "monitor_snapshot.json"
    tasks = build_tasks(repo, output_root)
    queue = [task for task in tasks if not task_completed(task)]
    running: dict[str, Running] = {}
    stopping = False

    def stop_handler(signum: int, _frame: Any) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"[{utc_now()}] received signal {signum}; no child process terminated\n")
        # Leave any already launched child running; stopping the supervisor must never
        # imply terminating a training process.  Exiting immediately also makes the
        # monitor safe to control from a service manager or a terminal.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    while queue or running:
        finished: list[str] = []
        for key, item in running.items():
            code = item.process.poll()
            if code is not None:
                finished.append(key)
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        f"[{utc_now()}] finished {key} gpu={item.gpu} exit={code}\n"
                    )
        for key in finished:
            running.pop(key, None)
        safe, gpu_rows = safe_gpus()
        occupied_by_supervisor = {item.gpu for item in running.values()}
        available = [gpu for gpu in safe if gpu not in occupied_by_supervisor]
        launched: list[str] = []
        if not stopping:
            while queue and available:
                task = queue.pop(0)
                if task_completed(task):
                    continue
                gpu = available.pop(0)
                item = launch(task, gpu, repo, log_path.parent / "tasks")
                running[task.key] = item
                launched.append(task.key)
        payload = {
            "timestamp": utc_now(),
            "poll_seconds": POLL_SECONDS,
            "min_free_mib": MIN_FREE_MIB,
            "max_utilization": MAX_UTILIZATION,
            "stopping": stopping,
            "queue": [task.key for task in queue],
            "running": {
                key: {
                    "gpu": item.gpu,
                    "pid": item.process.pid,
                    "started_at": item.started_at,
                    "log": str(item.log_path),
                }
                for key, item in running.items()
            },
            "safe_gpu_indices": safe,
            "launched_this_poll": launched,
            "gpus": gpu_rows,
        }
        write_snapshot(snapshot_path, payload)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if not queue and not running:
            break
        time.sleep(POLL_SECONDS)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{utc_now()}] supervisor complete; all queued tasks exited\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
