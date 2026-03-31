#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_MODELS = [
    "haiku",
    "opus",
    "deepseek-v3.2",
    "kimi-k2.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "o3",
]


@dataclass
class BulkRunSpec:
    pid: int
    label: str
    run_dir: Path | None
    expected_model: str
    expected_output_dir: Path
    expected_num_tasks: int


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def wait_for_pid_exit(pid: int, label: str, poll_seconds: int) -> None:
    while pid_alive(pid):
        log(f"Waiting for {label} pid={pid} to finish")
        time.sleep(poll_seconds)
    log(f"{label} pid={pid} finished")


def discover_new_generation_dir(
    known_dirs: set[Path],
    spec: BulkRunSpec,
    poll_seconds: int,
) -> Path:
    while True:
        candidates = sorted(
            path for path in (REPO_ROOT / "outputs" / "generations").glob("*") if path.is_dir() and path not in known_dirs
        )
        for candidate in candidates:
            launcher = candidate / "launcher.log"
            if not launcher.exists():
                continue
            try:
                text = launcher.read_text(errors="replace")
            except OSError:
                continue
            if f"Model:              \x1b[0;32m{spec.expected_model}\x1b[0m" not in text and f"Model:              {spec.expected_model}" not in text:
                continue
            if str(spec.expected_output_dir.relative_to(REPO_ROOT)) not in text and str(spec.expected_output_dir) not in text:
                continue
            if f"Total tasks:        \x1b[0;32m{spec.expected_num_tasks}\x1b[0m" not in text and f"Total tasks:        {spec.expected_num_tasks}" not in text:
                continue
            log(f"Discovered generation dir for {spec.label}: {candidate}")
            return candidate
        if not pid_alive(spec.pid):
            raise RuntimeError(f"Could not discover generation dir for {spec.label} before pid {spec.pid} exited")
        time.sleep(poll_seconds)


def iter_submitted_tasks(run_dir: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    workers_dir = run_dir / "workers"
    if not workers_dir.exists():
        return []
    for worker_file in sorted(workers_dir.glob("*/worker.json")):
        try:
            worker = json.loads(worker_file.read_text())
        except Exception:
            continue
        for task_path in worker.get("submitted_tasks", []):
            path = Path(task_path)
            if path.exists() and path not in seen:
                seen.add(path)
                yield path
    for events_file in sorted(workers_dir.glob("*/events.jsonl")):
        try:
            lines = events_file.read_text().splitlines()
        except Exception:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event_type") != "generation_finished":
                continue
            for task_path in event.get("submitted_tasks", []):
                path = Path(task_path)
                if path.exists() and path not in seen:
                    seen.add(path)
                    yield path


def copy_tasks(task_paths: list[Path], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in task_paths:
        dest = dest_dir / src.name
        if dest.exists():
            log(f"Skipping existing task file: {dest.name}")
            continue
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def find_active_benchmark_pids(tasks_dir: Path) -> list[int]:
    cmd = ["ps", "-eo", "pid,args"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    matches: list[int] = []
    tasks_dir_text = str(tasks_dir.resolve())
    tasks_dir_rel = str(tasks_dir)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        if not pid_text.isdigit():
            continue
        if "benchmark" not in args:
            continue
        if tasks_dir_text not in args and tasks_dir_rel not in args:
            continue
        matches.append(int(pid_text))
    return matches


def wait_for_benchmark_clear(tasks_dir: Path, poll_seconds: int) -> None:
    while True:
        pids = find_active_benchmark_pids(tasks_dir)
        if not pids:
            return
        log(f"Waiting for existing benchmark(s) on {tasks_dir} to finish: {pids}")
        time.sleep(poll_seconds)


def run_benchmark(model: str, tasks_dir: Path, max_workers: int, output_root: Path) -> int:
    safe_model = model.replace("/", "_")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_root / f"{timestamp}-batch-benchmark-{safe_model}"
    cmd = [
        "./emtom/run_emtom.sh",
        "benchmark",
        "--tasks-dir",
        str(tasks_dir),
        "--model",
        model,
        "--observation-mode",
        "text",
        "--max-workers",
        str(max_workers),
        "--output-dir",
        str(output_dir),
    ]
    log(f"Starting benchmark for model={model}")
    log(f"Command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    log(f"Benchmark finished for model={model} with exit_code={proc.returncode}")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for bulk runs, merge their tasks, then benchmark a target batch.")
    parser.add_argument("--bulk1-pid", type=int, required=True)
    parser.add_argument("--bulk1-label", default="bulk1")
    parser.add_argument("--bulk1-run-dir", default=None)
    parser.add_argument("--bulk1-model", required=True)
    parser.add_argument("--bulk1-output-dir", required=True)
    parser.add_argument("--bulk1-num-tasks", type=int, required=True)
    parser.add_argument("--bulk2-pid", type=int, required=True)
    parser.add_argument("--bulk2-label", default="bulk2")
    parser.add_argument("--bulk2-run-dir", default=None)
    parser.add_argument("--bulk2-model", required=True)
    parser.add_argument("--bulk2-output-dir", required=True)
    parser.add_argument("--bulk2-num-tasks", type=int, required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--benchmark-model", dest="benchmark_models", action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = (REPO_ROOT / args.batch_dir).resolve() if not Path(args.batch_dir).is_absolute() else Path(args.batch_dir).resolve()
    output_root = REPO_ROOT / "outputs" / "emtom"
    output_root.mkdir(parents=True, exist_ok=True)

    bulk1 = BulkRunSpec(
        pid=args.bulk1_pid,
        label=args.bulk1_label,
        run_dir=(
            (REPO_ROOT / args.bulk1_run_dir).resolve()
            if args.bulk1_run_dir and not Path(args.bulk1_run_dir).is_absolute()
            else Path(args.bulk1_run_dir).resolve()
        ) if args.bulk1_run_dir else None,
        expected_model=args.bulk1_model,
        expected_output_dir=((REPO_ROOT / args.bulk1_output_dir).resolve() if not Path(args.bulk1_output_dir).is_absolute() else Path(args.bulk1_output_dir).resolve()),
        expected_num_tasks=args.bulk1_num_tasks,
    )
    bulk2 = BulkRunSpec(
        pid=args.bulk2_pid,
        label=args.bulk2_label,
        run_dir=(
            (REPO_ROOT / args.bulk2_run_dir).resolve()
            if args.bulk2_run_dir and not Path(args.bulk2_run_dir).is_absolute()
            else Path(args.bulk2_run_dir).resolve()
        ) if args.bulk2_run_dir else None,
        expected_model=args.bulk2_model,
        expected_output_dir=((REPO_ROOT / args.bulk2_output_dir).resolve() if not Path(args.bulk2_output_dir).is_absolute() else Path(args.bulk2_output_dir).resolve()),
        expected_num_tasks=args.bulk2_num_tasks,
    )

    benchmark_models = args.benchmark_models or DEFAULT_BENCHMARK_MODELS

    log(f"bulk1 cmdline: {read_pid_cmdline(bulk1.pid)}")
    log(f"bulk2 cmdline: {read_pid_cmdline(bulk2.pid)}")
    log(f"Target batch dir: {batch_dir}")
    log(f"Benchmark models: {', '.join(benchmark_models)}")

    known_generation_dirs = {path.resolve() for path in (REPO_ROOT / "outputs" / "generations").glob("*") if path.is_dir()}

    if bulk1.run_dir is None:
        log(f"Waiting for {bulk1.label} to acquire and create its generation dir")
        bulk1.run_dir = discover_new_generation_dir(known_generation_dirs, bulk1, args.poll_seconds)
        known_generation_dirs.add(bulk1.run_dir)
    elif not bulk1.run_dir.exists():
        raise RuntimeError(f"Configured run dir does not exist for {bulk1.label}: {bulk1.run_dir}")

    wait_for_pid_exit(bulk1.pid, bulk1.label, args.poll_seconds)

    if bulk2.run_dir is None:
        log(f"Waiting for {bulk2.label} to acquire and create its generation dir")
        bulk2.run_dir = discover_new_generation_dir(known_generation_dirs, bulk2, args.poll_seconds)
        known_generation_dirs.add(bulk2.run_dir)
    elif not bulk2.run_dir.exists():
        raise RuntimeError(f"Configured run dir does not exist for {bulk2.label}: {bulk2.run_dir}")

    wait_for_pid_exit(bulk2.pid, bulk2.label, args.poll_seconds)

    bulk1_tasks = list(iter_submitted_tasks(bulk1.run_dir))
    bulk2_tasks = list(iter_submitted_tasks(bulk2.run_dir))
    log(f"{bulk1.label} submitted {len(bulk1_tasks)} task(s)")
    log(f"{bulk2.label} submitted {len(bulk2_tasks)} task(s)")

    copied1 = copy_tasks(bulk1_tasks, batch_dir)
    copied2 = copy_tasks(bulk2_tasks, batch_dir)
    log(f"Copied {len(copied1)} task(s) from {bulk1.label} into {batch_dir}")
    log(f"Copied {len(copied2)} task(s) from {bulk2.label} into {batch_dir}")

    wait_for_benchmark_clear(batch_dir, args.poll_seconds)

    failures: list[tuple[str, int]] = []
    for model in benchmark_models:
        code = run_benchmark(model, batch_dir, args.max_workers, output_root)
        if code != 0:
            failures.append((model, code))

    if failures:
        for model, code in failures:
            log(f"FAILED model={model} exit_code={code}")
        return 1

    log("All requested benchmarks completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
