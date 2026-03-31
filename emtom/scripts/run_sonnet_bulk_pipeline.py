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
from statistics import mean, stdev
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATIONS_DIR = REPO_ROOT / "outputs" / "generations"
AUTOMATION_DIR = REPO_ROOT / "outputs" / "automation"
DEFAULT_BENCHMARK_MODELS = [
    "deepseek-v3.2",
    "haiku",
    "opus",
    "kimi-k2.5",
    "o3",
    "gpt-5.4",
    "gpt-5.4-mini",
]


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


def find_generation_dir_for_pid(pid: int, known_dirs: set[Path], poll_seconds: int) -> Path:
    while True:
        current_dirs = {path.resolve() for path in GENERATIONS_DIR.glob("*") if path.is_dir()}
        for candidate in sorted(current_dirs - known_dirs):
            launcher_log = candidate / "launcher.log"
            if not launcher_log.exists():
                continue
            return candidate
        if not pid_alive(pid):
            raise RuntimeError(f"bulk pid {pid} exited before a generation dir was discovered")
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


@dataclass
class BulkSummary:
    submitted: int
    worker_successes: int
    worker_failures: int


def summarize_run(run_dir: Path) -> BulkSummary:
    workers_dir = run_dir / "workers"
    submitted = 0
    worker_successes = 0
    worker_failures = 0
    for worker_file in workers_dir.glob("*/worker.json"):
        try:
            worker = json.loads(worker_file.read_text())
        except Exception:
            continue
        submitted += len(worker.get("submitted_tasks", []))
        if worker.get("submitted_count", 0) > 0:
            worker_successes += 1
        elif worker.get("failed") or worker.get("status") == "stopped":
            worker_failures += 1
    return BulkSummary(
        submitted=submitted,
        worker_successes=worker_successes,
        worker_failures=worker_failures,
    )


def copy_tasks(task_paths: list[Path], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in task_paths:
        dest = dest_dir / src.name
        if dest.exists():
            continue
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def run_benchmark(tasks_dir: Path, model: str, max_workers: int) -> int:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO_ROOT / "outputs" / "emtom" / f"{timestamp}-sonnet-bulk-benchmark-{model.replace('/', '_')}"
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
        str(out_dir),
    ]
    log(f"Starting benchmark for {model}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    log(f"Finished benchmark for {model} exit_code={proc.returncode}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sonnet bulk generation, monitor it, collect submitted tasks, then benchmark models.")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--model", default="sonnet-4.6")
    parser.add_argument("--output-dir", default="data/emtom/tasks")
    parser.add_argument("--monitor-seconds", type=int, default=900)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--existing-bulk-pid", type=int, default=None)
    parser.add_argument("--existing-run-dir", default=None)
    parser.add_argument("--benchmark-model", dest="benchmark_models", action="append")
    args = parser.parse_args()

    benchmark_models = args.benchmark_models or DEFAULT_BENCHMARK_MODELS
    known_dirs = {path.resolve() for path in GENERATIONS_DIR.glob("*") if path.is_dir()}

    if args.existing_bulk_pid is not None:
        bulk_pid = args.existing_bulk_pid
        if not pid_alive(bulk_pid):
            raise RuntimeError(f"existing bulk pid is not alive: {bulk_pid}")
        if not args.existing_run_dir:
            raise RuntimeError("--existing-run-dir is required with --existing-bulk-pid")
        run_dir = (
            (REPO_ROOT / args.existing_run_dir).resolve()
            if not Path(args.existing_run_dir).is_absolute()
            else Path(args.existing_run_dir).resolve()
        )
        if not run_dir.exists():
            raise RuntimeError(f"existing run dir does not exist: {run_dir}")
        bulk_proc = None
        log(f"Attaching to existing bulk pid={bulk_pid}")
        log(f"Generation dir: {run_dir}")
    else:
        bulk_cmd = [
            "./emtom/bulk_generate.sh",
            "--num-tasks",
            str(args.num_tasks),
            "--remove",
            "tom",
            "pddl",
            "simulation",
            "--model",
            args.model,
            "--output-dir",
            args.output_dir,
        ]
        log(f"Launching bulk generation: {' '.join(bulk_cmd)}")
        bulk_proc = subprocess.Popen(bulk_cmd, cwd=REPO_ROOT)
        bulk_pid = bulk_proc.pid
        log(f"Bulk pid={bulk_pid}")
        run_dir = find_generation_dir_for_pid(bulk_pid, known_dirs, poll_seconds=2)
        log(f"Generation dir: {run_dir}")

    while pid_alive(bulk_pid):
        time.sleep(args.monitor_seconds)
        summary = summarize_run(run_dir)
        log(
            "Monitor heartbeat: "
            f"submitted_tasks={summary.submitted} "
            f"worker_successes={summary.worker_successes} "
            f"worker_failures={summary.worker_failures}"
        )

    exit_code = bulk_proc.wait() if bulk_proc is not None else 0
    log(f"Bulk generation exited with code {exit_code}")

    submitted_tasks = list(iter_submitted_tasks(run_dir))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    passed_dir = REPO_ROOT / "data" / "emtom" / f"tasks_{timestamp}_sonnet46_bulk_submitted"
    copied = copy_tasks(submitted_tasks, passed_dir)
    log(f"Collected {len(submitted_tasks)} submitted task(s); copied {len(copied)} into {passed_dir}")

    if not submitted_tasks:
        log("No submitted tasks were produced; skipping benchmarks")
        return 1

    benchmark_stats: dict[str, dict] = {}
    failures: list[tuple[str, int, int]] = []
    for model in benchmark_models:
        run_pass_rates: list[float] = []
        run_output_dirs: list[str] = []
        for repeat in range(1, args.benchmark_repeats + 1):
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = REPO_ROOT / "outputs" / "emtom" / f"{timestamp}-sonnet-bulk-benchmark-{model.replace('/', '_')}-run{repeat}"
            cmd = [
                "./emtom/run_emtom.sh",
                "benchmark",
                "--tasks-dir",
                str(passed_dir),
                "--model",
                model,
                "--observation-mode",
                "text",
                "--max-workers",
                str(args.max_workers),
                "--output-dir",
                str(out_dir),
            ]
            log(f"Starting benchmark for {model} run={repeat}/{args.benchmark_repeats}")
            proc = subprocess.run(cmd, cwd=REPO_ROOT)
            if proc.returncode != 0:
                log(f"Finished benchmark for {model} run={repeat} exit_code={proc.returncode}")
                failures.append((model, repeat, proc.returncode))
                continue
            from emtom.evolve.benchmark_wrapper import parse_benchmark_results
            results = parse_benchmark_results(str(out_dir), model)
            run_pass_rates.append(results.pass_rate)
            run_output_dirs.append(str(out_dir))
            log(
                f"Finished benchmark for {model} run={repeat} "
                f"pass_rate={results.pass_rate:.2f}% ({results.passed}/{results.total})"
            )
        if run_pass_rates:
            benchmark_stats[model] = {
                "runs": len(run_pass_rates),
                "pass_rates": run_pass_rates,
                "mean_pass_rate": mean(run_pass_rates),
                "std_pass_rate": stdev(run_pass_rates) if len(run_pass_rates) > 1 else 0.0,
                "output_dirs": run_output_dirs,
            }

    stats_path = AUTOMATION_DIR / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}-sonnet46-benchmark-stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "tasks_dir": str(passed_dir),
                "benchmark_repeats": args.benchmark_repeats,
                "models": benchmark_stats,
                "failures": [
                    {"model": model, "run": run, "exit_code": code}
                    for model, run, code in failures
                ],
            },
            f,
            indent=2,
        )
    log(f"Wrote benchmark stats to {stats_path}")

    if failures:
        for model, run, code in failures:
            log(f"Benchmark failed for {model} run={run} exit_code={code}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
