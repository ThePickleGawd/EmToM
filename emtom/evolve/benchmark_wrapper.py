"""Wrapper around run_emtom.sh benchmark for the evolutionary pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class TaskResult:
    task_id: str
    title: str
    task_path: str
    success: bool
    steps: int
    turns: int
    percent_complete: float
    skipped: bool
    error: Optional[str]
    evaluation: dict


@dataclass
class BenchmarkResults:
    model: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: List[TaskResult] = field(default_factory=list)


def run_benchmark(
    tasks_dir: str,
    model: str,
    output_dir: str,
    no_video: bool = True,
    category: Optional[str] = None,
) -> BenchmarkResults:
    """Run benchmark via run_emtom.sh and parse results.

    Args:
        tasks_dir: Directory containing task JSONs to benchmark.
        model: Model short name (e.g. "haiku", "gpt-5-mini").
        output_dir: Base output directory. Results land at
            <output_dir>-{N}agents/results/benchmark_summary.json.
        no_video: Disable video recording.
        category: Optional task category filter.

    Returns:
        BenchmarkResults with parsed per-task results.
    """
    cmd = [
        "./emtom/run_emtom.sh", "benchmark",
        "--tasks-dir", str(tasks_dir),
        "--model", model,
        "--output-dir", str(output_dir),
    ]
    if no_video:
        cmd.append("--no-video")
    if category:
        cmd.extend(["--category", category])

    print(f"[evolve] Running benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark command failed (exit={result.returncode}) for model={model}, tasks_dir={tasks_dir}"
        )

    return parse_benchmark_results(output_dir, model)


def parse_benchmark_results(output_dir: str, model: str) -> BenchmarkResults:
    """Parse benchmark_summary.json files from output directory.

    Merges results across agent-count groups by globbing
    <output_dir>-*agents/results/benchmark_summary.json.
    """
    output_path = Path(output_dir)
    summary_files = sorted(output_path.parent.glob(f"{output_path.name}-*agents/results/benchmark_summary.json"))

    if not summary_files:
        # Try exact path as well (single-agent-count case)
        exact = output_path / "results" / "benchmark_summary.json"
        if exact.exists():
            summary_files = [exact]

    if not summary_files:
        raise FileNotFoundError(
            f"No benchmark_summary.json found under '{output_dir}'. "
            "Benchmark likely failed before writing results."
        )

    all_task_results: List[TaskResult] = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for sf in summary_files:
        with open(sf) as f:
            summary = json.load(f)

        for r in summary.get("results", []):
            if r.get("skipped", False):
                total_skipped += 1
                continue

            evaluation = r.get("evaluation", {})
            task_result = TaskResult(
                task_id=r.get("task_id", ""),
                title=r.get("title", ""),
                task_path=r.get("task_id", ""),  # Will be resolved later
                success=r.get("success", False),
                steps=r.get("steps", 0),
                turns=r.get("turns", 0),
                percent_complete=evaluation.get("percent_complete", 0.0),
                skipped=False,
                error=r.get("error"),
                evaluation=evaluation,
            )
            all_task_results.append(task_result)

            if task_result.success:
                total_passed += 1
            else:
                total_failed += 1

    total = total_passed + total_failed
    if total == 0:
        raise RuntimeError(
            f"Benchmark produced zero non-skipped results for '{output_dir}' "
            f"(skipped={total_skipped}, summaries={len(summary_files)})."
        )

    pass_rate = (total_passed / total * 100) if total > 0 else 0.0

    return BenchmarkResults(
        model=model,
        total=total,
        passed=total_passed,
        failed=total_failed,
        pass_rate=pass_rate,
        results=all_task_results,
    )


def run_benchmark_parallel(
    tasks_dir: str,
    model: str,
    output_dir: str,
    max_workers: int = 50,
    no_video: bool = True,
    category: Optional[str] = None,
) -> BenchmarkResults:
    """Run benchmark in parallel — one process per task JSON.

    For each task file, creates a temp single-task directory and spawns
    a separate benchmark process. Manages a pool of up to max_workers
    concurrent processes. Merges all results into a single BenchmarkResults.

    Args:
        tasks_dir: Directory containing task JSONs.
        model: Model short name.
        output_dir: Base output directory for per-task results.
        max_workers: Maximum concurrent benchmark processes.
        no_video: Disable video recording.
        category: Optional category filter.

    Returns:
        Merged BenchmarkResults across all tasks.
    """
    tasks_path = Path(tasks_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_dir = out_path / "logs"
    log_dir.mkdir(exist_ok=True)

    task_files = sorted(tasks_path.glob("*.json"))
    if not task_files:
        print(f"[evolve] WARNING: no task files in {tasks_dir}", file=sys.stderr)
        return BenchmarkResults(model=model, total=0, passed=0, failed=0, pass_rate=0.0)

    # Prepare per-task jobs: (task_stem, task_input_dir, benchmark_output_dir)
    jobs = []
    for tf in task_files:
        stem = tf.stem
        task_input_dir = out_path / stem / "task_input"
        task_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tf, task_input_dir / tf.name)
        bench_out = str(out_path / stem / "benchmark")
        jobs.append((stem, str(task_input_dir), bench_out))

    total_tasks = len(jobs)
    job_idx = 0
    active: List[tuple] = []  # (stem, bench_out, Popen, log_file_handle)
    completed_stems: List[str] = []
    failed_stems: List[str] = []

    spinner_chars = ["|", "/", "-", "\\"]
    spinner_idx = 0
    last_status = None

    print(f"[evolve] Parallel benchmark: {total_tasks} tasks, max_workers={max_workers}")
    print(f"[evolve] Benchmark output dir: {out_path.resolve()}")
    print(f"[evolve] Benchmark logs: {log_dir.resolve()}")

    try:
        while True:
            # Reap finished processes
            still_active = []
            for stem, bench_out, proc, fh in active:
                if proc.poll() is not None:
                    fh.close()
                    completed_stems.append(stem)
                    if proc.returncode != 0:
                        failed_stems.append(stem)
                        print(
                            f"[evolve] WARNING: benchmark for {stem} exited with code {proc.returncode}",
                            file=sys.stderr,
                        )
                else:
                    still_active.append((stem, bench_out, proc, fh))
            active = still_active

            # Spawn new processes
            while len(active) < max_workers and job_idx < total_tasks:
                stem, task_input, bench_out = jobs[job_idx]
                cmd = [
                    "./emtom/run_emtom.sh", "benchmark",
                    "--tasks-dir", task_input,
                    "--model", model,
                    "--output-dir", bench_out,
                ]
                if no_video:
                    cmd.append("--no-video")
                if category:
                    cmd.extend(["--category", category])

                log_file = log_dir / f"bench_{stem}.log"
                fh = open(log_file, "w")
                proc = subprocess.Popen(cmd, stdout=fh, stderr=fh)
                active.append((stem, bench_out, proc, fh))
                job_idx += 1

            done = len(completed_stems)
            if done >= total_tasks and not active:
                if sys.stdout.isatty():
                    print()  # finish in-place status line
                print(f"[evolve] Benchmark: {done}/{total_tasks} tasks complete — done!")
                break

            spinner = spinner_chars[spinner_idx % len(spinner_chars)]
            spinner_idx += 1
            status = (
                f"[evolve] {spinner} benchmarking: {done}/{total_tasks} tasks complete "
                f"(active={len(active)})"
            )
            if sys.stdout.isatty():
                print(f"\r{status}", end="", flush=True)
            elif status != last_status:
                print(status)
                last_status = status

            if not active and job_idx >= total_tasks:
                if sys.stdout.isatty():
                    print()  # finish in-place status line
                break

            time.sleep(10)
    finally:
        for stem, bench_out, proc, fh in active:
            proc.terminate()
        for stem, bench_out, proc, fh in active:
            proc.wait()
            fh.close()

    if failed_stems:
        raise RuntimeError(
            f"Parallel benchmark had {len(failed_stems)} failed subprocesses: "
            f"{', '.join(sorted(failed_stems)[:10])}. See logs in {log_dir}"
        )

    # Merge results from all per-task benchmark outputs
    all_task_results: List[TaskResult] = []
    total_passed = 0
    total_failed = 0

    for stem, task_input, bench_out in jobs:
        try:
            per_task = parse_benchmark_results(bench_out, model)
        except Exception as e:
            raise RuntimeError(
                f"Failed parsing benchmark output for task '{stem}' at '{bench_out}': {e}"
            ) from e
        all_task_results.extend(per_task.results)
        total_passed += per_task.passed
        total_failed += per_task.failed

    total = total_passed + total_failed
    pass_rate = (total_passed / total * 100) if total > 0 else 0.0

    return BenchmarkResults(
        model=model,
        total=total,
        passed=total_passed,
        failed=total_failed,
        pass_rate=pass_rate,
        results=all_task_results,
    )


def update_calibration_from_benchmark(
    benchmark_results: BenchmarkResults,
    tasks_dir: str,
) -> None:
    """Write benchmark results back into task JSONs as calibration entries.

    After running a separate benchmark (e.g. on model upgrade), this merges
    the results into each task's calibration dict so subsequent logic can
    read pass/fail from the task JSON itself.
    """
    tasks_path = Path(tasks_dir)
    model = benchmark_results.model

    # Build lookup: task_id -> TaskResult
    result_map = {r.task_id: r for r in benchmark_results.results}

    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
        except Exception:
            continue

        task_id = task_data.get("task_id", "")
        # Try matching by task_id or by filename stem
        result = result_map.get(task_id) or result_map.get(task_file.stem)
        if result is None:
            continue

        if "calibration" not in task_data:
            task_data["calibration"] = {}

        task_data["calibration"][model] = {
            "passed": result.success,
            "tested_at": datetime.now().isoformat(),
            "steps": result.steps,
            "percent_complete": result.percent_complete,
        }

        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)

    print(f"[calibration] Updated calibration in {tasks_dir} for model {model}")