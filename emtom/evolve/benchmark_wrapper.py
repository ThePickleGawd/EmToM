"""Wrapper around run_emtom.sh benchmark for the evolutionary pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_calibration_entry(
    calibration: list,
    model: Optional[str] = None,
    agent_models: Optional[dict] = None,
) -> Optional[dict]:
    """Find a calibration entry matching the given criteria.

    If agent_models is provided, find exact match on agent_models dict.
    If only model is provided, find entry where ALL agents use that model.
    Returns the most recent match (last in list).
    """
    if not calibration:
        return None

    best = None
    for entry in calibration:
        entry_agent_models = entry.get("agent_models", {})

        if agent_models is not None:
            if entry_agent_models == agent_models:
                best = entry
        elif model is not None:
            if entry_agent_models and all(
                v == model for v in entry_agent_models.values()
            ):
                best = entry
            elif entry.get("_legacy_key") == model:
                # Match migrated legacy entries by their original dict key
                best = entry
    return best


def _build_trajectory_from_log(results_dir: str) -> List[Dict[str, Any]]:
    """Build trajectory from planner log files in a benchmark results directory.

    Reads planner-log-*.json files, extracts action_history, and groups by turn.
    Same logic as agent.py:_build_trajectory.
    """
    from collections import defaultdict

    results_path = Path(results_dir)
    log_files = sorted(results_path.glob("planner-log-*.json"))
    if not log_files:
        return []

    # Merge action_history from all planner logs
    all_actions: List[Dict[str, Any]] = []
    for log_file in log_files:
        try:
            with open(log_file) as f:
                log_data = json.load(f)
            all_actions.extend(log_data.get("action_history", []))
        except Exception:
            continue

    if not all_actions:
        return []

    # Group by turn
    turns: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "agents": {},
        "subtasks_completed": []
    })

    for record in all_actions:
        turn = record.get("turn", 0)
        if record.get("type") == "subtask_completion":
            turns[turn]["subtasks_completed"].extend(
                record.get("subtasks_completed", [])
            )
        else:
            agent_id = record.get("agent", "unknown")
            turns[turn]["agents"][agent_id] = {
                "action": record.get("action", ""),
                "observation": record.get("result", ""),
            }

    trajectory = []
    for turn_num in sorted(turns.keys()):
        entry = turns[turn_num]
        trajectory.append({
            "turn": turn_num,
            "agents": entry["agents"],
            "subtasks_completed": entry["subtasks_completed"],
        })

    return trajectory


def _migrate_legacy_calibration(calibration: Any) -> list:
    """Convert legacy dict-format calibration to list format.

    Old format: {"model_name": {passed, tested_at, ...}, ...}
    New format: [{agent_models: {...}, passed, tested_at, ...}, ...]
    """
    if isinstance(calibration, list):
        return calibration
    if not isinstance(calibration, dict):
        return []

    migrated = []
    for key, entry in calibration.items():
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        # Preserve original key for reference
        new_entry["_legacy_key"] = key
        # If no agent_models set, infer: all agents use this model
        if "agent_models" not in new_entry:
            new_entry["agent_models"] = {}  # unknown agents, but key preserved
        migrated.append(new_entry)
    return migrated


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
    category: str = ""
    team_model_mapping: Optional[dict] = None
    agent_model_mapping: Optional[dict] = None
    results_dir: str = ""


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

        # Derive results directory from the summary file location
        results_base_dir = str(sf.parent)

        for r in summary.get("results", []):
            if r.get("skipped", False):
                total_skipped += 1
                continue

            evaluation = r.get("evaluation", {})
            task_id = r.get("task_id", "")
            task_result = TaskResult(
                task_id=task_id,
                title=r.get("title", ""),
                task_path=r.get("task_id", ""),  # Will be resolved later
                success=r.get("success", False),
                steps=r.get("steps", 0),
                turns=r.get("turns", 0),
                percent_complete=evaluation.get("percent_complete", 0.0),
                skipped=False,
                error=r.get("error"),
                evaluation=evaluation,
                category=r.get("category", ""),
                team_model_mapping=r.get("team_model_mapping"),
                agent_model_mapping=r.get("agent_model_mapping"),
                results_dir=f"{results_base_dir}/{task_id}" if task_id else "",
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


def _detect_gpu_ids() -> List[int]:
    """Detect available CUDA GPU IDs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(x.strip()) for x in result.stdout.strip().split("\n")]
    except Exception:
        pass
    return [0]


def run_benchmark_parallel(
    tasks_dir: str,
    model: str,
    output_dir: str,
    max_workers: int = 50,
    no_video: bool = True,
    category: Optional[str] = None,
    team_model_map: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> BenchmarkResults:
    """Run benchmark in parallel — one process per task JSON.

    For each task file, creates a temp single-task directory and spawns
    a separate benchmark process. Manages a pool of up to max_workers
    concurrent processes. Merges all results into a single BenchmarkResults.

    Each process is assigned a GPU via round-robin over all available CUDA
    devices (set through CUDA_VISIBLE_DEVICES).

    Args:
        tasks_dir: Directory containing task JSONs.
        model: Model short name.
        output_dir: Base output directory for per-task results.
        max_workers: Maximum concurrent benchmark processes.
        no_video: Disable video recording.
        category: Optional category filter.
        team_model_map: Optional team→model mapping string (e.g. "team_0=gpt-5.2,team_1=sonnet").
        extra_args: Additional args forwarded to each subprocess benchmark call.

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

    # Detect GPUs for round-robin distribution
    gpu_ids = _detect_gpu_ids()

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
    print(f"[evolve] GPUs detected: {gpu_ids} ({len(gpu_ids)} devices)")
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
                    "--no-calibration",
                ]
                if no_video:
                    cmd.append("--no-video")
                if category:
                    cmd.extend(["--category", category])
                if team_model_map:
                    cmd.extend(["--team-model-map", team_model_map])
                if extra_args:
                    cmd.extend(extra_args)

                # Round-robin GPU assignment
                gpu_id = gpu_ids[job_idx % len(gpu_ids)]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

                log_file = log_dir / f"bench_{stem}.log"
                fh = open(log_file, "w")
                proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)
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


def _build_agent_models_from_result(
    result: TaskResult,
    model: str,
    team_model_map: Optional[Dict[str, str]] = None,
    task_data: Optional[dict] = None,
) -> Dict[str, str]:
    """Build per-agent model mapping from a TaskResult.

    Priority:
    1. result.agent_model_mapping (from benchmark_summary.json)
    2. Expand team_model_map using task's team_assignment
    3. Fall back to all agents using `model`
    """
    # 1. Direct agent_model_mapping from benchmark results
    if result.agent_model_mapping:
        return {
            agent_id: info.get("model", model)
            for agent_id, info in result.agent_model_mapping.items()
        }

    # 2. Expand team_model_map via task's team_assignment
    if team_model_map and task_data:
        team_assignment = task_data.get("team_assignment", {})
        agent_models = {}
        for team_id, agents in team_assignment.items():
            team_model = team_model_map.get(team_id, model)
            for agent_id in agents:
                agent_models[agent_id] = team_model
        if agent_models:
            return agent_models

    # 3. All agents use the same model
    num_agents = task_data.get("num_agents", 2) if task_data else 2
    return {f"agent_{i}": model for i in range(num_agents)}


def update_calibration_from_benchmark(
    benchmark_results: BenchmarkResults,
    tasks_dir: str,
    team_model_map: Optional[Dict[str, str]] = None,
) -> None:
    """Write benchmark results back into task JSONs as calibration entries.

    Uses the unified array-based calibration format. Each entry records
    per-agent model mappings, category-specific fields, and trajectory.
    Deduplicates by agent_models (replaces existing entry with same matchup).

    Args:
        benchmark_results: Parsed benchmark results.
        tasks_dir: Directory containing source task JSONs.
        team_model_map: Optional team->model mapping (e.g. {"team_0": "gpt-5.2", "team_1": "sonnet"}).
    """
    tasks_path = Path(tasks_dir)
    model = benchmark_results.model

    # Build lookup: task_id -> TaskResult
    result_map = {r.task_id: r for r in benchmark_results.results}

    updated = 0
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

        # Migrate legacy dict format to list
        raw_cal = task_data.get("calibration", [])
        calibration = _migrate_legacy_calibration(raw_cal)

        # Build per-agent model mapping
        agent_models = _build_agent_models_from_result(
            result, model, team_model_map, task_data
        )

        # Build calibration entry
        entry: Dict[str, Any] = {
            "tested_at": datetime.now().isoformat(),
            "agent_models": agent_models,
            "passed": result.success,
            "steps": result.steps,
            "percent_complete": result.percent_complete,
        }

        # Category-specific fields
        evaluation = result.evaluation
        category = result.category or task_data.get("category", "")

        if category == "competitive":
            entry["passed"] = evaluation.get("winner") is not None
            entry["winner"] = evaluation.get("winner")
            if evaluation.get("team_status"):
                entry["team_status"] = evaluation["team_status"]
            if evaluation.get("team_progress"):
                entry["team_progress"] = evaluation["team_progress"]

        elif category == "mixed":
            if "main_goal_success" in evaluation:
                entry["main_goal_success"] = evaluation["main_goal_success"]
            if "agent_subgoal_status" in evaluation:
                entry["agent_subgoal_status"] = evaluation["agent_subgoal_status"]

        # Build trajectory from planner logs if available
        if result.results_dir:
            trajectory = _build_trajectory_from_log(result.results_dir)
            if trajectory:
                entry["trajectory"] = trajectory

        # Deduplicate: replace existing entry with same agent_models, else append
        replaced = False
        for i, existing in enumerate(calibration):
            if existing.get("agent_models") == agent_models:
                calibration[i] = entry
                replaced = True
                break
        if not replaced:
            calibration.append(entry)

        task_data["calibration"] = calibration

        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)
        updated += 1

    print(f"[calibration] Updated {updated} task(s) in {tasks_dir} for model {model}")