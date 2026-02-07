"""Prepare annotated sampled_tasks directories for evolutionary generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from emtom.evolve.benchmark_wrapper import BenchmarkResults, TaskResult


def prepare_sampled_tasks_dir(
    benchmark_results: BenchmarkResults,
    tasks_dir: str,
    output_dir: str,
    fail_count: int = 9,
    pass_count: int = 1,
) -> Path:
    """Create an annotated sampled_tasks directory from benchmark results.

    Selects a mix of failed and passed tasks, annotates each with
    a _benchmark_result field, and writes them with descriptive filenames.

    Args:
        benchmark_results: Parsed benchmark results.
        tasks_dir: Directory containing the original task JSONs.
        output_dir: Where to write the sampled_tasks directory.
        fail_count: Number of failed tasks to include.
        pass_count: Number of passed tasks to include.

    Returns:
        Path to the created sampled_tasks directory.
    """
    sampled_dir = Path(output_dir)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    failed = [r for r in benchmark_results.results if not r.success]
    passed = [r for r in benchmark_results.results if r.success]

    # Prioritize partial completions — they reveal *why* the agent got stuck
    failed.sort(key=lambda r: r.percent_complete, reverse=True)

    selected_failed = failed[:fail_count]
    selected_passed = random.sample(passed, min(pass_count, len(passed))) if passed else []

    tasks_path = Path(tasks_dir)

    for i, result in enumerate(selected_failed, 1):
        task_data = _load_and_annotate(result, benchmark_results.model, tasks_path)
        if task_data is None:
            continue
        pct = int(result.percent_complete * 100)
        filename = f"failed_{i}_{pct}pct.json"
        with open(sampled_dir / filename, "w") as f:
            json.dump(task_data, f, indent=2)

    for i, result in enumerate(selected_passed, 1):
        task_data = _load_and_annotate(result, benchmark_results.model, tasks_path)
        if task_data is None:
            continue
        filename = f"passed_{i}.json"
        with open(sampled_dir / filename, "w") as f:
            json.dump(task_data, f, indent=2)

    total = len(list(sampled_dir.glob("*.json")))
    print(f"[evolve] Prepared sampled_tasks: {total} files ({len(selected_failed)} failed, {len(selected_passed)} passed)")
    return sampled_dir


def _load_and_annotate(
    result: TaskResult,
    model: str,
    tasks_dir: Path,
) -> Optional[dict]:
    """Load task JSON and add _benchmark_result annotation."""
    # Try to find the task file by task_id
    task_file = _find_task_file(result.task_id, tasks_dir)
    if task_file is None:
        print(f"[evolve] WARNING: Could not find task file for {result.task_id}")
        return None

    with open(task_file) as f:
        task_data = json.load(f)

    task_data["_benchmark_result"] = {
        "model": model,
        "outcome": "PASSED" if result.success else "FAILED",
        "percent_complete": result.percent_complete,
        "completed_subtasks": result.evaluation.get("completed_subtasks", []),
        "total_subtasks": result.evaluation.get("total_subtasks", 0),
    }

    return task_data


def _find_task_file(task_id: str, tasks_dir: Path) -> Optional[Path]:
    """Find a task JSON file by task_id in the tasks directory."""
    # Direct filename match
    direct = tasks_dir / f"{task_id}.json"
    if direct.exists():
        return direct

    # Search all JSON files for matching task_id
    for f in tasks_dir.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("task_id") == task_id:
                return f
        except Exception:
            continue

    return None


def build_evolution_query(
    benchmark_results: BenchmarkResults,
    tier_model: str,
    generation_idx: int,
) -> str:
    """Build the --query text for the next generation round.

    Tells the agent what the _benchmark_result annotations mean and
    guides it to produce harder tasks.
    """
    rate = benchmark_results.pass_rate

    # Directional guidance only — concrete constraints (tom_level, mechanics,
    # agent count) come from the --difficulty parameter to avoid conflicts.
    guidance = ""
    if rate < 10:
        guidance = (
            "The model barely solved any tasks. Try a DIFFERENT type of difficulty — "
            "vary the information asymmetry patterns, coordination structures, "
            "or deception/cooperation dynamics."
        )
    elif rate < 30:
        guidance = (
            "The model struggled significantly. Study what made the failed tasks hard "
            "and amplify those patterns in new scenarios."
        )
    elif rate < 60:
        guidance = (
            "The model had moderate success. Generate tasks that combine multiple "
            "sources of difficulty from the failed examples."
        )
    elif rate < 95:
        guidance = (
            "The model solved most tasks. Generate harder tasks that exploit the "
            "weaknesses shown in the failed examples. Follow the difficulty constraints "
            "from the Difficulty section — they define the specific complexity level."
        )
    else:
        guidance = (
            "The model solved nearly everything. Generate the hardest tasks you can "
            "within the difficulty constraints. Focus on novel challenge patterns."
        )

    return (
        f"The sampled_tasks/ directory contains benchmark-tested tasks from a previous tier. "
        f"Each has a `_benchmark_result` field showing how {tier_model} performed.\n"
        f"Files named `failed_*` = tasks the model COULD NOT solve.\n"
        f"Files named `passed_*` = tasks the model COULD solve.\n\n"
        f"Study each task's _benchmark_result to understand what makes tasks hard vs easy.\n"
        f"Current pass rate for {tier_model}: {rate:.1f}%\n\n"
        f"Generate tasks HARDER than what {tier_model} failed on. {guidance}"
    )
