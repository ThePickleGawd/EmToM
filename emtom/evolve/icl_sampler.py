"""Prepare annotated sampled_tasks directories for evolutionary generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from emtom.evolve.benchmark_wrapper import find_calibration_entry, cal_passed, cal_progress


def prepare_sampled_tasks_dir_from_calibration(
    tasks_dir: str,
    model: str,
    output_dir: str,
    fail_count: int = 9,
    pass_count: int = 1,
) -> Path:
    """Create annotated sampled_tasks directory from calibration data in task JSONs.

    Reads calibration[model] from each task JSON to determine pass/fail status.
    Selects a mix of failed and passed tasks, annotates each with a
    _benchmark_result field, and writes them with descriptive filenames.

    Args:
        tasks_dir: Directory containing task JSONs with calibration data.
        model: Model name to read calibration results for.
        output_dir: Where to write the sampled_tasks directory.
        fail_count: Number of failed tasks to include.
        pass_count: Number of passed tasks to include.

    Returns:
        Path to the created sampled_tasks directory.
    """
    sampled_dir = Path(output_dir)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = Path(tasks_dir)
    failed: List[Tuple[Path, dict, float]] = []  # (path, task_data, percent_complete)
    passed: List[Tuple[Path, dict, float]] = []

    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            cal = find_calibration_entry(task_data.get("calibration", []), model=model)
            if cal is None:
                continue
            pct = cal_progress(cal)
            if cal_passed(cal):
                passed.append((task_file, task_data, pct))
            else:
                failed.append((task_file, task_data, pct))
        except Exception:
            continue

    # Prioritize partial completions — they reveal *why* the agent got stuck
    failed.sort(key=lambda x: x[2], reverse=True)

    selected_failed = failed[:fail_count]
    selected_passed = random.sample(passed, min(pass_count, len(passed))) if passed else []

    for i, (_, task_data, pct) in enumerate(selected_failed, 1):
        annotated = dict(task_data)
        cal = find_calibration_entry(task_data.get("calibration", []), model=model)
        annotated["_benchmark_result"] = {
            "model": model,
            "outcome": "FAILED",
            "percent_complete": pct,
        }
        pct_int = int(pct * 100)
        filename = f"failed_{i}_{pct_int}pct.json"
        with open(sampled_dir / filename, "w") as f:
            json.dump(annotated, f, indent=2)

    for i, (_, task_data, pct) in enumerate(selected_passed, 1):
        annotated = dict(task_data)
        annotated["_benchmark_result"] = {
            "model": model,
            "outcome": "PASSED",
            "percent_complete": pct,
        }
        filename = f"passed_{i}.json"
        with open(sampled_dir / filename, "w") as f:
            json.dump(annotated, f, indent=2)

    total = len(list(sampled_dir.glob("*.json")))
    print(f"[evolve] Prepared sampled_tasks: {total} files ({len(selected_failed)} failed, {len(selected_passed)} passed)")
    return sampled_dir


def compute_pass_rate_from_calibration(tasks_dir: str, model: str) -> Dict[str, float]:
    """Compute pass rate for a model from calibration data in task JSONs.

    Returns:
        Dict with keys: passed, failed, untested, total, pass_rate (as percentage 0-100).
    """
    tasks_path = Path(tasks_dir)
    passed = 0
    failed = 0
    untested = 0

    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            cal = find_calibration_entry(task_data.get("calibration", []), model=model)
            if cal is None:
                untested += 1
            elif cal_passed(cal):
                passed += 1
            else:
                failed += 1
        except Exception:
            continue

    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0.0
    return {
        "passed": passed,
        "failed": failed,
        "untested": untested,
        "total": total,
        "pass_rate": pass_rate,
    }


def find_tasks_without_calibration(tasks_dir: str, model: str) -> List[Path]:
    """Find task JSONs that are missing calibration data for a given model."""
    tasks_path = Path(tasks_dir)
    missing = []

    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            cal = find_calibration_entry(task_data.get("calibration", []), model=model)
            if cal is None:
                missing.append(task_file)
        except Exception:
            continue

    return missing


def build_evolution_query(
    pass_rate: float,
    tier_model: str,
    generation_idx: int,
) -> str:
    """Build the --query text for the next generation round.

    Tells the agent what the _benchmark_result annotations mean and
    guides it to produce harder tasks.

    Args:
        pass_rate: Current pass rate as a percentage (0-100).
        tier_model: Name of the model being benchmarked.
        generation_idx: Which generation tier (1-based).
    """
    rate = pass_rate

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
