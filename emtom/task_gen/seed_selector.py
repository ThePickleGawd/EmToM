"""Seed-task selection for calibrated task generation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from emtom.evolve.benchmark_wrapper import cal_passed, cal_progress, find_calibration_entry


@dataclass(frozen=True)
class SeedSelectionConfig:
    """Configuration for selecting tasks from a seed pool."""

    tasks_dir: Path
    target_model: str
    target_pass_rate: float = 0.20
    current_pass_rate: Optional[float] = None
    category: Optional[str] = None
    tom_level: Optional[int] = None


@dataclass(frozen=True)
class SeedTaskCandidate:
    """A task candidate with enough metadata for weighted selection."""

    path: Path
    task_data: dict
    weight: float
    passed_target_model: Optional[bool]
    progress: Optional[float]


def is_task_like_json(path: Path) -> bool:
    """Return True when a JSON file looks like an EMTOM task spec."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return all(k in data for k in ("title", "task", "agent_actions"))


def resolve_seed_tasks_dir(explicit_dir: Optional[str], output_dir: str) -> Optional[Path]:
    """Pick the first non-empty task directory in priority order."""
    candidate_dirs = []
    if explicit_dir:
        candidate_dirs.append(Path(explicit_dir))
    candidate_dirs.extend(
        [
            Path(output_dir),
            Path("data/emtom/tasks"),
            Path("data/emtom/very_old_tasks/old_calibration_format_2_25_26"),
        ]
    )

    seen = set()
    for candidate in candidate_dirs:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists() or not candidate.is_dir():
            continue
        if any(is_task_like_json(path) for path in candidate.glob("*.json")):
            return candidate
    return None


def build_seed_candidates(config: SeedSelectionConfig) -> list[SeedTaskCandidate]:
    """Load and score all compatible task seeds from the pool."""
    if not config.tasks_dir.exists():
        return []

    candidates: list[SeedTaskCandidate] = []
    for path in sorted(config.tasks_dir.glob("*.json")):
        try:
            with open(path) as f:
                task_data = json.load(f)
        except Exception:
            continue
        if not isinstance(task_data, dict):
            continue
        if not all(k in task_data for k in ("title", "task", "agent_actions")):
            continue

        weight = _candidate_weight(task_data, config)
        if weight <= 0:
            continue

        cal = find_calibration_entry(task_data.get("calibration", []), model=config.target_model)
        passed = cal_passed(cal) if cal is not None else None
        progress = cal_progress(cal) if cal is not None else None
        candidates.append(
            SeedTaskCandidate(
                path=path,
                task_data=task_data,
                weight=weight,
                passed_target_model=passed,
                progress=progress,
            )
        )
    return candidates


def select_seed_tasks(
    config: SeedSelectionConfig,
    count: int,
    rng: Optional[random.Random] = None,
) -> list[SeedTaskCandidate]:
    """Sample seed tasks without replacement using the calibrated selector."""
    if count <= 0:
        return []

    rng = rng or random
    pool = build_seed_candidates(config)
    if not pool:
        return []

    chosen: list[SeedTaskCandidate] = []
    remaining = list(pool)
    target_count = min(count, len(remaining))
    while remaining and len(chosen) < target_count:
        weights = [max(candidate.weight, 0.001) for candidate in remaining]
        picked = rng.choices(remaining, weights=weights, k=1)[0]
        chosen.append(picked)
        remaining = [candidate for candidate in remaining if candidate.path != picked.path]
    return chosen


def _candidate_weight(task_data: dict, config: SeedSelectionConfig) -> float:
    """Weight a seed so the pool shifts toward the target pass-rate."""
    cal = find_calibration_entry(task_data.get("calibration", []), model=config.target_model)
    passed = cal_passed(cal) if cal is not None else None
    progress = cal_progress(cal) if cal is not None else None

    hard_bias = _hard_bias(config.current_pass_rate, config.target_pass_rate)
    hard_score, easy_score = _difficulty_scores(passed, progress)
    weight = hard_bias * hard_score + (1.0 - hard_bias) * easy_score

    task_category = task_data.get("category")
    if config.category:
        if task_category == config.category:
            weight *= 1.35
        elif task_category:
            weight *= 0.75

    if config.tom_level is not None:
        stored_level = task_data.get("tom_level")
        if stored_level == config.tom_level:
            weight *= 1.35
        elif isinstance(stored_level, int):
            weight *= 0.70

    return max(weight, 0.0)


def _hard_bias(current_rate: Optional[float], target_rate: float) -> float:
    """Return how aggressively to favor harder seeds."""
    if current_rate is None:
        return 0.75
    return min(max(0.5 + 2.0 * (current_rate - target_rate), 0.20), 0.95)


def _difficulty_scores(
    passed: Optional[bool],
    progress: Optional[float],
) -> tuple[float, float]:
    """Score how useful a task is as a hard or easy seed."""
    progress = min(max(progress or 0.0, 0.0), 1.0)

    if passed is True:
        return 0.12, 1.00
    if passed is False:
        hard_score = 1.20 - 0.55 * progress
        easy_score = 0.20 + 0.45 * progress
        return hard_score, easy_score

    # Untested tasks remain useful for exploration, but less than calibrated ones.
    return 0.75, 0.55
