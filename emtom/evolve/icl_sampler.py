"""Prepare annotated sampled_tasks directories for evolutionary generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from emtom.evolve.benchmark_wrapper import find_calibration_entry, cal_passed, cal_progress


def _has_problem_pddl(task_data: dict) -> bool:
    """Return True if the task has a non-empty problem_pddl field (not a legacy task)."""
    pddl = task_data.get("problem_pddl", "")
    return bool(pddl and isinstance(pddl, str) and pddl.strip())


def _diverse_sample(
    items: List[Tuple[Path, dict, float]],
    count: int,
    model: str,
) -> List[Tuple[Path, dict, float]]:
    """Select a diverse subset, preferring tom_level >= 2 and multi-agent tasks.

    Prioritizes higher tom_level, then stratifies by (category, tom_level)
    to ensure diversity across both dimensions.
    """
    if len(items) <= count:
        return items

    # Prefer tom_level >= 2, multi-agent, with mechanics
    preferred = [
        it for it in items
        if it[1].get("tom_level", 0) >= 2
        and it[1].get("num_agents", 0) >= 2
    ]
    fallback = [it for it in items if it not in preferred]

    # Build strata by (category, tom_level) from preferred pool first
    pool = preferred if len(preferred) >= count else items
    strata: Dict[tuple, list] = {}
    for item in pool:
        _, data, _ = item
        key = (data.get("category", "?"), data.get("tom_level", 0))
        strata.setdefault(key, []).append(item)

    # Round-robin from each stratum
    selected: List[Tuple[Path, dict, float]] = []
    seen: set = set()
    keys = list(strata.keys())
    random.shuffle(keys)
    idx = 0
    while len(selected) < count and idx < count * len(keys):
        key = keys[idx % len(keys)]
        bucket = strata[key]
        if bucket:
            item = bucket.pop(random.randrange(len(bucket)))
            item_id = id(item)
            if item_id not in seen:
                selected.append(item)
                seen.add(item_id)
        idx += 1

    # Fill remaining from fallback if needed
    if len(selected) < count:
        remaining = [it for it in items if id(it) not in seen]
        if remaining:
            extra = random.sample(remaining, min(count - len(selected), len(remaining)))
            selected.extend(extra)

    return selected


def _build_benchmark_annotation(
    task_data: dict,
    model: str,
    outcome: str,
    pct: float,
) -> dict:
    """Build _benchmark_result with trajectory from the target model only.

    Strips calibration data from other models so the generator only sees
    the target model's behavior.
    """
    cal = find_calibration_entry(task_data.get("calibration", []), model=model)
    annotation: dict = {
        "model": model,
        "outcome": outcome,
        "percent_complete": pct,
    }
    if cal:
        annotation["steps"] = cal.get("steps", 0)
        annotation["results"] = cal.get("results", {})
        trajectory = cal.get("trajectory", [])
        if trajectory:
            annotation["trajectory"] = trajectory
    return annotation


def prepare_sampled_tasks_dir_from_calibration(
    tasks_dir: str,
    model: str,
    output_dir: str,
    fail_count: int = 8,
    pass_count: int = 2,
) -> Path:
    """Create annotated sampled_tasks directory from calibration data in task JSONs.

    Reads calibration[model] from each task JSON to determine pass/fail status.
    Selects a diverse mix of failed and passed tasks, annotates each with a
    _benchmark_result field containing the full agent trajectory for the
    target model only. Strips calibration data from other models.

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
    failed: List[Tuple[Path, dict, float]] = []
    passed: List[Tuple[Path, dict, float]] = []

    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            if not _has_problem_pddl(task_data):
                continue
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

    selected_failed = _diverse_sample(failed, fail_count, model)
    selected_passed = _diverse_sample(passed, pass_count, model)

    for i, (_, task_data, pct) in enumerate(selected_failed, 1):
        annotated = dict(task_data)
        # Strip all calibration data — only the target model's result matters
        annotated.pop("calibration", None)
        annotated["_benchmark_result"] = _build_benchmark_annotation(
            task_data, model, "FAILED", pct,
        )
        pct_int = int(pct * 100)
        filename = f"failed_{i}_{pct_int}pct.json"
        with open(sampled_dir / filename, "w") as f:
            json.dump(annotated, f, indent=2)

    for i, (_, task_data, pct) in enumerate(selected_passed, 1):
        annotated = dict(task_data)
        annotated.pop("calibration", None)
        annotated["_benchmark_result"] = _build_benchmark_annotation(
            task_data, model, "PASSED", pct,
        )
        filename = f"passed_{i}.json"
        with open(sampled_dir / filename, "w") as f:
            json.dump(annotated, f, indent=2)

    total = len(list(sampled_dir.glob("*.json")))
    f_cats = [d.get("category", "?") for _, d, _ in selected_failed]
    f_toms = [d.get("tom_level", 0) for _, d, _ in selected_failed]
    print(
        f"[evolve] Prepared sampled_tasks: {total} files "
        f"({len(selected_failed)} failed, {len(selected_passed)} passed)"
    )
    print(f"[evolve]   Failed categories: {dict(zip(f_cats, f_cats))!r} → {f_cats}")
    print(f"[evolve]   Failed tom_levels: {f_toms}")
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
            if not _has_problem_pddl(task_data):
                continue  # skip legacy tasks without problem_pddl
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
            if not _has_problem_pddl(task_data):
                continue  # skip legacy tasks without problem_pddl
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
    guides it to analyze agent trajectories before authoring.

    Args:
        pass_rate: Current pass rate as a percentage (0-100).
        tier_model: Name of the model being benchmarked.
        generation_idx: Which generation tier (1-based).
    """
    rate = pass_rate

    return (
        f"## Calibrated Task Review\n"
        f"The `sampled_tasks/` directory contains 10 calibrated tasks for {tier_model} "
        f"standard mode: 2 passes and 8 fails from the same difficulty regime.\n"
        f"Files named `failed_*` = tasks {tier_model} COULD NOT solve.\n"
        f"Files named `passed_*` = tasks {tier_model} COULD solve.\n"
        f"Current pass rate for {tier_model}: {rate:.1f}%\n\n"
        f"Each task JSON contains the task spec plus a `_benchmark_result` field with "
        f"the {tier_model} calibration: `outcome`, `percent_complete`, `steps`, "
        f"`results` (per-agent breakdown), and `trajectory` (the full rollout — "
        f"action/observation/communication per turn).\n\n"
        f"For each task, inspect: `task`, `title`, `category`, `tom_level`, "
        f"`num_agents`, `active_mechanics`, `mechanic_bindings`, `agent_secrets`, "
        f"`problem_pddl`, and the `_benchmark_result` trajectory.\n\n"
        f"## Analysis (do this BEFORE authoring)\n"
        f"For each task, reason about:\n"
        f"- **Shared-plan clarity**: Can agents decompose the goal into roles early?\n"
        f"- **Information topology**: Who knows what? Are hidden facts decisive or decorative?\n"
        f"- **Communication topology**: Who can message whom, how many times, and is it enough?\n"
        f"- **Epistemic dependency**: Does completing a physical goal require knowing something "
        f"only another agent can observe?\n"
        f"- **Search burden**: Must agents search broadly, or is the search space narrow?\n"
        f"- **Conflict structure**: Do agents have misaligned private incentives?\n"
        f"- **Recovery slack**: If an agent makes a wrong move, can they recover?\n"
        f"- **Mechanic coupling**: Are mechanics load-bearing or cosmetic?\n"
        f"- **Goal load**: Number of conjuncts — are they related or just piled on?\n"
        f"- **Endgame fragility**: Does the task hinge on a final confirmation chain?\n\n"
        f"Then compare passes vs fails: what made the passed tasks hard but still legible, "
        f"and what made the failed tasks fail due to genuine Theory-of-Mind pressure versus "
        f"clutter, search, or over-coupling?\n\n"
        f"Write a brief analysis before designing your task.\n\n"
        f"## Design Principles for New Tasks\n"
        f"Make the core difficulty a **belief-routing problem**, not a search problem:\n"
        f"- Use one or two decisive hidden facts, not a pile of decorative secrets.\n"
        f"- Constrained but meaningful communication — messages must carry load.\n"
        f"- Asymmetric room/action access that forces genuine delegation.\n"
        f"- Precise confirmation chains (K() goals) that require agents to reason about "
        f"what others have observed.\n"
        f"- Keep shared goals crisp enough that a sensible role decomposition is possible early.\n\n"
        f"**Avoid** making tasks hard by:\n"
        f"- Piling on unrelated subtasks (goal load without epistemic depth).\n"
        f"- Broad object/room search (search burden without ToM).\n"
        f"- Too many overlapping private conflicts (clutter, not genuine tension).\n\n"
        f"## Diversity Requirement\n"
        f"Your task MUST differ from the sampled tasks in at least TWO of:\n"
        f"- Difficulty mechanism (different failure mode than the examples)\n"
        f"- Scenario theme (not another staging/inspection/cleanup/walkthrough task)\n"
        f"- ToM structure (different K-level or nesting pattern)\n"
        f"- Mechanic combination (try different mechanic stacks)\n"
    )
