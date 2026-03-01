"""Main evolutionary difficulty orchestrator.

Usage:
    python -m emtom.evolve.orchestrator [options]

Or via shell script:
    ./emtom/run_emtom.sh evolve [options]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from emtom.evolve.config import (
    EvolutionConfig,
    DEFAULT_MODEL_LADDER,
    DEFAULT_EVOLVE_FOCUS,
    DEFAULT_EVOLVE_CATEGORY,
    DEFAULT_EVOLVE_TOM_TARGET_L1,
    DEFAULT_EVOLVE_TOM_TARGET_L2,
    DEFAULT_EVOLVE_TOM_TARGET_L3,
    DEFAULT_EVOLVE_TOM_TOLERANCE,
)
from emtom.evolve.benchmark_wrapper import (
    run_benchmark_parallel,
    BenchmarkResults,
    update_calibration_from_benchmark,
)
from emtom.evolve.icl_sampler import (
    prepare_sampled_tasks_dir_from_calibration,
    compute_pass_rate_from_calibration,
    find_tasks_without_calibration,
    build_evolution_query,
)


# Subtask complexity ranges per tier position.
# tier 0 (seed) = simple tasks, ramping up to full complexity.
TIER_SUBTASK_RANGES = [
    (2, 4),   # tier 0: seed — very simple
    (2, 5),   # tier 1: easiest model
    (3, 7),   # tier 2
    (3, 10),  # tier 3
    (4, 12),  # tier 4
    (5, 15),  # tier 5
    (5, 20),  # tier 6+: hardest models — full range
]

VALID_TASK_CATEGORIES = ("cooperative", "competitive", "mixed")


def resolve_categories(category_spec: str) -> List[str]:
    """Normalize category selector into one or more concrete categories."""
    if not category_spec:
        return list(VALID_TASK_CATEGORIES)

    raw = category_spec.strip().lower()
    if raw in {"any", "all", "*"}:
        return list(VALID_TASK_CATEGORIES)

    categories = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if not categories:
        return list(VALID_TASK_CATEGORIES)

    invalid = [c for c in categories if c not in VALID_TASK_CATEGORIES]
    if invalid:
        valid = ", ".join(VALID_TASK_CATEGORIES)
        bad = ", ".join(invalid)
        raise ValueError(f"Invalid category value(s): {bad}. Valid: {valid}, any, all")

    # Deduplicate while preserving user order.
    seen = set()
    ordered: List[str] = []
    for c in categories:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def get_difficulty_for_tier(tier_idx: int, total_tiers: int) -> str:
    """Map tier index to difficulty label for the judge.

    Tier 0 (seed) is hardcoded as "easy" by the caller.
    Post-benchmark tiers (1+) escalate: medium → hard.
    """
    if total_tiers <= 1:
        return "medium"
    ratio = tier_idx / total_tiers  # 0.0 = seed, 1.0 = hardest
    if ratio < 0.5:
        return "medium"
    else:
        return "hard"


def get_subtask_range(tier_idx: int) -> tuple:
    """Return (subtasks_min, subtasks_max) for a given tier index."""
    idx = min(tier_idx, len(TIER_SUBTASK_RANGES) - 1)
    return TIER_SUBTASK_RANGES[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel task-upgrade loop seeded from existing tasks. "
            "Default mode ('either') increases difficulty and pushes toward higher ToM."
        )
    )
    parser.add_argument(
        "--model-ladder", type=str,
        default=",".join(DEFAULT_MODEL_LADDER),
        help="Comma-separated model ladder (default: %(default)s)",
    )
    parser.add_argument("--generator-model", type=str, default="gpt-5.2")
    parser.add_argument("--tasks-per-round", type=int, default=20)
    parser.add_argument("--seed-pool-size", type=int, default=30,
                        help="Minimum seed pool size — generate extra if copied tasks < this (default: 30)")
    parser.add_argument("--seed-tasks-dir", type=str, default="data/emtom/tasks",
                        help="Source directory for seed tasks (default: data/emtom/tasks)")
    parser.add_argument("--icl-total-examples", type=int, default=10)
    parser.add_argument("--icl-failure-ratio", type=float, default=0.9)
    parser.add_argument("--judge-threshold", type=float, default=0.7,
                        help="Judge threshold for generation quality gate (default: 0.7)")
    parser.add_argument(
        "--target-pass-rate",
        type=float,
        default=20.0,
        help="Target pass rate percent — generate until pass rate drops to this (default: 20.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/emtom/tasks",
        help="Directory where evolved task JSONs are written (default: data/emtom/tasks)",
    )
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Max parallel generation/benchmark processes (default: 50)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing output directory")
    parser.add_argument(
        "--focus",
        type=str,
        choices=["difficulty", "tom", "either"],
        default=DEFAULT_EVOLVE_FOCUS,
        help=(
            "Upgrade objective: difficulty (lower pass rate), tom (higher-order ToM), "
            "or either (default)"
        ),
    )
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_EVOLVE_CATEGORY,
        help=(
            "Generated task categories: single value, comma list, or any/all. "
            "Examples: cooperative | cooperative,mixed | all"
        ),
    )
    parser.add_argument(
        "--tom-target-l1",
        type=float,
        default=DEFAULT_EVOLVE_TOM_TARGET_L1,
        help=f"Target ratio for ToM level 1 (default: {DEFAULT_EVOLVE_TOM_TARGET_L1})",
    )
    parser.add_argument(
        "--tom-target-l2",
        type=float,
        default=DEFAULT_EVOLVE_TOM_TARGET_L2,
        help=f"Target ratio for ToM level 2 (default: {DEFAULT_EVOLVE_TOM_TARGET_L2})",
    )
    parser.add_argument(
        "--tom-target-l3",
        type=float,
        default=DEFAULT_EVOLVE_TOM_TARGET_L3,
        help=f"Target ratio for ToM level 3 (default: {DEFAULT_EVOLVE_TOM_TARGET_L3})",
    )
    parser.add_argument(
        "--tom-ratio-tolerance",
        type=float,
        default=DEFAULT_EVOLVE_TOM_TOLERANCE,
        help=f"ToM ratio tolerance (default: {DEFAULT_EVOLVE_TOM_TOLERANCE})",
    )
    return parser.parse_args()


def load_state(output_dir: Path) -> dict:
    """Load checkpoint state from state.json."""
    state_file = output_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"completed_tiers": [], "current_tier_idx": 0}


def save_state(output_dir: Path, state: dict) -> None:
    """Save checkpoint state to state.json."""
    with open(output_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)


def run_generate(
    model: str,
    num_tasks: int,
    output_dir: str,
    test_model: Optional[str] = None,
    query: Optional[str] = None,
    category: str = "any",
    sampled_tasks_dir: Optional[str] = None,
    judge_threshold: Optional[float] = None,
    subtasks_min: Optional[int] = None,
    subtasks_max: Optional[int] = None,
    difficulty: Optional[str] = None,
    tom_target_l1: Optional[float] = None,
    tom_target_l2: Optional[float] = None,
    tom_target_l3: Optional[float] = None,
    tom_ratio_tolerance: Optional[float] = None,
) -> Path:
    """Run task generation via run_emtom.sh generate.

    Returns:
        Path to the output directory containing generated tasks.
    """
    categories = resolve_categories(category)
    selected_category = categories[0]
    if len(categories) > 1:
        print(
            "[evolve] run_generate received multiple categories; "
            f"using first category only: {selected_category}"
        )

    cmd = [
        "./emtom/run_emtom.sh", "generate",
        "--model", model,
        "--num-tasks", str(num_tasks),
        "--category", selected_category,
        "--output-dir", output_dir,
    ]
    if test_model:
        cmd.extend(["--test-model", test_model])
    if query:
        cmd.extend(["--query", query])
    if sampled_tasks_dir:
        cmd.extend(["--sampled-tasks-dir", sampled_tasks_dir])
    if judge_threshold is not None:
        cmd.extend(["--threshold", str(judge_threshold)])
    if subtasks_min is not None:
        cmd.extend(["--subtasks-min", str(subtasks_min)])
    if subtasks_max is not None:
        cmd.extend(["--subtasks-max", str(subtasks_max)])
    if difficulty:
        cmd.extend(["--difficulty", difficulty])
    if tom_target_l1 is not None:
        cmd.extend(["--tom-target-l1", str(tom_target_l1)])
    if tom_target_l2 is not None:
        cmd.extend(["--tom-target-l2", str(tom_target_l2)])
    if tom_target_l3 is not None:
        cmd.extend(["--tom-target-l3", str(tom_target_l3)])
    if tom_ratio_tolerance is not None:
        cmd.extend(["--tom-ratio-tolerance", str(tom_ratio_tolerance)])

    print(f"[evolve] Running generation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[evolve] WARNING: generation exited with code {result.returncode}", file=sys.stderr)

    return Path(output_dir)


def run_generate_parallel(
    model: str,
    num_tasks: int,
    output_dir: str,
    worker_log_dir: Optional[str] = None,
    max_workers: int = 50,
    test_model: Optional[str] = None,
    query: Optional[str] = None,
    category: str = "any",
    sampled_tasks_dir: Optional[str] = None,
    judge_threshold: Optional[float] = None,
    subtasks_min: Optional[int] = None,
    subtasks_max: Optional[int] = None,
    difficulty: Optional[str] = None,
    tom_target_l1: Optional[float] = None,
    tom_target_l2: Optional[float] = None,
    tom_target_l3: Optional[float] = None,
    tom_ratio_tolerance: Optional[float] = None,
) -> Path:
    """Run task generation in parallel via N independent processes.

    Spawns up to max_workers concurrent `run_emtom.sh generate --num-tasks 1`
    processes, all writing to the same output_dir. Stops once num_tasks JSON
    files exist. Caps total spawns at num_tasks * 3 to avoid infinite retries.

    Returns:
        Path to output_dir containing generated tasks.

    Args:
        worker_log_dir: Optional directory for per-worker stdout/stderr logs.
            Defaults to <output_dir>/logs when not provided.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(worker_log_dir) if worker_log_dir else (out / "logs")
    log_dir.mkdir(exist_ok=True)
    categories = resolve_categories(category)

    # Build base command (each process generates 1 task)
    base_cmd = [
        "./emtom/run_emtom.sh", "generate",
        "--model", model,
        "--num-tasks", "1",
        "--output-dir", output_dir,
    ]
    if test_model:
        base_cmd.extend(["--test-model", test_model])
    if query:
        base_cmd.extend(["--query", query])
    if sampled_tasks_dir:
        base_cmd.extend(["--sampled-tasks-dir", sampled_tasks_dir])
    if judge_threshold is not None:
        base_cmd.extend(["--threshold", str(judge_threshold)])
    if subtasks_min is not None:
        base_cmd.extend(["--subtasks-min", str(subtasks_min)])
    if subtasks_max is not None:
        base_cmd.extend(["--subtasks-max", str(subtasks_max)])
    if difficulty:
        base_cmd.extend(["--difficulty", difficulty])
    if tom_target_l1 is not None:
        base_cmd.extend(["--tom-target-l1", str(tom_target_l1)])
    if tom_target_l2 is not None:
        base_cmd.extend(["--tom-target-l2", str(tom_target_l2)])
    if tom_target_l3 is not None:
        base_cmd.extend(["--tom-target-l3", str(tom_target_l3)])
    if tom_ratio_tolerance is not None:
        base_cmd.extend(["--tom-ratio-tolerance", str(tom_ratio_tolerance)])

    max_spawns = num_tasks * 5
    total_spawned = 0
    active: List[tuple] = []  # (Popen, log_file_handle)

    def _count_valid_tasks() -> int:
        # Ignore partial/corrupt JSON files that may still be in-flight writes.
        valid = 0
        for jf in out.glob("*.json"):
            try:
                with open(jf) as f:
                    json.load(f)
                valid += 1
            except Exception:
                continue
        return valid

    # Count existing tasks so we generate num_tasks NEW ones (not total)
    baseline_count = _count_valid_tasks()
    target_count = baseline_count + num_tasks

    # Use unique log prefix to avoid overwriting logs from previous phases
    existing_logs = len(list(log_dir.glob("gen_*.log")))

    spinner_chars = ["|", "/", "-", "\\"]
    spinner_idx = 0
    last_status = None

    print(
        f"[evolve] Parallel generation: target={num_tasks} new tasks "
        f"(baseline={baseline_count}), max_workers={max_workers}, "
        f"categories={','.join(categories)}"
    )
    print(f"[evolve] Output tasks dir: {out.resolve()}")
    print(f"[evolve] Generation logs: {log_dir.resolve()}")

    try:
        while True:
            # Reap finished processes first.
            still_active = []
            for proc, fh in active:
                if proc.poll() is not None:
                    fh.close()
                else:
                    still_active.append((proc, fh))
            active = still_active

            done_count = _count_valid_tasks()
            new_count = done_count - baseline_count

            # Once target is reached, do not kill workers mid-write.
            # Stop spawning and wait for active workers to finish naturally.
            if done_count >= target_count:
                if not active:
                    if sys.stdout.isatty():
                        print()  # finish the in-place status line
                    print(f"[evolve] Generation complete: {new_count}/{num_tasks} new tasks (total: {done_count})")
                    print(f"[evolve] Tasks saved to: {out.resolve()}")
                    break
                spinner = spinner_chars[spinner_idx % len(spinner_chars)]
                spinner_idx += 1
                status = (
                    f"[evolve] {spinner} target reached ({new_count}/{num_tasks}); "
                    f"waiting for {len(active)} active workers to finish..."
                )
                if sys.stdout.isatty():
                    print(f"\r{status}", end="", flush=True)
                elif status != last_status:
                    print(status)
                    last_status = status
                time.sleep(5)
                continue

            # Spawn new processes — cap concurrency at tasks still needed
            needed = target_count - done_count
            concurrency_cap = min(max_workers, needed)
            while len(active) < concurrency_cap and total_spawned < max_spawns:
                # Re-check in case tasks appeared while spawning
                if _count_valid_tasks() >= target_count:
                    break
                chosen_category = categories[total_spawned % len(categories)]
                log_file = log_dir / f"gen_{existing_logs + total_spawned}_{chosen_category}.log"
                fh = open(log_file, "w")
                cmd = list(base_cmd)
                cmd.extend(["--category", chosen_category])
                proc = subprocess.Popen(cmd, stdout=fh, stderr=fh)
                active.append((proc, fh))
                total_spawned += 1

            if not active and total_spawned >= max_spawns:
                done_count = _count_valid_tasks()
                new_count = done_count - baseline_count
                if sys.stdout.isatty():
                    print()  # finish the in-place status line
                print(f"[evolve] Generation: exhausted {max_spawns} spawns, got {new_count}/{num_tasks} new tasks")
                break

            spinner = spinner_chars[spinner_idx % len(spinner_chars)]
            spinner_idx += 1
            status = (
                f"[evolve] {spinner} generating: {new_count}/{num_tasks} new tasks "
                f"(active={len(active)}, spawned={total_spawned})"
            )
            if sys.stdout.isatty():
                print(f"\r{status}", end="", flush=True)
            elif status != last_status:
                print(status)
                last_status = status
            time.sleep(10)
    except Exception:
        # On hard failures, clean up active workers.
        for proc, fh in active:
            proc.terminate()
        for proc, fh in active:
            proc.wait()
            fh.close()
        raise

    final_count = _count_valid_tasks()
    final_new = final_count - baseline_count
    if final_new < num_tasks:
        if final_new == 0:
            raise RuntimeError(
                f"Generation failed: got 0/{num_tasks} new task files in {out}"
            )
        print(
            f"[evolve] WARNING: got {final_new}/{num_tasks} new tasks "
            f"(accepting partial result)",
            file=sys.stderr,
        )

    return out


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = int(minutes // 60)
    remaining_min = int(minutes % 60)
    return f"{hours}h {remaining_min}m"


def _scan_tom_distribution(tasks_dir: str) -> Dict[int, int]:
    """Scan task JSONs for tom_level distribution. Returns {level: count}."""
    counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    tasks_path = Path(tasks_dir)
    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                data = json.load(f)
            level = data.get("tom_level")
            if isinstance(level, int) and level in counts:
                counts[level] += 1
        except Exception:
            continue
    return counts


def print_summary_table(
    pipeline_start: float,
    seed_duration: Optional[float],
    seed_tasks_generated: int,
    tier_metrics: List[dict],
    all_tasks_dir: str,
    config: EvolutionConfig,
) -> None:
    """Print a formatted terminal summary table for the evolution run."""
    total_duration = time.time() - pipeline_start

    # ── Header ──
    print()
    print("=" * 80)
    print("  EVOLUTION PIPELINE SUMMARY")
    print("=" * 80)
    print()

    # ── Config ──
    print(f"  Generator model : {config.generator_model}")
    print(f"  Model ladder    : {' -> '.join(config.model_ladder)}")
    print(f"  Focus           : {config.focus}")
    print(f"  Category        : {config.category}")
    print(f"  Target pass rate: {config.target_pass_rate:.0f}%")
    print(f"  Total duration  : {_fmt_duration(total_duration)}")
    print()

    # ── Seed phase ──
    if seed_duration is not None:
        print(f"  Seed phase: {seed_tasks_generated} tasks generated in {_fmt_duration(seed_duration)}")
    else:
        print(f"  Seed phase: skipped (resumed)")
    print()

    # ── Tier table ──
    #   Tier | Model | Bench Time | Gen Time | +Tasks | Pass Rate (before->after) | Difficulty
    header = (
        f"  {'Tier':<6} {'Model':<14} {'Bench':>8} {'Gen':>8} "
        f"{'New':>5} {'Pass Rate':>18} {'Difficulty':<10}"
    )
    separator = "  " + "-" * (len(header) - 2)
    print(header)
    print(separator)

    for tm in tier_metrics:
        tier_label = f"T{tm['tier_idx']}"
        model = tm["model"]
        bench_time = _fmt_duration(tm["benchmark_seconds"]) if tm["benchmark_seconds"] > 0 else "-"
        gen_time = _fmt_duration(tm["generation_seconds"]) if tm["generation_seconds"] > 0 else "-"
        new_tasks = str(tm["generated"]) if tm["generated"] > 0 else "-"
        pr_before = tm["pass_rate_before"]
        pr_after = tm["pass_rate_after"]
        pass_col = f"{pr_before:.1f}% -> {pr_after:.1f}%"
        difficulty = tm["difficulty"]
        print(
            f"  {tier_label:<6} {model:<14} {bench_time:>8} {gen_time:>8} "
            f"{new_tasks:>5} {pass_col:>18} {difficulty:<10}"
        )

    print(separator)
    print()

    # ── ToM distribution ──
    tom = _scan_tom_distribution(all_tasks_dir)
    tom_total = sum(tom.values())
    print("  ToM Distribution:")
    if tom_total > 0:
        for level in (1, 2, 3):
            count = tom[level]
            pct = count / tom_total * 100
            bar = "#" * int(pct / 2)
            print(f"    K({level}): {count:>4} ({pct:5.1f}%)  {bar}")
    else:
        print("    (no tom_level data in task JSONs)")
    print()

    # ── Final model pass rates ──
    total_tasks = len(list(Path(all_tasks_dir).glob("*.json")))
    print(f"  Total tasks in pool: {total_tasks}")
    print()
    pr_header = f"  {'Model':<14} {'Passed':>8} {'Failed':>8} {'Untested':>10} {'Pass Rate':>10}"
    print(pr_header)
    print("  " + "-" * (len(pr_header) - 2))
    for model in config.model_ladder:
        stats = compute_pass_rate_from_calibration(all_tasks_dir, model)
        print(
            f"  {model:<14} {stats['passed']:>8} {stats['failed']:>8} "
            f"{stats['untested']:>10} {stats['pass_rate']:>9.1f}%"
        )
    print()

    print(f"  Tasks directory : {Path(all_tasks_dir).resolve()}")
    print("=" * 80)
    print()


def run_evolution(config: EvolutionConfig, resume_dir: Optional[str] = None) -> None:
    """Main evolutionary loop.

    Design: Stay-in-tier with accumulated task pool.
    - Single tasks/ directory accumulates all tasks across tiers.
    - test_task calibration data IS the benchmark (no separate benchmark for new tasks).
    - On model upgrade: benchmark only tasks missing calibration for the new model.
    - Stay in tier generating until pass rate drops to target, then advance.
    """
    pipeline_start = time.time()

    # Setup run metadata directory and task output directory.
    if resume_dir:
        run_dir = Path(resume_dir)
        print(f"[evolve] Resuming from {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("outputs/evolve") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Legacy resume compatibility: old runs used run_dir/tasks as task pool.
    if resume_dir and (run_dir / "tasks").exists():
        all_tasks_dir = run_dir / "tasks"
    else:
        all_tasks_dir = Path(config.output_dir)
    all_tasks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evolve] Run metadata directory: {run_dir.resolve()}")
    print(f"[evolve] Task output directory: {all_tasks_dir.resolve()}")
    print(
        "[evolve] Defaults: "
        f"focus={config.focus}, category={config.category}, "
        f"tom_target=({config.tom_target_l1:.0%}, {config.tom_target_l2:.0%}, {config.tom_target_l3:.0%}), "
        f"workers={config.max_workers}"
    )

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Load state for resumption
    state = load_state(run_dir)
    completed_tiers = set(state.get("completed_tiers", []))
    start_tier_idx = state.get("current_tier_idx", 0)

    # Tracking for summary table
    seed_duration: Optional[float] = None
    seed_tasks_generated = 0
    tier_metrics_list: List[dict] = []

    # ---- PHASE 1: Seed from existing tasks ----
    seed_tier = "seed"
    if seed_tier not in completed_tiers:
        seed_start = time.time()
        # Copy existing tasks from seed directory
        copied = 0
        skipped_legacy = 0
        if config.seed_tasks_dir:
            source = Path(config.seed_tasks_dir)
            if source.exists():
                for task_file in source.glob("*.json"):
                    dest = all_tasks_dir / task_file.name
                    if not dest.exists():
                        # Skip legacy tasks without problem_pddl
                        try:
                            with open(task_file) as f:
                                td = json.load(f)
                            pddl = td.get("problem_pddl", "")
                            if not (pddl and isinstance(pddl, str) and pddl.strip()):
                                skipped_legacy += 1
                                continue
                        except Exception:
                            continue
                        shutil.copy2(task_file, dest)
                        copied += 1
                print(f"[evolve] Seeded {copied} tasks from {source}")
                if skipped_legacy:
                    print(f"[evolve] Skipped {skipped_legacy} legacy tasks (no problem_pddl)")
            else:
                print(f"[evolve] Seed tasks dir does not exist: {source}")

        # Generate more if we don't have enough
        existing = len(list(all_tasks_dir.glob("*.json")))
        shortfall = config.seed_pool_size - existing
        if shortfall > 0:
            print(f"\n{'='*60}")
            print(f"SEED: Generating {shortfall} easy tasks (have {existing}, need {config.seed_pool_size})")
            print(f"  test_model: {config.model_ladder[0]}")
            print(f"{'='*60}\n")

            seed_sub_min, seed_sub_max = get_subtask_range(0)
            run_generate_parallel(
                model=config.generator_model,
                num_tasks=shortfall,
                output_dir=str(all_tasks_dir),
                worker_log_dir=str(run_dir / "seed_generation_logs"),
                max_workers=config.max_workers,
                test_model=config.model_ladder[0],
                category=config.category,
                judge_threshold=config.judge_threshold,
                subtasks_min=seed_sub_min,
                subtasks_max=seed_sub_max,
                difficulty="easy",
                tom_target_l1=config.tom_target_l1,
                tom_target_l2=config.tom_target_l2,
                tom_target_l3=config.tom_target_l3,
                tom_ratio_tolerance=config.tom_ratio_tolerance,
            )
        else:
            print(f"[evolve] Seed pool sufficient ({existing} tasks >= {config.seed_pool_size})")

        seed_duration = time.time() - seed_start
        seed_tasks_generated = max(0, shortfall) if shortfall > 0 else 0

        completed_tiers.add(seed_tier)
        state["completed_tiers"] = list(completed_tiers)
        state["current_tier_idx"] = 0
        save_state(run_dir, state)
    else:
        print(f"[evolve] Skipping seed phase (already completed)")

    # ---- PHASE 2: Tier loop ----
    for tier_idx, model in enumerate(config.model_ladder):
        tier_name = f"tier_{tier_idx + 1}_{model}"

        if tier_name in completed_tiers:
            print(f"[evolve] Skipping {tier_name} (already completed)")
            continue

        tier_dir = run_dir / tier_name
        tier_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"TIER {tier_idx + 1}: {model}")
        print(f"{'='*60}\n")

        tier_benchmark_seconds = 0.0
        tier_generation_seconds = 0.0

        # a. Benchmark tasks missing calibration for this model
        missing = find_tasks_without_calibration(str(all_tasks_dir), model)
        if missing:
            print(f"[evolve] {len(missing)} tasks missing calibration for {model} — benchmarking...")
            # Create temp dir with just the missing tasks for benchmarking
            missing_dir = tier_dir / "missing_tasks"
            missing_dir.mkdir(parents=True, exist_ok=True)
            for task_path in missing:
                shutil.copy2(task_path, missing_dir / task_path.name)

            benchmark_output = str(tier_dir / "benchmark")
            bench_start = time.time()
            results = run_benchmark_parallel(
                tasks_dir=str(missing_dir),
                model=model,
                output_dir=benchmark_output,
                max_workers=config.max_workers,
                no_video=True,
            )
            tier_benchmark_seconds = time.time() - bench_start

            # Write benchmark results back into the task JSONs in all_tasks_dir
            update_calibration_from_benchmark(results, str(all_tasks_dir))

            # Cleanup temp dir
            shutil.rmtree(missing_dir, ignore_errors=True)

        # b. Compute pass rate across ALL tasks
        stats = compute_pass_rate_from_calibration(str(all_tasks_dir), model)
        pass_rate = stats["pass_rate"]
        pass_rate_before = pass_rate
        print(f"[evolve] {model}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")
        print(f"[evolve]   untested: {stats['untested']}")

        # Save tier metrics
        with open(tier_dir / "tier_metrics.json", "w") as f:
            json.dump({
                "model": model,
                "phase": "initial",
                **stats,
            }, f, indent=2)

        # c. Generate until pass rate drops to target
        tier_sub_min, tier_sub_max = get_subtask_range(tier_idx + 1)
        tier_difficulty = get_difficulty_for_tier(tier_idx + 1, len(config.model_ladder) + 1)
        if config.focus == "tom":
            # ToM-focused upgrades still need sufficient complexity budget.
            tier_difficulty = "hard"
        generated_this_tier = 0

        if pass_rate > config.target_pass_rate:
            print(f"[evolve] Pass rate ({pass_rate:.1f}%) > target ({config.target_pass_rate:.1f}%)")
            print(f"[evolve] Generating harder tasks tested against {model}...")
            print(f"[evolve] Subtask range: {tier_sub_min}-{tier_sub_max}, difficulty: {tier_difficulty}")

            # Prepare ICL sampled tasks from calibration data
            fail_count = int(config.icl_total_examples * config.icl_failure_ratio)
            pass_count = config.icl_total_examples - fail_count
            sampled_dir = str(tier_dir / "sampled_tasks")
            prepare_sampled_tasks_dir_from_calibration(
                tasks_dir=str(all_tasks_dir),
                model=model,
                output_dir=sampled_dir,
                fail_count=fail_count,
                pass_count=pass_count,
            )

            # Build evolution query
            evolution_query = build_evolution_query(pass_rate, model, tier_idx + 1)
            if config.focus in {"tom", "either"}:
                evolution_query += (
                    " Prioritize higher-order ToM structures (nested epistemic goals) "
                    "that are mechanically grounded and non-trivial."
                )

            # Generate tasks in parallel batches up to max_workers,
            # re-checking pass rate after each batch.
            gen_start = time.time()
            batch_size = max(1, config.max_workers)
            while pass_rate > config.target_pass_rate and generated_this_tier < config.tasks_per_round:
                remaining = min(batch_size, config.tasks_per_round - generated_this_tier)

                run_generate_parallel(
                    model=config.generator_model,
                    num_tasks=remaining,
                    output_dir=str(all_tasks_dir),
                    worker_log_dir=str(tier_dir / "generation_logs"),
                    max_workers=config.max_workers,
                    test_model=model,  # Test against current tier's model
                    query=evolution_query,
                    category=config.category,
                    sampled_tasks_dir=sampled_dir,
                    judge_threshold=config.judge_threshold,
                    subtasks_min=tier_sub_min,
                    subtasks_max=tier_sub_max,
                    difficulty=tier_difficulty,
                    tom_target_l1=config.tom_target_l1,
                    tom_target_l2=config.tom_target_l2,
                    tom_target_l3=config.tom_target_l3,
                    tom_ratio_tolerance=config.tom_ratio_tolerance,
                )

                generated_this_tier += remaining

                # Recompute pass rate
                stats = compute_pass_rate_from_calibration(str(all_tasks_dir), model)
                pass_rate = stats["pass_rate"]
                print(
                    f"[evolve] After generating {generated_this_tier} tasks: "
                    f"{stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)"
                )
            tier_generation_seconds = time.time() - gen_start
        else:
            print(
                f"[evolve] Pass rate ({pass_rate:.1f}%) <= target ({config.target_pass_rate:.1f}%) "
                f"— advancing to next model"
            )

        # Save final tier metrics
        final_stats = compute_pass_rate_from_calibration(str(all_tasks_dir), model)
        with open(tier_dir / "tier_metrics_final.json", "w") as f:
            json.dump({
                "model": model,
                "phase": "final",
                "generated_this_tier": generated_this_tier,
                "difficulty": tier_difficulty,
                **final_stats,
            }, f, indent=2)

        # Collect metrics for summary table
        tier_metrics_list.append({
            "tier_idx": tier_idx + 1,
            "model": model,
            "benchmark_seconds": tier_benchmark_seconds,
            "generation_seconds": tier_generation_seconds,
            "generated": generated_this_tier,
            "pass_rate_before": pass_rate_before,
            "pass_rate_after": final_stats["pass_rate"],
            "difficulty": tier_difficulty,
        })

        # Checkpoint
        completed_tiers.add(tier_name)
        state["completed_tiers"] = list(completed_tiers)
        state["current_tier_idx"] = tier_idx + 1
        save_state(run_dir, state)

    # ---- Final summary ----
    total_tasks = len(list(all_tasks_dir.glob("*.json")))

    # Write machine-readable summary
    summary = {
        "total_tasks": total_tasks,
        "tasks_dir": str(all_tasks_dir.resolve()),
        "total_duration_seconds": time.time() - pipeline_start,
        "model_stats": {},
        "tier_metrics": tier_metrics_list,
    }
    for model in config.model_ladder:
        summary["model_stats"][model] = compute_pass_rate_from_calibration(str(all_tasks_dir), model)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print human-readable terminal table
    print_summary_table(
        pipeline_start=pipeline_start,
        seed_duration=seed_duration,
        seed_tasks_generated=seed_tasks_generated,
        tier_metrics=tier_metrics_list,
        all_tasks_dir=str(all_tasks_dir),
        config=config,
    )

    print(f"  Run metadata in {run_dir}")


def main():
    args = parse_args()
    try:
        resolve_categories(args.category)
    except ValueError as e:
        raise SystemExit(str(e))

    tom_target_sum = args.tom_target_l1 + args.tom_target_l2 + args.tom_target_l3
    if abs(tom_target_sum - 1.0) > 1e-6:
        raise SystemExit(
            f"--tom-target-l1/2/3 must sum to 1.0, got {tom_target_sum:.6f}"
        )
    if args.tom_ratio_tolerance < 0:
        raise SystemExit("--tom-ratio-tolerance must be non-negative")

    config = EvolutionConfig(
        model_ladder=args.model_ladder.split(","),
        generator_model=args.generator_model,
        tasks_per_round=args.tasks_per_round,
        icl_total_examples=args.icl_total_examples,
        icl_failure_ratio=args.icl_failure_ratio,
        judge_threshold=args.judge_threshold,
        target_pass_rate=args.target_pass_rate,
        seed_tasks_dir=args.seed_tasks_dir,
        seed_pool_size=args.seed_pool_size,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        focus=args.focus,
        category=args.category,
        tom_target_l1=args.tom_target_l1,
        tom_target_l2=args.tom_target_l2,
        tom_target_l3=args.tom_target_l3,
        tom_ratio_tolerance=args.tom_ratio_tolerance,
    )

    run_evolution(config, resume_dir=args.resume)


if __name__ == "__main__":
    main()
