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

from emtom.evolve.config import EvolutionConfig, DEFAULT_MODEL_LADDER
from emtom.evolve.benchmark_wrapper import (
    run_benchmark,
    run_benchmark_parallel,
    BenchmarkResults,
    TaskResult,
)
from emtom.evolve.icl_sampler import prepare_sampled_tasks_dir, build_evolution_query
from emtom.evolve.report import generate_report


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


def get_difficulty_for_tier(tier_idx: int, total_tiers: int) -> str:
    """Map tier index to difficulty label for the judge."""
    if total_tiers <= 1:
        return "medium"
    ratio = tier_idx / total_tiers  # 0.0 = seed, 1.0 = hardest
    if ratio <= 0.25:
        return "easy"
    elif ratio <= 0.6:
        return "medium"
    else:
        return "hard"


def get_subtask_range(tier_idx: int) -> tuple:
    """Return (subtasks_min, subtasks_max) for a given tier index."""
    idx = min(tier_idx, len(TIER_SUBTASK_RANGES) - 1)
    return TIER_SUBTASK_RANGES[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolutionary difficulty task generation")
    parser.add_argument(
        "--model-ladder", type=str,
        default=",".join(DEFAULT_MODEL_LADDER),
        help="Comma-separated model ladder (default: %(default)s)",
    )
    parser.add_argument("--generator-model", type=str, default="gpt-5.2")
    parser.add_argument("--tasks-per-round", type=int, default=20)
    parser.add_argument("--seed-pool-size", type=int, default=30)
    parser.add_argument("--icl-total-examples", type=int, default=10)
    parser.add_argument("--icl-failure-ratio", type=float, default=0.9)
    parser.add_argument(
        "--seed-query", type=str,
        default=None,
        help="Optional seed query for tier-0 generation (default: none)",
    )
    parser.add_argument("--judge-threshold", type=float, default=0.7,
                        help="Judge threshold for generation quality gate (default: 0.7)")
    parser.add_argument(
        "--target-pass-rate",
        type=float,
        default=30.0,
        help="Reuse tasks (skip generation) when pass rate is at/below this percent (default: 30.0)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/evolve")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Max parallel generation/benchmark processes (default: 50)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing output directory")
    return parser.parse_args()


def load_state(output_dir: Path) -> dict:
    """Load checkpoint state from state.json."""
    state_file = output_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"completed_tiers": [], "tier_results": {}}


def save_state(output_dir: Path, state: dict) -> None:
    """Save checkpoint state to state.json."""
    with open(output_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)


def run_generate(
    model: str,
    num_tasks: int,
    output_dir: str,
    query: Optional[str] = None,
    category: str = "cooperative",
    sampled_tasks_dir: Optional[str] = None,
    judge_threshold: Optional[float] = None,
    subtasks_min: Optional[int] = None,
    subtasks_max: Optional[int] = None,
    difficulty: Optional[str] = None,
) -> Path:
    """Run task generation via run_emtom.sh generate.

    Returns:
        Path to the output directory containing generated tasks.
    """
    cmd = [
        "./emtom/run_emtom.sh", "generate",
        "--model", model,
        "--num-tasks", str(num_tasks),
        "--category", category,
        "--output-dir", output_dir,
    ]
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

    print(f"[evolve] Running generation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[evolve] WARNING: generation exited with code {result.returncode}", file=sys.stderr)

    return Path(output_dir)


def run_generate_parallel(
    model: str,
    num_tasks: int,
    output_dir: str,
    max_workers: int = 50,
    query: Optional[str] = None,
    category: str = "cooperative",
    sampled_tasks_dir: Optional[str] = None,
    judge_threshold: Optional[float] = None,
    subtasks_min: Optional[int] = None,
    subtasks_max: Optional[int] = None,
    difficulty: Optional[str] = None,
) -> Path:
    """Run task generation in parallel via N independent processes.

    Spawns up to max_workers concurrent `run_emtom.sh generate --num-tasks 1`
    processes, all writing to the same output_dir. Stops once num_tasks JSON
    files exist. Caps total spawns at num_tasks * 3 to avoid infinite retries.

    Returns:
        Path to output_dir containing generated tasks.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)

    # Build base command (each process generates 1 task)
    base_cmd = [
        "./emtom/run_emtom.sh", "generate",
        "--model", model,
        "--num-tasks", "1",
        "--category", category,
        "--output-dir", output_dir,
    ]
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

    spinner_chars = ["|", "/", "-", "\\"]
    spinner_idx = 0
    last_status = None

    print(f"[evolve] Parallel generation: target={num_tasks}, max_workers={max_workers}")
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

            # Once target is reached, do not kill workers mid-write.
            # Stop spawning and wait for active workers to finish naturally.
            if done_count >= num_tasks:
                if not active:
                    if sys.stdout.isatty():
                        print()  # finish the in-place status line
                    print(f"[evolve] Generation complete: {done_count}/{num_tasks} valid tasks")
                    print(f"[evolve] Tasks saved to: {out.resolve()}")
                    break
                spinner = spinner_chars[spinner_idx % len(spinner_chars)]
                spinner_idx += 1
                status = (
                    f"[evolve] {spinner} target reached ({done_count}/{num_tasks}); "
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
            needed = num_tasks - done_count
            concurrency_cap = min(max_workers, needed)
            while len(active) < concurrency_cap and total_spawned < max_spawns:
                # Re-check in case tasks appeared while spawning
                if _count_valid_tasks() >= num_tasks:
                    break
                log_file = log_dir / f"gen_{total_spawned}.log"
                fh = open(log_file, "w")
                proc = subprocess.Popen(base_cmd, stdout=fh, stderr=fh)
                active.append((proc, fh))
                total_spawned += 1

            if not active and total_spawned >= max_spawns:
                done_count = _count_valid_tasks()
                if sys.stdout.isatty():
                    print()  # finish the in-place status line
                print(f"[evolve] Generation: exhausted {max_spawns} spawns, got {done_count}/{num_tasks} tasks")
                break

            spinner = spinner_chars[spinner_idx % len(spinner_chars)]
            spinner_idx += 1
            status = (
                f"[evolve] {spinner} generating: {done_count}/{num_tasks} valid tasks "
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
    if final_count < num_tasks:
        if final_count == 0:
            raise RuntimeError(
                f"Generation failed: got 0/{num_tasks} valid task files in {out}"
            )
        print(
            f"[evolve] WARNING: got {final_count}/{num_tasks} tasks "
            f"(accepting partial result)",
            file=sys.stderr,
        )

    return out


def collect_tasks(tasks_dir: Path) -> List[Path]:
    """Collect all task JSON files from a directory."""
    return sorted(tasks_dir.glob("*.json"))


def run_evolution(config: EvolutionConfig, resume_dir: Optional[str] = None) -> None:
    """Main evolutionary loop."""
    # Setup output directory
    if resume_dir:
        output_dir = Path(resume_dir)
        print(f"[evolve] Resuming from {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evolve] Run output directory: {output_dir.resolve()}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Load state for resumption
    state = load_state(output_dir)
    completed_tiers = set(state.get("completed_tiers", []))
    tier_results: Dict[str, BenchmarkResults] = {}

    # Reconstruct tier_results from state if resuming
    for tier_name, tier_data in state.get("tier_results", {}).items():
        reconstructed_task_results = []
        for task_data in tier_data.get("results", []):
            try:
                reconstructed_task_results.append(TaskResult(**task_data))
            except TypeError:
                # Backward-compatible with older state files that didn't save full fields.
                continue

        tier_results[tier_name] = BenchmarkResults(
            model=tier_data["model"],
            total=tier_data["total"],
            passed=tier_data["passed"],
            failed=tier_data["failed"],
            pass_rate=tier_data["pass_rate"],
            results=reconstructed_task_results,
        )

    # ---- TIER 0: Seed pool ----
    seed_tier = "tier_0_seed"
    seed_tasks_dir = output_dir / seed_tier / "tasks"

    if seed_tier not in completed_tiers:
        print(f"\n{'='*60}")
        print(f"TIER 0: Generating seed pool ({config.seed_pool_size} tasks)")
        print(f"{'='*60}\n")

        seed_sub_min, seed_sub_max = get_subtask_range(0)
        run_generate_parallel(
            model=config.generator_model,
            num_tasks=config.seed_pool_size,
            output_dir=str(seed_tasks_dir),
            max_workers=config.max_workers,
            query=config.seed_query,
            category="cooperative",
            judge_threshold=config.judge_threshold,
            subtasks_min=seed_sub_min,
            subtasks_max=seed_sub_max,
            difficulty="easy",
        )

        completed_tiers.add(seed_tier)
        state["completed_tiers"] = list(completed_tiers)
        save_state(output_dir, state)
    else:
        print(f"[evolve] Skipping {seed_tier} (already completed)")

    # ---- Tiers 1..N: Benchmark + Generate ----
    prev_tasks_dir = seed_tasks_dir

    for tier_idx, model in enumerate(config.model_ladder):
        tier_name = f"tier_{tier_idx + 1}_{model}"
        tier_dir = output_dir / tier_name

        if tier_name in completed_tiers:
            print(f"[evolve] Skipping {tier_name} (already completed)")
            # Restore prev_tasks_dir for next tier
            tier_tasks = tier_dir / "tasks"
            if tier_tasks.exists() and list(tier_tasks.glob("*.json")):
                prev_tasks_dir = tier_tasks
            continue

        tier_dir.mkdir(parents=True, exist_ok=True)
        is_last_tier = (tier_idx == len(config.model_ladder) - 1)

        print(f"\n{'='*60}")
        print(f"TIER {tier_idx + 1}: {model}")
        print(f"{'='*60}\n")

        # a. BENCHMARK previous tier's tasks against this model
        print(f"[evolve] Benchmarking {prev_tasks_dir} with {model}...")
        benchmark_output = str(tier_dir / "benchmark")
        results = run_benchmark_parallel(
            tasks_dir=str(prev_tasks_dir),
            model=model,
            output_dir=benchmark_output,
            max_workers=config.max_workers,
            no_video=True,
        )
        tier_results[tier_name] = results

        # Save tier metrics
        with open(tier_dir / "tier_metrics.json", "w") as f:
            json.dump({
                "model": model,
                "total": results.total,
                "passed": results.passed,
                "failed": results.failed,
                "pass_rate": results.pass_rate,
            }, f, indent=2)

        print(f"[evolve] {model}: {results.passed}/{results.total} passed ({results.pass_rate:.1f}%)")

        # b. CHECK pass rate threshold and generation size
        skip_generation = False
        tasks_to_generate = config.tasks_per_round
        if results.pass_rate <= config.target_pass_rate and not is_last_tier:
            print(
                f"[evolve] Pass rate ({results.pass_rate:.1f}%) <= target "
                f"({config.target_pass_rate:.1f}%) — reusing same tasks for next tier"
            )
            skip_generation = True
        elif not is_last_tier:
            # Only generate as many harder tasks as needed to push difficulty back down.
            numerator = max(0.0, results.pass_rate - config.target_pass_rate)
            denominator = max(1e-9, 100.0 - config.target_pass_rate)
            difficulty_pressure = min(1.0, numerator / denominator)
            tasks_to_generate = max(1, int(round(config.tasks_per_round * difficulty_pressure)))
            print(
                f"[evolve] Pass rate ({results.pass_rate:.1f}%) > target "
                f"({config.target_pass_rate:.1f}%) — generating {tasks_to_generate}/"
                f"{config.tasks_per_round} harder tasks"
            )

        # c-e. PREPARE sampled tasks, BUILD query, GENERATE harder tasks
        if not is_last_tier and not skip_generation:
            # c. Prepare sampled_tasks dir
            fail_count = int(config.icl_total_examples * config.icl_failure_ratio)
            pass_count = config.icl_total_examples - fail_count
            sampled_dir = str(tier_dir / "sampled_tasks")
            prepare_sampled_tasks_dir(
                benchmark_results=results,
                tasks_dir=str(prev_tasks_dir),
                output_dir=sampled_dir,
                fail_count=fail_count,
                pass_count=pass_count,
            )

            # d. Build evolution query
            evolution_query = build_evolution_query(results, model, tier_idx + 1)

            # e. Generate harder tasks while keeping a fixed quality bar.
            tier_threshold = config.judge_threshold
            print(f"[evolve] Judge threshold for tier {tier_idx + 1}: {tier_threshold:.2f}")

            tier_tasks_dir = tier_dir / "tasks"
            tier_sub_min, tier_sub_max = get_subtask_range(tier_idx + 1)
            # tier_idx+1 because tier 0 is the seed
            tier_difficulty = get_difficulty_for_tier(
                tier_idx + 1, len(config.model_ladder) + 1
            )
            print(f"[evolve] Subtask range for tier {tier_idx + 1}: {tier_sub_min}-{tier_sub_max}")
            print(f"[evolve] Difficulty for tier {tier_idx + 1}: {tier_difficulty}")
            run_generate_parallel(
                model=config.generator_model,
                num_tasks=tasks_to_generate,
                output_dir=str(tier_tasks_dir),
                max_workers=config.max_workers,
                query=evolution_query,
                category="cooperative",
                sampled_tasks_dir=sampled_dir,
                judge_threshold=tier_threshold,
                subtasks_min=tier_sub_min,
                subtasks_max=tier_sub_max,
                difficulty=tier_difficulty,
            )
            prev_tasks_dir = tier_tasks_dir
        elif not is_last_tier and skip_generation:
            # Reuse same tasks — prev_tasks_dir stays the same
            pass

        # f. Save checkpoint
        completed_tiers.add(tier_name)
        state["completed_tiers"] = list(completed_tiers)
        state["tier_results"][tier_name] = {
            "model": results.model,
            "total": results.total,
            "passed": results.passed,
            "failed": results.failed,
            "pass_rate": results.pass_rate,
            "results": [asdict(r) for r in results.results],
        }
        save_state(output_dir, state)

    # ---- Final report ----
    print(f"\n{'='*60}")
    print("Generating final report...")
    print(f"{'='*60}\n")
    generate_report(config, tier_results, str(output_dir))

    print(f"\n[evolve] Evolution complete! Results in {output_dir}")


def main():
    args = parse_args()

    config = EvolutionConfig(
        model_ladder=args.model_ladder.split(","),
        generator_model=args.generator_model,
        tasks_per_round=args.tasks_per_round,
        seed_pool_size=args.seed_pool_size,
        icl_total_examples=args.icl_total_examples,
        icl_failure_ratio=args.icl_failure_ratio,
        judge_threshold=args.judge_threshold,
        target_pass_rate=args.target_pass_rate,
        seed_query=args.seed_query,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )

    run_evolution(config, resume_dir=args.resume)


if __name__ == "__main__":
    main()
