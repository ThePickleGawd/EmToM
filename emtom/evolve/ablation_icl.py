#!/usr/bin/env python3
"""Ablation: ICL-guided evolution vs unguided generation for gpt-5.4 targeting.

ICL condition: annotated benchmark examples (8 fail + 2 pass from gpt-5.2 calibration)
               plus evolution query explaining failure patterns.
No-ICL condition: default seed selection (structural seeds, no annotations, no query).

Both conditions:
- Generator model: gpt-5.2
- Test model: gpt-5.4 (medium reasoning, automatic)
- Seed pool: data/emtom/tasks (300 tasks with gpt-5.2 calibration)
- Difficulty: medium
- Judge gate + test_task gate + submit_task validation
- 50 cooperative + 50 mixed per condition
- 20 workers per group (4 groups = 80 total)

Usage:
    python -m emtom.evolve.ablation_icl
    python -m emtom.evolve.ablation_icl --resume  # resume from existing run
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from emtom.evolve.icl_sampler import (
    prepare_sampled_tasks_dir_from_calibration,
    compute_pass_rate_from_calibration,
    build_evolution_query,
)

# ── Config ──────────────────────────────────────────────────────────────────
SEED_TASKS_DIR = "data/emtom/tasks"
GENERATOR_MODEL = "gpt-5.4"
TEST_MODEL = "gpt-5.4"
CALIBRATION_MODEL = "gpt-5.4"  # ICL samples drawn from this model's results
DIFFICULTY = "hard"
TASKS_PER_CATEGORY = 25
WORKERS_PER_GROUP = 24  # 2 groups × 24 = 48 total
MAX_SPAWNS_MULTIPLIER = 5  # max spawns = tasks × this
ICL_FAIL_COUNT = 8
ICL_PASS_COUNT = 2
MONITOR_INTERVAL_SECONDS = 900  # 15 minutes
# Skip expensive pipeline stages: PDDL verification, simulation, ToM proofs
# test_task (gpt-5.4 benchmark) is kept — needed for pass rate data.
REMOVE_STEPS = ["pddl", "simulation", "tom"]

BASE_OUTPUT = Path("outputs/evolve_ablation")

CONDITIONS = ["icl", "no_icl"]
CATEGORIES = ["cooperative"]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _count_valid_json(directory: Path) -> int:
    """Count valid JSON files in directory."""
    count = 0
    if not directory.exists():
        return 0
    for jf in directory.glob("*.json"):
        try:
            with open(jf) as f:
                json.load(f)
            count += 1
        except Exception:
            continue
    return count


def _count_by_category(directory: Path) -> Dict[str, int]:
    """Count tasks by category in a directory."""
    counts: Dict[str, int] = {"cooperative": 0, "competitive": 0, "mixed": 0}
    if not directory.exists():
        return counts
    for jf in directory.glob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
            cat = data.get("category", "unknown")
            if cat in counts:
                counts[cat] += 1
        except Exception:
            continue
    return counts


def _extract_step_counts_from_logs(log_dir: Path) -> Dict[str, float]:
    """Extract generation step counts from log files.

    Parses 'Iteration X/Y | Submitted: N/M' lines to find
    how many steps each submitted task took.
    """
    if not log_dir.exists():
        return {"total_steps": 0, "submitted_tasks": 0, "avg_steps_per_task": 0.0,
                "log_files_parsed": 0}

    total_steps = 0
    total_submitted = 0
    logs_parsed = 0

    for log_file in sorted(log_dir.glob("*.log")):
        try:
            with open(log_file) as f:
                content = f.read()
        except Exception:
            continue

        logs_parsed += 1

        # Find the last "Iteration X/Y" line to get total steps used
        iter_matches = re.findall(r"Iteration (\d+)/\d+", content)
        if iter_matches:
            total_steps += int(iter_matches[-1])
        else:
            # Fallback: count "mini-swe-agent (step N," lines
            step_matches = re.findall(r"mini-swe-agent \(step (\d+),", content)
            if step_matches:
                total_steps += int(step_matches[-1])

        # Find the last "Submitted: N/M" to get tasks submitted
        submit_matches = re.findall(r"Submitted: (\d+)/\d+", content)
        if submit_matches:
            total_submitted += int(submit_matches[-1])

    avg = total_steps / total_submitted if total_submitted > 0 else 0.0
    return {
        "total_steps": total_steps,
        "submitted_tasks": total_submitted,
        "avg_steps_per_task": round(avg, 1),
        "log_files_parsed": logs_parsed,
    }


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = int(minutes // 60)
    remaining = int(minutes % 60)
    return f"{hours}h {remaining}m"


# ── GPU detection ───────────────────────────────────────────────────────────

def _detect_gpu_ids() -> List[int]:
    """Detect available CUDA GPU indices via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        pass
    return [0]


# ── Generation runner ───────────────────────────────────────────────────────

def run_generation_group(
    condition: str,
    category: str,
    output_dir: Path,
    log_dir: Path,
    sampled_tasks_dir: Optional[str] = None,
    query: Optional[str] = None,
    num_tasks: int = TASKS_PER_CATEGORY,
    max_workers: int = WORKERS_PER_GROUP,
) -> None:
    """Run parallel generation for one condition × category group.

    Spawns up to max_workers concurrent `run_emtom.sh generate --num-tasks 1`
    processes. Polls for completed JSON files until num_tasks are produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    group_name = f"{condition}/{category}"
    max_spawns = num_tasks * MAX_SPAWNS_MULTIPLIER

    # Build base command
    base_cmd = [
        "./emtom/run_emtom.sh", "generate",
        "--model", GENERATOR_MODEL,
        "--num-tasks", "1",
        "--output-dir", str(output_dir),
        "--test-model", TEST_MODEL,
        "--seed-tasks-dir", SEED_TASKS_DIR,
        "--category", category,
        "--difficulty", DIFFICULTY,
    ]
    if REMOVE_STEPS:
        base_cmd.extend(["--remove"] + REMOVE_STEPS)
    if sampled_tasks_dir:
        base_cmd.extend(["--sampled-tasks-dir", sampled_tasks_dir])
    if query:
        base_cmd.extend(["--query", query])

    # Detect GPUs for round-robin
    gpu_ids = _detect_gpu_ids()
    print(f"[ablation:{group_name}] GPUs detected: {gpu_ids}")

    total_spawned = 0
    active: List[Tuple[subprocess.Popen, object]] = []
    baseline_count = _count_valid_json(output_dir)
    target_count = baseline_count + num_tasks

    print(f"[ablation:{group_name}] Starting: target={num_tasks} tasks, "
          f"workers={max_workers}, baseline={baseline_count}")
    print(f"[ablation:{group_name}] Output: {output_dir.resolve()}")
    print(f"[ablation:{group_name}] Logs: {log_dir.resolve()}")

    try:
        while True:
            # Reap finished processes
            still_active = []
            for proc, fh in active:
                if proc.poll() is not None:
                    fh.close()
                else:
                    still_active.append((proc, fh))
            active = still_active

            done_count = _count_valid_json(output_dir)
            new_count = done_count - baseline_count

            # Target reached — wait for active workers to finish
            if done_count >= target_count:
                if not active:
                    print(f"\n[ablation:{group_name}] DONE: {new_count}/{num_tasks} tasks")
                    break
                time.sleep(5)
                continue

            # Spawn new processes with GPU round-robin
            needed = target_count - done_count
            concurrency_cap = min(max_workers, needed)
            while len(active) < concurrency_cap and total_spawned < max_spawns:
                if _count_valid_json(output_dir) >= target_count:
                    break
                log_file = log_dir / f"gen_{total_spawned}_{category}.log"
                fh = open(log_file, "w")
                gpu_id = gpu_ids[total_spawned % len(gpu_ids)] if gpu_ids else 0
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                proc = subprocess.Popen(base_cmd, stdout=fh, stderr=fh, env=env)
                active.append((proc, fh))
                total_spawned += 1

            if not active and total_spawned >= max_spawns:
                done_count = _count_valid_json(output_dir)
                new_count = done_count - baseline_count
                print(f"\n[ablation:{group_name}] Exhausted {max_spawns} spawns, "
                      f"got {new_count}/{num_tasks}")
                break

            time.sleep(15)

    except Exception:
        for proc, fh in active:
            proc.terminate()
        for proc, fh in active:
            proc.wait()
            fh.close()
        raise


# ── Setup ───────────────────────────────────────────────────────────────────

def setup_ablation(run_dir: Path) -> Tuple[str, str, str]:
    """Set up directory structure and ICL examples.

    Returns:
        (icl_sampled_dir, no_icl_sampled_dir, evolution_query).
        no_icl_sampled_dir is an empty directory — no reference examples.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create output dirs for each group
    for condition in CONDITIONS:
        for category in CATEGORIES:
            (run_dir / condition / f"gen_{category}").mkdir(parents=True, exist_ok=True)
            (run_dir / condition / f"logs_{category}").mkdir(parents=True, exist_ok=True)

    # ICL condition: annotated examples with gpt-5.4 calibration trajectories
    sampled_dir = str(run_dir / "icl" / "sampled_tasks")
    prepare_sampled_tasks_dir_from_calibration(
        tasks_dir=SEED_TASKS_DIR,
        model=CALIBRATION_MODEL,
        output_dir=sampled_dir,
        fail_count=ICL_FAIL_COUNT,
        pass_count=ICL_PASS_COUNT,
    )

    # No-ICL condition: empty sampled_tasks directory (no reference examples)
    no_icl_sampled_dir = str(run_dir / "no_icl" / "sampled_tasks_empty")
    Path(no_icl_sampled_dir).mkdir(parents=True, exist_ok=True)

    # Build evolution query from gpt-5.4 pass rate
    stats = compute_pass_rate_from_calibration(SEED_TASKS_DIR, CALIBRATION_MODEL)
    pass_rate = stats["pass_rate"]
    evolution_query = build_evolution_query(pass_rate, CALIBRATION_MODEL, generation_idx=1)
    print(f"[ablation] {CALIBRATION_MODEL} pass rate on seed pool: {pass_rate:.1f}% "
          f"({stats['passed']}/{stats['total']})")

    # Save config
    config = {
        "created": datetime.now().isoformat(),
        "seed_tasks_dir": SEED_TASKS_DIR,
        "generator_model": GENERATOR_MODEL,
        "test_model": TEST_MODEL,
        "calibration_model": CALIBRATION_MODEL,
        "difficulty": DIFFICULTY,
        "tasks_per_category": TASKS_PER_CATEGORY,
        "workers_per_group": WORKERS_PER_GROUP,
        "icl_fail_count": ICL_FAIL_COUNT,
        "icl_pass_count": ICL_PASS_COUNT,
        "seed_pool_pass_rate": pass_rate,
        "seed_pool_stats": stats,
        "evolution_query": evolution_query,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save evolution query for reference
    with open(run_dir / "icl" / "evolution_query.txt", "w") as f:
        f.write(evolution_query)

    return sampled_dir, no_icl_sampled_dir, evolution_query


# ── Monitoring ──────────────────────────────────────────────────────────────

def collect_progress(run_dir: Path) -> Dict:
    """Collect progress across all groups."""
    progress = {
        "timestamp": datetime.now().isoformat(),
        "groups": {},
    }
    for condition in CONDITIONS:
        for category in CATEGORIES:
            group_key = f"{condition}/{category}"
            gen_dir = run_dir / condition / f"gen_{category}"
            log_dir = run_dir / condition / f"logs_{category}"

            task_count = _count_valid_json(gen_dir)
            step_stats = _extract_step_counts_from_logs(log_dir)

            progress["groups"][group_key] = {
                "tasks_generated": task_count,
                "target": TASKS_PER_CATEGORY,
                **step_stats,
            }

    return progress


def print_progress(progress: Dict, elapsed: float) -> None:
    """Print formatted progress table."""
    print(f"\n{'='*72}")
    print(f"  ABLATION PROGRESS — {progress['timestamp']}  (elapsed: {_fmt_duration(elapsed)})")
    print(f"{'='*72}")

    header = f"  {'Group':<20} {'Tasks':>7} {'Steps':>8} {'Avg S/T':>8} {'Logs':>6}"
    print(header)
    print("  " + "-" * 52)

    for group_key in sorted(progress["groups"]):
        g = progress["groups"][group_key]
        tasks = f"{g['tasks_generated']}/{g['target']}"
        steps = str(g["total_steps"])
        avg = f"{g['avg_steps_per_task']:.1f}" if g["avg_steps_per_task"] > 0 else "-"
        logs = str(g["log_files_parsed"])
        print(f"  {group_key:<20} {tasks:>7} {steps:>8} {avg:>8} {logs:>6}")

    print(f"{'='*72}\n")


def save_progress_snapshot(run_dir: Path, progress: Dict, elapsed: float) -> None:
    """Append progress snapshot to history file."""
    history_file = run_dir / "progress_history.jsonl"
    entry = {"elapsed_seconds": round(elapsed), **progress}
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ICL ablation for gpt-5.4 evolution")
    parser.add_argument("--resume", action="store_true", help="Resume latest run")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory")
    args = parser.parse_args()

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
        # Find latest run
        if not BASE_OUTPUT.exists():
            print("No existing runs to resume.", file=sys.stderr)
            sys.exit(1)
        runs = sorted([d for d in BASE_OUTPUT.iterdir() if d.is_dir()])
        if not runs:
            print("No existing runs to resume.", file=sys.stderr)
            sys.exit(1)
        run_dir = runs[-1]
        print(f"[ablation] Resuming from {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = BASE_OUTPUT / timestamp

    # Setup
    print(f"[ablation] Run directory: {run_dir.resolve()}")
    sampled_dir, no_icl_sampled_dir, evolution_query = setup_ablation(run_dir)

    # Check existing progress for resume
    for condition in CONDITIONS:
        for category in CATEGORIES:
            gen_dir = run_dir / condition / f"gen_{category}"
            existing = _count_valid_json(gen_dir)
            if existing > 0:
                print(f"[ablation] {condition}/{category}: {existing} tasks already exist (resuming)")

    pipeline_start = time.time()

    # Launch 4 groups as threads (each runs its own subprocess pool)
    threads: Dict[str, threading.Thread] = {}
    errors: Dict[str, Optional[Exception]] = {}

    def _run_group_safe(condition: str, category: str):
        """Thread-safe wrapper that captures exceptions."""
        group_key = f"{condition}/{category}"
        try:
            gen_dir = run_dir / condition / f"gen_{category}"
            log_dir = run_dir / condition / f"logs_{category}"

            # Compute remaining tasks (for resume)
            existing = _count_valid_json(gen_dir)
            remaining = TASKS_PER_CATEGORY - existing
            if remaining <= 0:
                print(f"[ablation:{group_key}] Already have {existing} tasks, skipping")
                return

            # ICL: annotated examples + evolution query
            # No-ICL: empty sampled_tasks dir (no reference examples), no query
            if condition == "icl":
                stdir = sampled_dir
                q = evolution_query
            else:
                stdir = no_icl_sampled_dir
                q = None

            run_generation_group(
                condition=condition,
                category=category,
                output_dir=gen_dir,
                log_dir=log_dir,
                sampled_tasks_dir=stdir,
                query=q,
                num_tasks=remaining,
                max_workers=WORKERS_PER_GROUP,
            )
        except Exception as e:
            errors[group_key] = e
            print(f"\n[ablation:{group_key}] ERROR: {e}", file=sys.stderr)

    for condition in CONDITIONS:
        for category in CATEGORIES:
            key = f"{condition}/{category}"
            t = threading.Thread(target=_run_group_safe, args=(condition, category), name=key)
            threads[key] = t
            errors[key] = None
            t.start()

    total_groups = len(CONDITIONS) * len(CATEGORIES)
    print(f"\n[ablation] All {total_groups} groups launched ({WORKERS_PER_GROUP * total_groups} total workers)")

    # Monitor loop
    try:
        while any(t.is_alive() for t in threads.values()):
            time.sleep(MONITOR_INTERVAL_SECONDS)
            elapsed = time.time() - pipeline_start
            progress = collect_progress(run_dir)
            print_progress(progress, elapsed)
            save_progress_snapshot(run_dir, progress, elapsed)

            # Check for errors
            for key, err in errors.items():
                if err is not None:
                    print(f"[ablation] WARNING: {key} failed with: {err}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[ablation] Interrupted — generation processes are still running in background")
        print(f"[ablation] Resume with: python -m emtom.evolve.ablation_icl --run-dir {run_dir}")
        sys.exit(1)

    # Wait for all threads
    for t in threads.values():
        t.join()

    # Final report
    elapsed = time.time() - pipeline_start
    progress = collect_progress(run_dir)
    print_progress(progress, elapsed)
    save_progress_snapshot(run_dir, progress, elapsed)

    # Write final summary
    summary = {
        "completed": datetime.now().isoformat(),
        "total_duration_seconds": round(elapsed),
        "total_duration_human": _fmt_duration(elapsed),
        "groups": progress["groups"],
        "errors": {k: str(v) if v else None for k, v in errors.items()},
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ablation] Summary written to {run_dir / 'summary.json'}")
    print(f"[ablation] Total duration: {_fmt_duration(elapsed)}")

    # Print comparison table
    print(f"\n{'='*72}")
    print("  ABLATION RESULTS — ICL vs No-ICL")
    print(f"{'='*72}")
    header = f"  {'Condition':<12} {'Category':<14} {'Tasks':>7} {'Avg Steps/Task':>16}"
    print(header)
    print("  " + "-" * 50)
    for condition in CONDITIONS:
        for category in CATEGORIES:
            g = progress["groups"][f"{condition}/{category}"]
            avg = f"{g['avg_steps_per_task']:.1f}" if g["avg_steps_per_task"] > 0 else "-"
            print(f"  {condition:<12} {category:<14} {g['tasks_generated']:>7} {avg:>16}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
