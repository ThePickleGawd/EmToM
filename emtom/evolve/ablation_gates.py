#!/usr/bin/env python3
"""Ablation: skip_baseline vs skip_judge gate experiments.

skip_baseline: test_task runs only standard mode (no baseline solvability check).
               --remove pddl simulation tom baseline
skip_judge:    LLM council auto-passes (no quality check).
               --remove pddl simulation tom llm-council

Both conditions:
- Generator model: gpt-5.4
- Test model: gpt-5.4
- ICL enabled (shared sampled_tasks from gpt-5.4 calibration)
- 24 workers per condition (48 total)
- Cooperative + mixed tasks (round-robin per spawn)
- Output stored separately — NOT copied to data/emtom/tasks/

Post-hoc analysis (after generation):
- skip_baseline tasks: run baseline benchmark, report solvability rate
- skip_judge tasks: run LLM council judge, report avg score + threshold pass rate

Usage:
    python -m emtom.evolve.ablation_gates
    python -m emtom.evolve.ablation_gates --resume
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
from typing import Any, Dict, List, Optional, Tuple

from emtom.evolve.icl_sampler import (
    prepare_sampled_tasks_dir_from_calibration,
    compute_pass_rate_from_calibration,
    build_evolution_query,
)

# ── Config ──────────────────────────────────────────────────────────────────
SEED_TASKS_DIR = "data/emtom/tasks"
GENERATOR_MODEL = "gpt-5.4"
TEST_MODEL = "gpt-5.4"
CALIBRATION_MODEL = "gpt-5.4"
DIFFICULTY = "hard"
TASKS_PER_CONDITION = 50
WORKERS_PER_CONDITION = 24
MAX_SPAWNS_MULTIPLIER = 5
ICL_FAIL_COUNT = 8
ICL_PASS_COUNT = 2
MONITOR_INTERVAL_SECONDS = 900  # 15 minutes

CONDITIONS = ["skip_baseline", "skip_judge"]
CONDITION_REMOVE_STEPS: Dict[str, List[str]] = {
    "skip_baseline": ["pddl", "simulation", "tom", "baseline"],
    "skip_judge": ["pddl", "simulation", "tom", "llm-council"],
}
CATEGORIES = ["cooperative", "mixed"]

BASE_OUTPUT = Path("outputs/evolve_ablation_gates")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _count_valid_json(directory: Path) -> int:
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

        iter_matches = re.findall(r"Iteration (\d+)/\d+", content)
        if iter_matches:
            total_steps += int(iter_matches[-1])
        else:
            step_matches = re.findall(r"mini-swe-agent \(step (\d+),", content)
            if step_matches:
                total_steps += int(step_matches[-1])

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


def _detect_gpu_ids() -> List[int]:
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
    output_dir: Path,
    log_dir: Path,
    remove_steps: List[str],
    sampled_tasks_dir: Optional[str] = None,
    query: Optional[str] = None,
    num_tasks: int = TASKS_PER_CONDITION,
    max_workers: int = WORKERS_PER_CONDITION,
) -> None:
    """Run parallel generation for one condition.

    Categories rotate round-robin per spawn (cooperative, mixed, cooperative, ...).
    No auto-copy to data/emtom/tasks/ — ablation tasks stay separate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    max_spawns = num_tasks * MAX_SPAWNS_MULTIPLIER
    gpu_ids = _detect_gpu_ids()
    print(f"[ablation:{condition}] GPUs detected: {gpu_ids}")

    total_spawned = 0
    active: List[Tuple[subprocess.Popen, Any]] = []
    baseline_count = _count_valid_json(output_dir)
    target_count = baseline_count + num_tasks

    print(f"[ablation:{condition}] Starting: target={num_tasks} tasks, "
          f"workers={max_workers}, baseline={baseline_count}")
    print(f"[ablation:{condition}] Remove steps: {remove_steps}")
    print(f"[ablation:{condition}] Output: {output_dir.resolve()}")

    try:
        while True:
            # Reap finished
            still_active = []
            for proc, fh in active:
                if proc.poll() is not None:
                    fh.close()
                else:
                    still_active.append((proc, fh))
            active = still_active

            done_count = _count_valid_json(output_dir)
            new_count = done_count - baseline_count

            if done_count >= target_count:
                if not active:
                    print(f"\n[ablation:{condition}] DONE: {new_count}/{num_tasks} tasks")
                    break
                time.sleep(5)
                continue

            # Spawn with GPU + category round-robin
            needed = target_count - done_count
            concurrency_cap = min(max_workers, needed)
            while len(active) < concurrency_cap and total_spawned < max_spawns:
                if _count_valid_json(output_dir) >= target_count:
                    break

                category = CATEGORIES[total_spawned % len(CATEGORIES)]
                log_file = log_dir / f"gen_{total_spawned}_{category}.log"
                fh = open(log_file, "w")
                gpu_id = gpu_ids[total_spawned % len(gpu_ids)] if gpu_ids else 0

                cmd = [
                    "./emtom/run_emtom.sh", "generate",
                    "--model", GENERATOR_MODEL,
                    "--num-tasks", "1",
                    "--output-dir", str(output_dir),
                    "--test-model", TEST_MODEL,
                    "--seed-tasks-dir", SEED_TASKS_DIR,
                    "--category", category,
                    "--difficulty", DIFFICULTY,
                ]
                if remove_steps:
                    cmd.extend(["--remove"] + remove_steps)
                if sampled_tasks_dir:
                    cmd.extend(["--sampled-tasks-dir", sampled_tasks_dir])
                if query:
                    cmd.extend(["--query", query])

                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)
                active.append((proc, fh))
                total_spawned += 1

            if not active and total_spawned >= max_spawns:
                done_count = _count_valid_json(output_dir)
                new_count = done_count - baseline_count
                print(f"\n[ablation:{condition}] Exhausted {max_spawns} spawns, "
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

def setup_ablation(run_dir: Path) -> Tuple[str, str]:
    """Set up directory structure and shared ICL examples.

    Returns:
        (sampled_tasks_dir, evolution_query)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        (run_dir / condition / "gen").mkdir(parents=True, exist_ok=True)
        (run_dir / condition / "logs").mkdir(parents=True, exist_ok=True)

    # Shared ICL examples from gpt-5.4 calibration (both conditions get ICL)
    sampled_dir = str(run_dir / "shared_sampled_tasks")
    prepare_sampled_tasks_dir_from_calibration(
        tasks_dir=SEED_TASKS_DIR,
        model=CALIBRATION_MODEL,
        output_dir=sampled_dir,
        fail_count=ICL_FAIL_COUNT,
        pass_count=ICL_PASS_COUNT,
    )

    # Build evolution query
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
        "tasks_per_condition": TASKS_PER_CONDITION,
        "workers_per_condition": WORKERS_PER_CONDITION,
        "conditions": CONDITIONS,
        "condition_remove_steps": CONDITION_REMOVE_STEPS,
        "categories": CATEGORIES,
        "seed_pool_pass_rate": pass_rate,
        "seed_pool_stats": stats,
        "note": "Ablation tasks NOT copied to data/emtom/tasks/",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(run_dir / "evolution_query.txt", "w") as f:
        f.write(evolution_query)

    return sampled_dir, evolution_query


# ── Monitoring ──────────────────────────────────────────────────────────────

def collect_progress(run_dir: Path) -> Dict:
    progress = {
        "timestamp": datetime.now().isoformat(),
        "groups": {},
    }
    for condition in CONDITIONS:
        gen_dir = run_dir / condition / "gen"
        log_dir = run_dir / condition / "logs"

        task_count = _count_valid_json(gen_dir)
        step_stats = _extract_step_counts_from_logs(log_dir)
        cat_counts = _count_by_category(gen_dir)

        progress["groups"][condition] = {
            "tasks_generated": task_count,
            "target": TASKS_PER_CONDITION,
            "by_category": cat_counts,
            **step_stats,
        }

    return progress


def print_progress(progress: Dict, elapsed: float) -> None:
    print(f"\n{'='*72}")
    print(f"  GATE ABLATION PROGRESS — {progress['timestamp']}  (elapsed: {_fmt_duration(elapsed)})")
    print(f"{'='*72}")

    header = f"  {'Condition':<18} {'Tasks':>7} {'Coop':>6} {'Mixed':>6} {'Steps':>8} {'Avg S/T':>8}"
    print(header)
    print("  " + "-" * 58)

    for condition in CONDITIONS:
        g = progress["groups"][condition]
        tasks = f"{g['tasks_generated']}/{g['target']}"
        cats = g.get("by_category", {})
        coop = str(cats.get("cooperative", 0))
        mixed = str(cats.get("mixed", 0))
        steps = str(g["total_steps"])
        avg = f"{g['avg_steps_per_task']:.1f}" if g["avg_steps_per_task"] > 0 else "-"
        print(f"  {condition:<18} {tasks:>7} {coop:>6} {mixed:>6} {steps:>8} {avg:>8}")

    print(f"{'='*72}\n")


def save_progress_snapshot(run_dir: Path, progress: Dict, elapsed: float) -> None:
    history_file = run_dir / "progress_history.jsonl"
    entry = {"elapsed_seconds": round(elapsed), **progress}
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gate ablation: skip_baseline vs skip_judge")
    parser.add_argument("--resume", action="store_true", help="Resume latest run")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
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

    print(f"[ablation] Run directory: {run_dir.resolve()}")
    sampled_dir, evolution_query = setup_ablation(run_dir)

    # Check existing progress
    for condition in CONDITIONS:
        gen_dir = run_dir / condition / "gen"
        existing = _count_valid_json(gen_dir)
        if existing > 0:
            print(f"[ablation] {condition}: {existing} tasks already exist (resuming)")

    pipeline_start = time.time()

    # Launch one thread per condition
    threads: Dict[str, threading.Thread] = {}
    errors: Dict[str, Optional[Exception]] = {}

    def _run_condition_safe(condition: str):
        try:
            gen_dir = run_dir / condition / "gen"
            log_dir = run_dir / condition / "logs"

            existing = _count_valid_json(gen_dir)
            remaining = TASKS_PER_CONDITION - existing
            if remaining <= 0:
                print(f"[ablation:{condition}] Already have {existing} tasks, skipping")
                return

            run_generation_group(
                condition=condition,
                output_dir=gen_dir,
                log_dir=log_dir,
                remove_steps=CONDITION_REMOVE_STEPS[condition],
                sampled_tasks_dir=sampled_dir,
                query=evolution_query,
                num_tasks=remaining,
                max_workers=WORKERS_PER_CONDITION,
            )
        except Exception as e:
            errors[condition] = e
            print(f"\n[ablation:{condition}] ERROR: {e}", file=sys.stderr)

    for condition in CONDITIONS:
        t = threading.Thread(target=_run_condition_safe, args=(condition,), name=condition)
        threads[condition] = t
        errors[condition] = None
        t.start()

    total_workers = WORKERS_PER_CONDITION * len(CONDITIONS)
    print(f"\n[ablation] All {len(CONDITIONS)} conditions launched ({total_workers} total workers)")

    # Monitor loop
    try:
        while any(t.is_alive() for t in threads.values()):
            time.sleep(MONITOR_INTERVAL_SECONDS)
            elapsed = time.time() - pipeline_start
            progress = collect_progress(run_dir)
            print_progress(progress, elapsed)
            save_progress_snapshot(run_dir, progress, elapsed)

            for key, err in errors.items():
                if err is not None:
                    print(f"[ablation] WARNING: {key} failed with: {err}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[ablation] Interrupted — processes still running in background")
        print(f"[ablation] Resume with: python -m emtom.evolve.ablation_gates --run-dir {run_dir}")
        sys.exit(1)

    for t in threads.values():
        t.join()

    # Final report
    elapsed = time.time() - pipeline_start
    progress = collect_progress(run_dir)
    print_progress(progress, elapsed)
    save_progress_snapshot(run_dir, progress, elapsed)

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

    print(f"\n{'='*72}")
    print("  GATE ABLATION RESULTS — skip_baseline vs skip_judge")
    print(f"{'='*72}")
    header = f"  {'Condition':<18} {'Tasks':>7} {'Coop':>6} {'Mixed':>6} {'Avg Steps/Task':>16}"
    print(header)
    print("  " + "-" * 56)
    for condition in CONDITIONS:
        g = progress["groups"][condition]
        cats = g.get("by_category", {})
        coop = str(cats.get("cooperative", 0))
        mixed = str(cats.get("mixed", 0))
        avg = f"{g['avg_steps_per_task']:.1f}" if g["avg_steps_per_task"] > 0 else "-"
        print(f"  {condition:<18} {g['tasks_generated']:>7} {coop:>6} {mixed:>6} {avg:>16}")
    print(f"{'='*72}")

    print(f"\n[ablation] Post-hoc analysis needed:")
    print(f"  skip_baseline: run baseline benchmark on {run_dir / 'skip_baseline' / 'gen'}")
    print(f"  skip_judge: run LLM council judge on {run_dir / 'skip_judge' / 'gen'}")


if __name__ == "__main__":
    main()
