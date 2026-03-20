#!/usr/bin/env python3
"""Remove salvaged tasks that GPT-5.2 passes in the latest benchmark.

Usage:
    python emtom/scripts/cleanup_passing_salvaged.py <benchmark_dir> [--dry-run]
"""

import json
import glob
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python emtom/scripts/cleanup_passing_salvaged.py <benchmark_dir> [--dry-run]")
        sys.exit(1)

    benchmark_dir = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    tasks_dir = "data/emtom/tasks"

    # Collect all passing task IDs from benchmark
    passing_tids = set()
    for f in glob.glob(f"{benchmark_dir}/*/benchmark-*/results/benchmark_summary.json"):
        d = json.load(open(f))
        for r in d.get("results", []):
            success = r.get("success", False)
            eval_data = r.get("evaluation", {})
            progress = eval_data.get("percent_complete", 0)
            if success or progress >= 1.0:
                passing_tids.add(r.get("task_id", ""))

    print(f"Passing task IDs from benchmark: {len(passing_tids)}")

    # Find salvaged task files that match passing IDs
    removed = 0
    kept = 0
    for f in sorted(glob.glob(os.path.join(tasks_dir, "202602*.json"))):
        basename = os.path.basename(f)
        # Extract task hash from filename
        task_hash = basename.split("-")[-1].replace(".json", "")

        # Check if this task passed
        is_passing = any(task_hash in tid for tid in passing_tids)

        if is_passing:
            if dry_run:
                print(f"  [DRY-RUN] Would remove: {basename[:60]}")
            else:
                os.remove(f)
                print(f"  [REMOVED] {basename[:60]}")
            removed += 1
        else:
            kept += 1

    total_tasks = len(glob.glob(os.path.join(tasks_dir, "*.json")))
    print(f"\nRemoved: {removed} passing salvaged tasks")
    print(f"Kept: {kept} failing salvaged tasks")
    print(f"Total tasks remaining: {total_tasks - (0 if dry_run else removed)}")

    if dry_run:
        print("\n[DRY-RUN — no files removed]")


if __name__ == "__main__":
    main()
