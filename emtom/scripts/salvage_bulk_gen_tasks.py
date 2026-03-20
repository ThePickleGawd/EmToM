#!/usr/bin/env python3
"""Salvage structurally valid tasks from old bulk gen outputs.

Copies tasks from outputs/bulk_gen_tasks/ into data/emtom/tasks/,
filtering for:
  - Structurally valid (has problem_pddl, golden_trajectory)
  - Not already in the main pool (by title + scene_id)
  - GPT-5.2 failed in old calibration (optional filter)
"""

import json
import glob
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict


def load_existing_keys(tasks_dir: str) -> set:
    """Build a set of (scene_id, title) keys from the existing pool."""
    keys = set()
    for f in glob.glob(os.path.join(tasks_dir, "*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
            key = (d.get("scene_id", ""), d.get("title", ""))
            keys.add(key)
        except Exception:
            pass
    return keys


def check_gpt52_passed(calibration) -> bool:
    """Check if GPT-5.2 passed in calibration data.

    Handles both old format (dict keyed by model name) and new format
    (list of entries with agent_models).
    """
    if not calibration:
        return False

    # Old format: {"gpt-5.2": {"passed": true, ...}}
    if isinstance(calibration, dict) and "gpt-5.2" in calibration:
        entry = calibration["gpt-5.2"]
        if isinstance(entry, dict):
            if entry.get("passed"):
                return True
            if entry.get("percent_complete", 0) >= 1.0:
                return True
            main_goal = entry.get("main_goal", {})
            if isinstance(main_goal, dict) and main_goal.get("passed"):
                return True
        return False

    # New format: list of entries with agent_models
    entries = calibration if isinstance(calibration, list) else [calibration]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        models = entry.get("agent_models", {})
        if not any("gpt-5.2" in str(v) for v in models.values()):
            continue
        results = entry.get("results", {})
        if results.get("passed"):
            return True
        if results.get("success"):
            return True
        main_goal = results.get("main_goal", {})
        if isinstance(main_goal, dict) and main_goal.get("passed"):
            return True
        if results.get("progress", 0) >= 1.0:
            return True
    return False


def validate_task(data: dict) -> list:
    """Return list of validation errors (empty = valid)."""
    errors = []
    if not data.get("problem_pddl", "").strip():
        errors.append("missing problem_pddl")
    if not data.get("golden_trajectory"):
        errors.append("missing golden_trajectory")
    if not data.get("agent_secrets"):
        errors.append("missing agent_secrets")
    if not data.get("task"):
        errors.append("missing task description")
    if data.get("num_agents", 0) < 2:
        errors.append("fewer than 2 agents")
    return errors


def main():
    bulk_gen_dir = "outputs/bulk_gen_tasks"
    tasks_dir = "data/emtom/tasks"
    only_gpt52_failures = "--only-failures" in sys.argv
    dry_run = "--dry-run" in sys.argv

    print(f"Salvaging tasks from {bulk_gen_dir} -> {tasks_dir}")
    print(f"Filter: only GPT-5.2 failures = {only_gpt52_failures}")
    print(f"Dry run: {dry_run}")
    print()

    existing_keys = load_existing_keys(tasks_dir)
    print(f"Existing tasks: {len(existing_keys)}")

    stats = defaultdict(int)
    salvaged = []

    for bulk_dir in sorted(glob.glob(os.path.join(bulk_gen_dir, "*"))):
        if not os.path.isdir(bulk_dir):
            continue
        for task_file in sorted(glob.glob(os.path.join(bulk_dir, "*.json"))):
            stats["total"] += 1
            try:
                with open(task_file) as f:
                    data = json.load(f)
            except Exception as e:
                stats["json_error"] += 1
                continue

            # Validate structure
            errors = validate_task(data)
            if errors:
                stats["invalid"] += 1
                continue

            # Check for duplicates
            key = (data.get("scene_id", ""), data.get("title", ""))
            if key in existing_keys:
                stats["duplicate"] += 1
                continue

            # Filter: only tasks GPT-5.2 failed
            if only_gpt52_failures:
                cal = data.get("calibration", [])
                if check_gpt52_passed(cal):
                    stats["gpt52_passed"] += 1
                    continue

            # Task is valid and unique — salvage it
            filename = os.path.basename(task_file)
            dest = os.path.join(tasks_dir, filename)

            # Avoid filename collisions
            if os.path.exists(dest):
                base, ext = os.path.splitext(filename)
                dest = os.path.join(tasks_dir, f"{base}_salvaged{ext}")

            if not dry_run:
                shutil.copy2(task_file, dest)

            salvaged.append({
                "source": task_file,
                "dest": dest,
                "category": data.get("category", "unknown"),
                "title": data.get("title", "unknown"),
            })
            existing_keys.add(key)
            stats["salvaged"] += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"Salvage Summary")
    print(f"{'='*60}")
    print(f"Total scanned:     {stats['total']}")
    print(f"JSON errors:       {stats['json_error']}")
    print(f"Invalid structure: {stats['invalid']}")
    print(f"Duplicates:        {stats['duplicate']}")
    if only_gpt52_failures:
        print(f"GPT-5.2 passed:    {stats['gpt52_passed']}")
    print(f"Salvaged:          {stats['salvaged']}")

    # Category breakdown
    from collections import Counter
    cats = Counter(s["category"] for s in salvaged)
    print(f"\nBy category:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    if dry_run:
        print("\n[DRY RUN — no files copied]")


if __name__ == "__main__":
    main()
