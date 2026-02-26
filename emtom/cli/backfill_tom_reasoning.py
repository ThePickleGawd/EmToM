"""
Backfill tom_reasoning for existing tasks using LLM-generated explanations.

Usage:
    python -m emtom.cli.backfill_tom_reasoning [--tasks-dir DIR] [--model MODEL] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def backfill(
    tasks_dir: str = "data/emtom/tasks",
    model: str = "gpt-5.2",
    dry_run: bool = False,
) -> None:
    """Re-generate tom_reasoning for all tasks in tasks_dir."""
    from emtom.pddl.tom_verifier import generate_tom_reasoning

    tasks_path = Path(tasks_dir)
    if not tasks_path.is_dir():
        print(f"Error: {tasks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(tasks_path.glob("*.json"))
    print(f"Found {len(task_files)} tasks in {tasks_dir}")

    updated = 0
    skipped = 0
    failed = 0

    for task_file in task_files:
        try:
            with open(task_file) as f:
                task_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {task_file.name}: {e}")
            skipped += 1
            continue

        tom_level = task_data.get("tom_level")
        if not isinstance(tom_level, int) or tom_level < 1:
            print(f"  SKIP {task_file.name}: no valid tom_level ({tom_level!r})")
            skipped += 1
            continue

        # Build information_gaps from mechanic_bindings
        information_gaps = []
        for b in task_data.get("mechanic_bindings", []):
            if b.get("mechanic_type") == "room_restriction":
                agents = b.get("for_agents", [])
                rooms = b.get("restricted_rooms", [])
                for agent in agents:
                    information_gaps.append(f"{agent} cannot see rooms: {sorted(rooms)}")

        old_reasoning = task_data.get("tom_reasoning", "")
        title = task_data.get("title", task_file.name)

        try:
            new_reasoning = generate_tom_reasoning(
                task_data,
                tom_level=tom_level,
                information_gaps=information_gaps,
                model=model,
            )
        except Exception as e:
            print(f"  FAIL {task_file.name}: {e}")
            failed += 1
            continue

        if dry_run:
            print(f"  [{tom_level}] {title}")
            print(f"       OLD: {old_reasoning[:100]}...")
            print(f"       NEW: {new_reasoning[:100]}...")
        else:
            task_data["tom_reasoning"] = new_reasoning
            with open(task_file, "w") as f:
                json.dump(task_data, f, indent=2)
                f.write("\n")
            print(f"  OK [{tom_level}] {title}")

        updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill tom_reasoning with LLM explanations")
    parser.add_argument("--tasks-dir", default="data/emtom/tasks", help="Directory with task JSON files")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model for reasoning generation")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    backfill(tasks_dir=args.tasks_dir, model=args.model, dry_run=args.dry_run)
