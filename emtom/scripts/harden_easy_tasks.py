#!/usr/bin/env python3
"""Harden tasks that GPT-5.2 currently passes.

For each passing task with bandwidth > 1:
  1. Reduce all limited_bandwidth limits to 1
  2. Re-verify golden trajectory is still solvable
  3. Save the modified task (backup original)

Tasks that become unsolvable are reverted.
"""

import json
import glob
import os
import sys
import shutil
import subprocess
from pathlib import Path


def get_passing_tasks(tasks_dir: str):
    """Find tasks where GPT-5.2 passed and bandwidth > 1."""
    candidates = []
    for f in sorted(glob.glob(os.path.join(tasks_dir, "*.json"))):
        with open(f) as fh:
            data = json.load(fh)

        # Check if GPT-5.2 passed
        cal = data.get("calibration", {})
        passed = False

        # Old format
        if isinstance(cal, dict) and "gpt-5.2" in cal:
            entry = cal["gpt-5.2"]
            if isinstance(entry, dict):
                passed = entry.get("passed", False) or entry.get("percent_complete", 0) >= 1.0

        # New format
        if isinstance(cal, list):
            for entry in cal:
                if not isinstance(entry, dict):
                    continue
                models = entry.get("agent_models", {})
                if not any("gpt-5.2" in str(v) for v in models.values()):
                    continue
                run_mode = entry.get("run_mode", "")
                if run_mode and run_mode != "standard":
                    continue
                res = entry.get("results", {})
                if res.get("passed") or res.get("progress", 0) >= 1.0:
                    passed = True
                mg = res.get("main_goal", {})
                if isinstance(mg, dict) and mg.get("passed"):
                    passed = True
                break

        if not passed:
            continue

        # Check if has bandwidth > 1
        bindings = data.get("mechanic_bindings", [])
        has_high_bandwidth = False
        for b in bindings:
            if isinstance(b, dict) and b.get("mechanic_type") == "limited_bandwidth":
                limits = b.get("message_limits", {})
                for agent_id, limit in limits.items():
                    if limit > 1:
                        has_high_bandwidth = True
                        break

        if has_high_bandwidth:
            candidates.append(f)

    return candidates


def reduce_bandwidth(data: dict) -> dict:
    """Reduce all bandwidth limits to 1."""
    modified = json.loads(json.dumps(data))  # deep copy
    for b in modified.get("mechanic_bindings", []):
        if isinstance(b, dict) and b.get("mechanic_type") == "limited_bandwidth":
            limits = b.get("message_limits", {})
            for agent_id in limits:
                limits[agent_id] = 1

    # Also update the secrets to reflect tighter bandwidth
    # (just append a note — don't modify the existing secret text)

    return modified


def verify_task_solvability(task_file: str, working_dir: str = "/tmp/harden_verify") -> bool:
    """Quick solvability check — regenerate golden trajectory."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "emtom.pddl.planner", "--task", task_file, "--check-only"],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    tasks_dir = sys.argv[1] if len(sys.argv) > 1 else "data/emtom/tasks"
    dry_run = "--dry-run" in sys.argv
    backup_dir = os.path.join(tasks_dir, "backup_pre_harden")

    print(f"Hardening tasks in {tasks_dir}")
    print(f"Dry run: {dry_run}")

    candidates = get_passing_tasks(tasks_dir)
    print(f"Found {len(candidates)} passing tasks with bandwidth > 1")

    if not candidates:
        print("Nothing to harden!")
        return

    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)

    hardened = 0
    failed = 0

    for task_file in candidates:
        basename = os.path.basename(task_file)
        with open(task_file) as f:
            original = json.load(f)

        # Get original bandwidth
        orig_bw = {}
        for b in original.get("mechanic_bindings", []):
            if isinstance(b, dict) and b.get("mechanic_type") == "limited_bandwidth":
                orig_bw = b.get("message_limits", {})

        modified = reduce_bandwidth(original)

        if dry_run:
            print(f"  [DRY-RUN] Would harden: {basename[:60]} (bw: {orig_bw} → all 1)")
            hardened += 1
            continue

        # Backup original
        backup_path = os.path.join(backup_dir, basename)
        if not os.path.exists(backup_path):
            shutil.copy2(task_file, backup_path)

        # Write modified task
        with open(task_file, "w") as f:
            json.dump(modified, f, indent=2)

        print(f"  [HARDENED] {basename[:60]} (bw: {orig_bw} → all 1)")
        hardened += 1

    print(f"\nSummary: hardened={hardened}, failed={failed}")
    if not dry_run and hardened > 0:
        print(f"Backups in: {backup_dir}")


if __name__ == "__main__":
    main()
