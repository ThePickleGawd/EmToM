#!/usr/bin/env python3
"""Scan benchmark outputs and generate static JSON data for the visualizer."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "emtom"
DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"


def make_relative_path(abs_path: str) -> str:
    """Convert absolute path to web-relative path under /outputs/."""
    prefix = str(PROJECT_ROOT) + "/"
    if abs_path.startswith(prefix):
        return "/" + abs_path[len(prefix):]
    return abs_path


def process_action(entry: dict) -> dict:
    """Extract relevant fields from an action_history entry."""
    frame_paths = [make_relative_path(p) for p in entry.get("selected_frame_paths", [])]
    out = {
        "turn": entry.get("turn"),
        "sim_step": entry.get("sim_step"),
        "agent": entry.get("agent"),
        "action": entry.get("action", ""),
        "result": entry.get("result", ""),
        "skill_steps": entry.get("skill_steps", 0),
        "selected_frames": entry.get("selected_frames", []),
        "frame_paths": frame_paths,
    }
    if entry.get("thought"):
        out["thought"] = entry["thought"]
    return out


def process_task_dir(task_dir: Path) -> Optional[dict]:
    """Read a task's planner-log and return cleaned task data."""
    log_dir = task_dir / "planner-log"
    if not log_dir.exists():
        return None

    log_files = sorted(log_dir.glob("planner-log-*.json"))
    if not log_files:
        return None

    log_file = log_files[-1]  # use the latest
    try:
        with open(log_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    actions = [process_action(a) for a in data.get("action_history", [])]

    return {
        "task_id": data.get("task_id", task_dir.name),
        "task_title": data.get("task_title", ""),
        "instruction": data.get("instruction", {}),
        "mechanics_active": data.get("mechanics_active", []),
        "steps": data.get("steps", 0),
        "turns": data.get("turns", 0),
        "done": data.get("done", False),
        "success": data.get("success", False),
        "llm_agents": data.get("llm_agents", []),
        "human_agents": data.get("human_agents", []),
        "action_history": actions,
    }


def process_run(run_dir: Path) -> Optional[dict]:
    """Process a benchmark run directory."""
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return None

    # Read summary if available
    summary_file = results_dir / "benchmark_summary.json"
    summary = {}
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    run_id = run_dir.name
    tasks_dir = DATA_DIR / "tasks" / run_id
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_summaries = []
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith(".") or task_dir.name == "benchmark_summary.json":
            continue

        task_data = process_task_dir(task_dir)
        if task_data is None:
            continue

        # Write per-task detail file
        task_file = tasks_dir / f"{task_data['task_id']}.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)

        # Find category from summary results
        category = ""
        if "results" in summary:
            for r in summary["results"]:
                if r.get("task_id") == task_data["task_id"]:
                    category = r.get("category", "")
                    break

        task_summaries.append({
            "task_id": task_data["task_id"],
            "title": task_data["task_title"],
            "category": category,
            "success": task_data["success"],
            "turns": task_data["turns"],
            "steps": task_data["steps"],
            "agents": len(task_data["llm_agents"]),
        })

    if not task_summaries:
        return None

    return {
        "id": run_id,
        "model": summary.get("model", ""),
        "observation_mode": summary.get("benchmark_observation_mode", ""),
        "total": summary.get("total", len(task_summaries)),
        "passed": summary.get("passed", sum(1 for t in task_summaries if t["success"])),
        "pass_rate": summary.get("pass_rate", 0),
        "tasks": task_summaries,
    }


def main():
    if not OUTPUTS_DIR.exists():
        print(f"Error: {OUTPUTS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    run_dirs = sorted(
        [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and "benchmark" in d.name],
        reverse=True,
    )

    print(f"Found {len(run_dirs)} benchmark directories")

    for run_dir in run_dirs:
        print(f"  Processing {run_dir.name}...", end=" ")
        run_data = process_run(run_dir)
        if run_data:
            runs.append(run_data)
            print(f"{len(run_data['tasks'])} tasks")
        else:
            print("skipped")

    runs_file = DATA_DIR / "runs.json"
    with open(runs_file, "w") as f:
        json.dump({"runs": runs}, f)

    total_tasks = sum(len(r["tasks"]) for r in runs)
    print(f"\nDone: {len(runs)} runs, {total_tasks} tasks")
    print(f"Output: {runs_file}")


if __name__ == "__main__":
    main()
