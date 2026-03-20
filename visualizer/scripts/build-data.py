#!/usr/bin/env python3
"""Scan benchmark outputs and generate static JSON data for the visualizer."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "emtom"
TASKS_DIR = PROJECT_ROOT / "data" / "emtom" / "tasks"
DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"


def normalize_secrets_text(secrets) -> str:
    if isinstance(secrets, list):
        return "\n".join(s for s in secrets if isinstance(s, str))
    if isinstance(secrets, str):
        return secrets
    return ""


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


def flatten_calibration(calibration: list) -> list:
    """Flatten calibration trajectory into action_history format."""
    entries = []
    for cal in calibration:
        for turn_data in cal.get("trajectory", []):
            turn = turn_data.get("turn", 0)
            for agent_id, agent_data in turn_data.get("agents", {}).items():
                entry = {
                    "turn": turn,
                    "sim_step": turn,
                    "agent": agent_id,
                    "action": agent_data.get("action", ""),
                    "result": agent_data.get("observation", ""),
                    "skill_steps": 0,
                    "selected_frames": [],
                    "frame_paths": [],
                }
                if agent_data.get("thought"):
                    entry["thought"] = agent_data["thought"]
                entries.append(entry)
    return entries


def flatten_golden(golden: list) -> list:
    """Flatten golden_trajectory into action_history format."""
    entries = []
    for step_idx, step in enumerate(golden):
        for agent_action in step.get("actions", []):
            entries.append({
                "turn": step_idx + 1,
                "sim_step": step_idx + 1,
                "agent": agent_action.get("agent", ""),
                "action": agent_action.get("action", ""),
                "result": "",
                "skill_steps": 0,
                "selected_frames": [],
                "frame_paths": [],
            })
    return entries


def process_task_file(task_file: Path) -> Optional[dict]:
    """Process a task JSON from data/emtom/tasks/."""
    try:
        with open(task_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    task_id = data.get("task_id", task_file.stem)
    agents = [f"agent_{i}" for i in range(data.get("num_agents", 2))]

    # Build instruction from agent_secrets
    instruction = {}
    for agent_id, secrets in data.get("agent_secrets", {}).items():
        instruction[agent_id] = normalize_secrets_text(secrets)

    # Extract mechanic types from bindings
    mechanics = list({b.get("mechanic_type", "") for b in data.get("mechanic_bindings", []) if b.get("mechanic_type")})

    calibration = data.get("calibration", [])
    cal_history = flatten_calibration(calibration) if calibration else []
    golden_history = flatten_golden(data.get("golden_trajectory", []))

    cal_passed = False
    cal_steps = 0
    if calibration:
        cal_passed = calibration[-1].get("results", {}).get("passed", False)
        cal_steps = calibration[-1].get("steps", 0)

    return {
        "task_id": task_id,
        "task_title": data.get("title", ""),
        "task_description": data.get("task", ""),
        "category": data.get("category", ""),
        "instruction": instruction,
        "mechanics_active": mechanics,
        "steps": cal_steps,
        "turns": len(calibration[-1].get("trajectory", [])) if calibration else 0,
        "done": True,
        "success": cal_passed,
        "llm_agents": agents,
        "human_agents": [],
        "action_history": cal_history,
        "golden_trajectory": golden_history,
        "problem_pddl": data.get("problem_pddl", ""),
        "tom_level": data.get("tom_level"),
        "tom_reasoning": data.get("tom_reasoning"),
        "calibration_meta": {
            "tested_at": calibration[-1].get("tested_at", "") if calibration else "",
            "agent_models": calibration[-1].get("agent_models", {}) if calibration else {},
            "passed": cal_passed,
            "progress": calibration[-1].get("results", {}).get("progress", 0) if calibration else 0,
        } if calibration else None,
    }


def find_results_dirs(run_dir: Path) -> list:
    """Find all results directories in a benchmark run, handling two layouts.

    Flat:   {run_dir}/results/{task_id}/planner-log/...
    Nested: {run_dir}/{wrapper}/benchmark-*/results/{task_id}/planner-log/...
    """
    # Flat layout
    flat = run_dir / "results"
    if flat.exists() and flat.is_dir():
        summary = {}
        sf = flat / "benchmark_summary.json"
        if sf.exists():
            try:
                with open(sf) as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return [(flat, summary)]

    # Nested layout
    out = []
    for wrapper in sorted(run_dir.iterdir()):
        if not wrapper.is_dir() or wrapper.name.startswith(".") or wrapper.name == "logs":
            continue
        for bd in sorted(wrapper.iterdir()):
            if not bd.is_dir() or not bd.name.startswith("benchmark-"):
                continue
            results_dir = bd / "results"
            if not results_dir.exists() or not results_dir.is_dir():
                continue
            summary = {}
            sf = results_dir / "benchmark_summary.json"
            if sf.exists():
                try:
                    with open(sf) as f:
                        summary = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
            out.append((results_dir, summary))
    return out


def process_run(run_dir: Path) -> Optional[dict]:
    """Process a benchmark run directory."""
    results_dirs = find_results_dirs(run_dir)
    if not results_dirs:
        return None

    run_id = run_dir.name
    tasks_dir = DATA_DIR / "tasks" / run_id
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_summaries = []
    merged_summary = {}

    for results_dir, summary in results_dirs:
        if not merged_summary.get("model") and summary.get("model"):
            merged_summary = summary

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

    passed = sum(1 for t in task_summaries if t["success"])
    return {
        "id": run_id,
        "model": merged_summary.get("model", ""),
        "observation_mode": merged_summary.get("benchmark_observation_mode", ""),
        "total": len(task_summaries),
        "passed": passed,
        "pass_rate": (passed / len(task_summaries) * 100) if task_summaries else 0,
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

    # ── Task library from data/emtom/tasks/ ──
    library_tasks = []
    if TASKS_DIR.exists():
        lib_dir = DATA_DIR / "tasks" / "_library"
        lib_dir.mkdir(parents=True, exist_ok=True)

        task_files = sorted(TASKS_DIR.glob("*.json"), reverse=True)
        print(f"\nFound {len(task_files)} task library files")

        for tf in task_files:
            task_data = process_task_file(tf)
            if task_data is None:
                continue

            out_file = lib_dir / f"{task_data['task_id']}.json"
            with open(out_file, "w") as f:
                json.dump(task_data, f)

            library_tasks.append({
                "task_id": task_data["task_id"],
                "title": task_data["task_title"],
                "category": task_data["category"],
                "success": task_data["success"],
                "turns": task_data["turns"],
                "steps": task_data["steps"],
                "agents": len(task_data["llm_agents"]),
            })

        print(f"  Processed {len(library_tasks)} tasks")

    runs_file = DATA_DIR / "runs.json"
    with open(runs_file, "w") as f:
        json.dump({"runs": runs, "library": library_tasks}, f)

    total_tasks = sum(len(r["tasks"]) for r in runs)
    print(f"\nDone: {len(runs)} runs, {total_tasks} benchmark tasks, {len(library_tasks)} library tasks")
    print(f"Output: {runs_file}")


if __name__ == "__main__":
    main()
