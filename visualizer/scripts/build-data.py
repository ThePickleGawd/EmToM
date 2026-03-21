#!/usr/bin/env python3
"""Scan benchmark outputs and generate static JSON data for benchmark views.

Generation data is intentionally live-only in dev via the Vite filesystem
plugin. It is not indexed here.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "emtom"
TASKS_DIR = PROJECT_ROOT / "data" / "emtom" / "tasks"
DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"


WORKSPACE_RE = re.compile(
    r"tmp/task_gen/(?P<workspace>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-[a-z]+-[0-9a-f]{8})"
)
WORKER_RE = re.compile(r"gpu(?P<gpu>\d+)_slot(?P<slot>\d+)_(?P<category>\w+)\.log")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_run_timestamp(name: str) -> str:
    try:
        return datetime.strptime(name[:19], "%Y-%m-%d_%H-%M-%S").isoformat()
    except ValueError:
        return ""


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


def make_repo_relative_path(path: str | Path) -> str:
    path_str = str(path)
    prefix = str(PROJECT_ROOT) + "/"
    if path_str.startswith(prefix):
        return path_str[len(prefix):]
    return path_str


def normalize_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarize_log_excerpt(text: str, head_lines: int = 40, tail_lines: int = 60) -> dict:
    lines = text.splitlines()
    return {
        "head": lines[:head_lines],
        "tail": lines[-tail_lines:] if len(lines) > tail_lines else lines[:],
        "line_count": len(lines),
    }


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def calibration_result(task_data: dict) -> bool:
    calibration = task_data.get("calibration", [])
    standard = None
    if isinstance(calibration, list):
        for entry in calibration:
            if str(entry.get("run_mode", "standard")) == "standard":
                standard = entry
                break
        if standard is None and calibration:
            standard = calibration[0]
    if not standard:
        return False
    results = standard.get("results", {})
    if "main_goal" in results:
        return bool(results.get("main_goal", {}).get("passed", False))
    if "winner" in results:
        return results.get("winner") is not None
    return bool(results.get("passed", False))


def load_submitted_task_summary(path_str: str) -> Optional[dict]:
    task_path = normalize_path(path_str)
    task_data = read_json(task_path)
    if task_data is None:
        return None
    stat = task_path.stat()
    return {
        "task_id": task_data.get("task_id", task_path.stem),
        "title": task_data.get("title", task_path.stem),
        "category": task_data.get("category", ""),
        "tom_level": task_data.get("tom_level"),
        "success": calibration_result(task_data),
        "path": make_repo_relative_path(task_path),
        "submitted_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def summarize_event(event: dict) -> dict:
    data = event.get("data")
    if not isinstance(data, dict):
        data = {}
    summary = {
        "timestamp": event.get("timestamp", ""),
        "event_type": event.get("event_type", ""),
        "command": event.get("command"),
        "success": event.get("success"),
        "error": event.get("error"),
        "agent_name": event.get("agent_name"),
        "model": event.get("model"),
        "reason": event.get("reason"),
        "num_agents": event.get("num_agents"),
        "keep": event.get("keep"),
        "message": data.get("message"),
        "scene_id": data.get("scene_id"),
        "episode_id": data.get("episode_id"),
        "output_path": data.get("output_path"),
        "submitted_count": data.get("submitted_count"),
        "next_required_k_level": data.get("next_required_k_level"),
        "return_code": event.get("return_code"),
        "finished": event.get("finished"),
        "failed": event.get("failed"),
        "fail_reason": event.get("fail_reason"),
    }
    return {k: v for k, v in summary.items() if v not in (None, "", [], {})}


def parse_mini_trace(workspace_dir: Path) -> dict:
    trajectory_path = workspace_dir / "mini_trajectory.json"
    if not trajectory_path.exists():
        return {"entries": [], "stats": None}

    data = read_json(trajectory_path)
    if data is None:
        return {"entries": [], "stats": None}

    entries: list[dict[str, Any]] = []
    for index, message in enumerate(data.get("messages", []), start=1):
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                entries.append(
                    {
                        "index": index,
                        "kind": "assistant",
                        "content": content.strip(),
                    }
                )
            for tool_call in message.get("tool_calls", []) or []:
                arguments = tool_call.get("function", {}).get("arguments", "")
                command = ""
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                        command = parsed.get("command", "")
                    except json.JSONDecodeError:
                        command = arguments
                entries.append(
                    {
                        "index": index,
                        "kind": "tool_call",
                        "tool": tool_call.get("function", {}).get("name", ""),
                        "command": command,
                    }
                )
        elif role == "tool":
            content = message.get("content")
            if isinstance(content, dict):
                output = content.get("output")
                if output is None:
                    output = "\n".join(
                        part
                        for part in [content.get("output_head", ""), content.get("output_tail", "")]
                        if part
                    )
                entries.append(
                    {
                        "index": index,
                        "kind": "tool_result",
                        "returncode": content.get("returncode"),
                        "output": str(output or "")[:4000],
                    }
                )

    stats = data.get("info", {}).get("model_stats", {})
    return {
        "entries": entries,
        "stats": {
            "api_calls": stats.get("api_calls"),
            "instance_cost": stats.get("instance_cost"),
        },
    }


def parse_worker_workspace(log_text: str) -> Optional[Path]:
    match = WORKSPACE_RE.search(log_text)
    if not match:
        return None
    return PROJECT_ROOT / "tmp" / "task_gen" / match.group("workspace")


def parse_worker(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    clean_text = strip_ansi(text)
    match = WORKER_RE.match(log_path.name)
    gpu = int(match.group("gpu")) if match else -1
    slot = int(match.group("slot")) if match else -1
    category = match.group("category") if match else "unknown"

    workspace_dir = parse_worker_workspace(text)
    workspace_state = (
        read_json(workspace_dir / "taskgen_state.json") if workspace_dir and (workspace_dir / "taskgen_state.json").exists() else {}
    ) or {}
    event_log = (
        [summarize_event(event) for event in read_jsonl(workspace_dir / "taskgen_events.jsonl")]
        if workspace_dir
        else []
    )
    mini_trace = parse_mini_trace(workspace_dir) if workspace_dir else {"entries": [], "stats": None}

    submitted_tasks = []
    for submitted_path in workspace_state.get("submitted_tasks", []):
        summary = load_submitted_task_summary(submitted_path)
        if summary:
            submitted_tasks.append(summary)
    if not submitted_tasks:
        for match in re.finditer(r"^\s*-\s+(.*\.json)\s*$", clean_text, re.MULTILINE):
            summary = load_submitted_task_summary(match.group(1).strip())
            if summary:
                submitted_tasks.append(summary)

    status = "running"
    if workspace_state.get("failed"):
        status = "failed"
    elif workspace_state.get("finished"):
        status = "finished"
    elif "Agent FAILED:" in clean_text:
        status = "failed"
    elif "Generation Complete" in clean_text:
        status = "finished"
    elif "Generation Result" in clean_text:
        status = "stopped"

    fail_reason = workspace_state.get("fail_reason", "")
    if not fail_reason:
        failure_match = re.findall(r"Agent FAILED:\s*(.+)", clean_text)
        if failure_match:
            fail_reason = failure_match[-1].strip()

    target_tasks = workspace_state.get("num_tasks_target")
    if not target_tasks:
        target_match = re.search(r"Target tasks:\s*(\d+)", clean_text)
        target_tasks = int(target_match.group(1)) if target_match else 0

    return {
        "id": log_path.stem,
        "gpu": gpu,
        "slot": slot,
        "category": category,
        "status": status,
        "workspace_id": workspace_dir.name if workspace_dir else "",
        "workspace_path": make_repo_relative_path(workspace_dir) if workspace_dir else "",
        "worker_log_path": make_repo_relative_path(log_path),
        "task_gen_agent": workspace_state.get("task_gen_agent"),
        "task_gen_model": workspace_state.get("task_gen_model"),
        "submitted_count": len(submitted_tasks),
        "target_tasks": target_tasks,
        "current_task_index": workspace_state.get("current_task_index"),
        "current_k_level": workspace_state.get("current_k_level"),
        "scene_id": workspace_state.get("scene_id"),
        "episode_id": workspace_state.get("episode_id"),
        "finished": bool(workspace_state.get("finished")),
        "failed": bool(workspace_state.get("failed")) or status == "failed",
        "fail_reason": fail_reason,
        "submitted_tasks": submitted_tasks,
        "events": event_log,
        "agent_trace": mini_trace["entries"],
        "agent_stats": mini_trace["stats"],
        "log_excerpt": summarize_log_excerpt(text),
    }


def build_success_series(workers: list[dict]) -> list[dict]:
    timeline = []
    for worker in workers:
        for task in worker.get("submitted_tasks", []):
            timeline.append(
                {
                    "timestamp": task.get("submitted_at", ""),
                    "task_id": task.get("task_id", ""),
                    "title": task.get("title", ""),
                    "category": task.get("category", ""),
                    "success": bool(task.get("success")),
                    "worker_id": worker.get("id", ""),
                }
            )
    timeline.sort(key=lambda item: (item["timestamp"], item["task_id"]))
    passed = 0
    series = []
    for index, item in enumerate(timeline, start=1):
        if item["success"]:
            passed += 1
        series.append(
            {
                **item,
                "index": index,
                "cumulative_pass_rate": passed / index,
                "cumulative_passed": passed,
            }
        )
    return series


def process_generation_run(log_dir: Path) -> Optional[dict]:
    worker_logs = sorted(log_dir.glob("gpu*_slot*_*.log"))
    if not worker_logs:
        return None

    workers = [parse_worker(log_path) for log_path in worker_logs]
    success_series = build_success_series(workers)
    launcher_log = log_dir.parent / f"{log_dir.name}-launcher.log"

    requested_tasks = sum(int(worker.get("target_tasks") or 0) for worker in workers)
    submitted_tasks = sum(len(worker.get("submitted_tasks", [])) for worker in workers)
    finished_workers = sum(1 for worker in workers if worker.get("status") == "finished")
    failed_workers = sum(1 for worker in workers if worker.get("status") == "failed")
    running_workers = sum(1 for worker in workers if worker.get("status") == "running")

    categories = sorted({worker.get("category", "") for worker in workers if worker.get("category")})
    detail = {
        "id": log_dir.name,
        "started_at": parse_run_timestamp(log_dir.name),
        "log_dir": make_repo_relative_path(log_dir),
        "launcher_log": make_repo_relative_path(launcher_log) if launcher_log.exists() else "",
        "total_workers": len(workers),
        "requested_tasks": requested_tasks,
        "submitted_tasks": submitted_tasks,
        "finished_workers": finished_workers,
        "failed_workers": failed_workers,
        "running_workers": running_workers,
        "categories": categories,
        "success_series": success_series,
        "workers": workers,
    }
    return detail


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
    evaluation = data.get("evaluation", {})

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
        "literal_tom_probe_results": evaluation.get("literal_tom_probe_results", []),
    }


def flatten_calibration_entry(cal: dict) -> list:
    """Flatten one calibration trajectory into action_history format."""
    entries = []
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


def calibration_progress(cal: dict) -> float:
    """Extract progress from a calibration entry across task categories."""
    results = cal.get("results", {})
    if "main_goal" in results:
        return results.get("main_goal", {}).get("progress", 0)
    if "teams" in results:
        return max((team.get("progress", 0) for team in results.get("teams", {}).values()), default=0)
    return results.get("progress", 0)


def calibration_passed(cal: dict) -> bool:
    """Extract pass/fail from a calibration entry across task categories."""
    results = cal.get("results", {})
    if "main_goal" in results:
        return results.get("main_goal", {}).get("passed", False)
    if "winner" in results:
        return results.get("winner") is not None
    return results.get("passed", False)


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
    golden_history = flatten_golden(data.get("golden_trajectory", []))
    calibration_by_mode = {}
    for cal in calibration:
        run_mode = str(cal.get("run_mode", "standard") or "standard")
        calibration_by_mode[run_mode] = {
            "run_mode": run_mode,
            "tested_at": cal.get("tested_at", ""),
            "agent_models": cal.get("agent_models", {}),
            "passed": calibration_passed(cal),
            "progress": calibration_progress(cal),
            "steps": cal.get("steps", 0),
            "turns": len(cal.get("trajectory", [])),
            "trajectory": flatten_calibration_entry(cal),
        }

    default_mode = "standard" if "standard" in calibration_by_mode else next(iter(calibration_by_mode), None)
    default_cal = calibration_by_mode.get(default_mode) if default_mode else None
    cal_history = default_cal.get("trajectory", []) if default_cal else []
    cal_passed = default_cal.get("passed", False) if default_cal else False
    cal_steps = default_cal.get("steps", 0) if default_cal else 0
    cal_turns = default_cal.get("turns", 0) if default_cal else 0

    return {
        "task_id": task_id,
        "task_title": data.get("title", ""),
        "task_description": data.get("task", ""),
        "category": data.get("category", ""),
        "instruction": instruction,
        "mechanics_active": mechanics,
        "steps": cal_steps,
        "turns": cal_turns,
        "done": True,
        "success": cal_passed,
        "llm_agents": agents,
        "human_agents": [],
        "action_history": cal_history,
        "golden_trajectory": golden_history,
        "problem_pddl": data.get("problem_pddl", ""),
        "tom_level": data.get("tom_level"),
        "tom_reasoning": data.get("tom_reasoning"),
        "calibration_meta": default_cal,
        "calibration_by_mode": calibration_by_mode,
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
