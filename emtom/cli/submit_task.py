"""
Submit a validated task to the output directory.

Validates structure, checks goal/agent counts, generates filename,
and copies to output directories.

Gate checks (verify/judge/test passed) are NOT enforced here — they are
agent-level state managed by agent.py. This module only handles the
file operations and final validation.

Usage:
    # CLI
    python -m emtom.cli.submit_task task.json --output-dir DIR [--working-dir DIR]

    # Programmatic
    from emtom.cli.submit_task import run
    result = run("task.json", output_dir="data/emtom/tasks")
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from emtom.cli import CLIResult, failure, success


def run(
    task_file: str,
    output_dir: str,
    working_dir: str = None,
    scene_file: str = None,
    submitted_dir: str = None,
    subtasks_min: int = 3,
    subtasks_max: int = 20,
    agents_min: int = 2,
    agents_max: int = 10,
) -> CLIResult:
    """
    Submit a task to the output directory.

    Args:
        task_file: Path to task JSON file.
        output_dir: Permanent output directory.
        working_dir: Optional working directory (for scene data).
        scene_file: Optional explicit scene data JSON file.
        submitted_dir: Optional session-scoped submitted_tasks directory.
        subtasks_min: Minimum PDDL conjuncts or subtasks.
        subtasks_max: Maximum PDDL conjuncts or subtasks.
        agents_min: Minimum agent count.
        agents_max: Maximum agent count.

    Returns:
        CLIResult with data keys: output_path, filename, submitted_path.
    """
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")

    # Validate task structure
    from emtom.cli.validate_task import run as validate_run

    validation = validate_run(task_file, working_dir=working_dir, scene_file=scene_file)
    if not validation["success"]:
        return validation

    # Validate goal count (PDDL conjuncts or legacy subtasks)
    if task_data.get("pddl_goal"):
        from emtom.pddl.dsl import parse_goal_string

        try:
            goal = parse_goal_string(task_data["pddl_goal"])
            num_goals = len(goal.flatten())
        except Exception:
            num_goals = 0
        if num_goals < subtasks_min:
            return failure(
                f"Task has {num_goals} PDDL goal conjuncts, minimum is {subtasks_min}.",
                data={"current": num_goals, "required_min": subtasks_min},
            )
        if num_goals > subtasks_max:
            return failure(
                f"Task has {num_goals} PDDL goal conjuncts, maximum is {subtasks_max}.",
                data={"current": num_goals, "required_max": subtasks_max},
            )
    else:
        subtasks = task_data.get("subtasks", [])
        num_subtasks = len(subtasks)
        if num_subtasks < subtasks_min:
            return failure(
                f"Task has {num_subtasks} subtasks, minimum is {subtasks_min}.",
                data={"current": num_subtasks, "required_min": subtasks_min},
            )
        if num_subtasks > subtasks_max:
            return failure(
                f"Task has {num_subtasks} subtasks, maximum is {subtasks_max}.",
                data={"current": num_subtasks, "required_max": subtasks_max},
            )

    # Validate agent count
    num_agents = task_data.get("num_agents", 2)
    if num_agents < agents_min:
        return failure(
            f"Task has {num_agents} agents, minimum is {agents_min}.",
            data={"current": num_agents, "required_min": agents_min},
        )
    if num_agents > agents_max:
        return failure(
            f"Task has {num_agents} agents, maximum is {agents_max}.",
            data={"current": num_agents, "required_max": agents_max},
        )

    # Generate filename: {datetime}_{title_slug}.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    title = task_data.get("title", "untitled")
    title_slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:50]
    output_filename = f"{timestamp}_{title_slug}.json"

    # Copy to main output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / output_filename
    shutil.copy(task_path, output_path)

    result_data = {
        "output_path": str(output_path),
        "filename": output_filename,
        "title": task_data.get("title", "untitled"),
    }

    # Also copy to submitted_tasks/ for session tracking
    if submitted_dir:
        sub_dir = Path(submitted_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)
        submitted_path = sub_dir / output_filename
        shutil.copy(task_path, submitted_path)
        result_data["submitted_path"] = str(submitted_path)

    return success(result_data)


if __name__ == "__main__":
    import argparse

    from emtom.cli import print_result

    parser = argparse.ArgumentParser(description="Submit a validated task")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--output-dir", required=True, help="Permanent output directory")
    parser.add_argument("--working-dir", default=None, help="Working directory (for scene data)")
    parser.add_argument("--scene-file", default=None, help="Scene data JSON file")
    parser.add_argument("--submitted-dir", default=None, help="Session submitted_tasks directory")
    parser.add_argument("--subtasks-min", type=int, default=3)
    parser.add_argument("--subtasks-max", type=int, default=20)
    parser.add_argument("--agents-min", type=int, default=2)
    parser.add_argument("--agents-max", type=int, default=10)
    args = parser.parse_args()

    result = run(
        args.task_file,
        output_dir=args.output_dir,
        working_dir=args.working_dir,
        scene_file=args.scene_file,
        submitted_dir=args.submitted_dir,
        subtasks_min=args.subtasks_min,
        subtasks_max=args.subtasks_max,
        agents_min=args.agents_min,
        agents_max=args.agents_max,
    )
    print_result(result)
