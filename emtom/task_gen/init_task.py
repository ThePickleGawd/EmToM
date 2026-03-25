#!/usr/bin/env python3
"""
Scene loader + task initializer for hand-crafting EmToM tasks.

Loads a PARTNR scene (random or specific), creates a working_task.json
pre-filled with scene data, and saves the full scene inventory.

Usage:
    python emtom/task_gen/init_task.py \
        --agents 2 \
        --output-dir /tmp/task_workdir \
        [--scene-id 102344280]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent

from emtom.task_gen.task_bootstrap import build_scene_bootstrap_problem_pddl


def get_config_name(num_agents: int) -> str:
    """Get Hydra config name for agent count."""
    if not 2 <= num_agents <= 10:
        print(f"Error: --agents must be 2-10, got {num_agents}", file=sys.stderr)
        sys.exit(1)
    return f"examples/emtom_{num_agents}_robots"


def load_scene_subprocess(
    num_agents: int,
    output_dir: str,
    scene_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Load a scene via subprocess (needs fresh GL context)."""
    result_file = tempfile.mktemp(suffix=".json", prefix="scene_result_")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "emtom" / "task_gen" / "load_scene.py"),
        "--result-file", result_file,
        "--working-dir", output_dir,
        "--config-name", get_config_name(num_agents),
    ]
    if scene_id:
        cmd += ["--scene-id", scene_id]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    print(f"Loading scene ({num_agents} agents)...")
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"Error: Scene loading failed (exit code {e.returncode})", file=sys.stderr)
        if os.path.exists(result_file):
            with open(result_file) as f:
                print(json.dumps(json.load(f), indent=2), file=sys.stderr)
        sys.exit(1)

    with open(result_file) as f:
        result = json.load(f)

    os.unlink(result_file)

    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    return result["scene_data"]


def create_working_task(scene_data: dict, num_agents: int) -> dict:
    """Create a working_task.json from template + scene data."""
    template_path = PROJECT_ROOT / "emtom" / "task_gen" / "template" / "template.json"
    with open(template_path) as f:
        task = json.load(f)

    # Fill scene fields
    task["scene_id"] = scene_data["scene_id"]
    task["episode_id"] = scene_data["episode_id"]
    task["agent_spawns"] = scene_data.get("agent_spawns", {})
    task["num_agents"] = num_agents
    task["problem_pddl"] = build_scene_bootstrap_problem_pddl(
        scene_data,
        num_agents,
        problem_name=f"scene_{scene_data['scene_id']}",
    )

    # Generate agent skeletons for the requested agent count
    default_actions = [
        "Navigate", "Open", "Pick", "Place", "UseItem",
        "FindObjectTool", "FindReceptacleTool", "FindRoomTool",
        "Communicate", "Wait",
    ]
    task["agent_secrets"] = {
        f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"] for i in range(num_agents)
    }
    task["agent_actions"] = {
        f"agent_{i}": default_actions.copy() for i in range(num_agents)
    }
    task["golden_trajectory"] = [
        {
            "actions": [
                {
                    "agent": f"agent_{i}",
                    "action": "ACTION_NAME[TARGET]" if i == 0 else "Wait",
                }
                for i in range(num_agents)
            ]
        }
    ]

    return task


def print_scene_summary(scene_data: dict) -> None:
    """Print a human-readable scene summary."""
    print(f"\n{'='*60}")
    print(f"Scene: {scene_data['scene_id']}  (episode {scene_data['episode_id']})")
    print(f"{'='*60}")

    rooms = scene_data.get("rooms", [])
    furniture_in_rooms = scene_data.get("furniture_in_rooms", {})
    objects_on_furniture = scene_data.get("objects_on_furniture", {})

    print(f"\nRooms ({len(rooms)}):")
    for room in sorted(rooms):
        furn_list = furniture_in_rooms.get(room, [])
        print(f"  {room} ({len(furn_list)} furniture)")
        for furn in sorted(furn_list):
            objs = objects_on_furniture.get(furn, [])
            if objs:
                print(f"    {furn}: {', '.join(sorted(objs))}")
            else:
                print(f"    {furn}")

    all_objects = scene_data.get("objects", [])
    articulated = scene_data.get("articulated_furniture", [])
    print(f"\nTotals: {len(rooms)} rooms, {len(scene_data.get('furniture', []))} furniture, "
          f"{len(all_objects)} objects, {len(articulated)} articulated")


def main():
    parser = argparse.ArgumentParser(
        description="Load a scene and create a working task template"
    )
    parser.add_argument(
        "--agents", type=int, default=2,
        help="Number of agents (2-10, default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for working_task.json and current_scene.json",
    )
    parser.add_argument(
        "--scene-id", type=str, default=None,
        help="Specific scene ID to load (default: random)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for scene selection",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load scene via subprocess
    scene_data = load_scene_subprocess(args.agents, args.output_dir, args.scene_id, args.seed)

    # Save full scene inventory
    scene_file = os.path.join(args.output_dir, "current_scene.json")
    with open(scene_file, "w") as f:
        json.dump(scene_data, f, indent=2)
    print(f"Saved scene inventory: {scene_file}")

    # Create and save working task
    task = create_working_task(scene_data, args.agents)
    task_file = os.path.join(args.output_dir, "working_task.json")
    with open(task_file, "w") as f:
        json.dump(task, f, indent=2)
    print(f"Saved working task:    {task_file}")

    # Print scene summary
    print_scene_summary(scene_data)

    print(f"\nNext steps:")
    print(f"  1. Edit {task_file} with your task design")
    print(f"  2. ./emtom/run_emtom.sh verify-static --task {task_file}")
    print(f"  3. ./emtom/run_emtom.sh verify --task {task_file}")
    print(f"  4. ./emtom/run_emtom.sh judge --task {task_file}")


if __name__ == "__main__":
    main()
