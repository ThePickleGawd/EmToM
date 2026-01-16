#!/usr/bin/env python3
"""
Convert EMTOM task JSON files to a CSV summary.

Usage:
    python emtom/scripts/tasks_to_csv.py
    python emtom/scripts/tasks_to_csv.py --input data/emtom/tasks --output tasks_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Maximum number of agent secret columns to support
MAX_AGENTS = 5


def extract_mechanics(task: Dict[str, Any]) -> str:
    """Extract mechanics as a formatted string."""
    mechanics = set()

    # Get from active_mechanics
    if "active_mechanics" in task:
        mechanics.update(task["active_mechanics"])

    # Get from mechanic_bindings
    if "mechanic_bindings" in task:
        for binding in task["mechanic_bindings"]:
            if "mechanic_type" in binding:
                mechanics.add(binding["mechanic_type"])

    return ", ".join(sorted(mechanics)) if mechanics else "none"


def extract_dag(task: Dict[str, Any]) -> str:
    """
    Extract DAG as a visually formatted string.

    Format: Shows tree structure with arrows indicating flow.
    Example:
        [root] trigger_stand
           └─> open_cabinet
               └─> unlock_target
                   └─> place_item
    """
    subtasks = task.get("subtasks", [])
    if not subtasks:
        return "none"

    # Build dependency graph
    subtask_ids = [s.get("id", f"step_{i}") for i, s in enumerate(subtasks)]
    depends_map = {s.get("id", f"step_{i}"): s.get("depends_on", [])
                   for i, s in enumerate(subtasks)}

    # Find roots (no dependencies)
    roots = [sid for sid in subtask_ids if not depends_map.get(sid)]

    # Build reverse map (parent -> children)
    children_map: Dict[str, List[str]] = {sid: [] for sid in subtask_ids}
    for sid, deps in depends_map.items():
        for dep in deps:
            if dep in children_map:
                children_map[dep].append(sid)

    # Format as tree
    lines = []
    visited = set()

    def format_node(node_id: str, depth: int, is_last: bool, prefix: str):
        if node_id in visited:
            return
        visited.add(node_id)

        if depth == 0:
            lines.append(f"[root] {node_id}")
        else:
            connector = "└─>" if is_last else "├─>"
            lines.append(f"{prefix}{connector} {node_id}")

        children = children_map.get(node_id, [])
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            format_node(child, depth + 1, is_last_child, new_prefix)

    for i, root in enumerate(roots):
        format_node(root, 0, i == len(roots) - 1, "")

    # Handle any unvisited nodes (cycles or disconnected)
    for sid in subtask_ids:
        if sid not in visited:
            lines.append(f"[disconnected] {sid}")

    return "\n".join(lines) if lines else "none"


def extract_dag_compact(task: Dict[str, Any]) -> str:
    """
    Extract DAG as a compact single-line format for CSV compatibility.

    Format: root1 → child1 → grandchild1; root2 → child2
    """
    subtasks = task.get("subtasks", [])
    if not subtasks:
        return "none"

    # Build dependency graph
    subtask_ids = [s.get("id", f"step_{i}") for i, s in enumerate(subtasks)]
    depends_map = {s.get("id", f"step_{i}"): s.get("depends_on", [])
                   for i, s in enumerate(subtasks)}

    # Find roots (no dependencies)
    roots = [sid for sid in subtask_ids if not depends_map.get(sid)]

    # Build reverse map (parent -> children)
    children_map: Dict[str, List[str]] = {sid: [] for sid in subtask_ids}
    for sid, deps in depends_map.items():
        for dep in deps:
            if dep in children_map:
                children_map[dep].append(sid)

    # Build chains from each root
    chains = []
    visited = set()

    def build_chain(node_id: str) -> List[str]:
        if node_id in visited:
            return []
        visited.add(node_id)

        children = children_map.get(node_id, [])
        if not children:
            return [node_id]

        # Follow first child for main chain
        result = [node_id]
        for child in children:
            child_chain = build_chain(child)
            if child_chain:
                result.extend(child_chain)
                break
        return result

    for root in roots:
        chain = build_chain(root)
        if chain:
            chains.append(" → ".join(chain))

    return "; ".join(chains) if chains else "none"


def extract_models(task: Dict[str, Any]) -> str:
    """Extract models used for calibration."""
    calibration = task.get("calibration", {})
    if not calibration:
        return "none"
    return ", ".join(sorted(calibration.keys()))


def extract_agent_secrets_separate(task: Dict[str, Any]) -> Dict[str, str]:
    """Extract agent secrets as separate columns."""
    secrets = task.get("agent_secrets", {})
    result = {}

    for i in range(MAX_AGENTS):
        agent_key = f"agent_{i}"
        col_name = f"agent{i}_secret"

        if agent_key in secrets and secrets[agent_key]:
            # Join multiple secrets with semicolons
            result[col_name] = "; ".join(secrets[agent_key])
        else:
            result[col_name] = ""

    return result


def process_task_file(filepath: Path) -> Dict[str, Any]:
    """Process a single task JSON file and extract relevant fields."""
    with open(filepath, "r") as f:
        task = json.load(f)

    row = {
        "filename": filepath.name,
        "task_name": task.get("title", task.get("task_id", "unknown")),
        "task_prompt": task.get("task", ""),
        "model_used": extract_models(task),
        "mechanics": extract_mechanics(task),
        "dag": extract_dag_compact(task),
        "num_agents": task.get("num_agents", len(task.get("agent_spawns", {}))),
        "category": task.get("category", "unknown"),
        "scene_id": task.get("scene_id", ""),
    }

    # Add separate agent secret columns
    agent_secrets = extract_agent_secrets_separate(task)
    row.update(agent_secrets)

    return row


def main():
    parser = argparse.ArgumentParser(description="Convert EMTOM task JSONs to CSV")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/emtom/tasks",
        help="Input directory containing task JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/emtom/tasks_summary.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--exclude-working",
        action="store_true",
        help="Exclude working_task.json from output"
    )
    parser.add_argument(
        "--verbose-dag",
        action="store_true",
        help="Print verbose DAG tree to console for each task"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    # Find all JSON files (excluding subdirectories like tom_judgments, curated)
    json_files = [
        f for f in input_dir.glob("*.json")
        if f.is_file() and not f.name.startswith(".")
    ]

    if args.exclude_working:
        json_files = [f for f in json_files if f.name != "working_task.json"]

    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return 1

    print(f"Processing {len(json_files)} task files...")

    # Process all files
    rows = []
    for filepath in sorted(json_files):
        try:
            row = process_task_file(filepath)
            rows.append(row)

            if args.verbose_dag:
                with open(filepath, "r") as f:
                    task = json.load(f)
                print(f"\n{'='*60}")
                print(f"Task: {row['task_name']}")
                print(f"{'='*60}")
                print(extract_dag(task))
        except Exception as e:
            print(f"Warning: Failed to process {filepath.name}: {e}")

    # Write CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build fieldnames dynamically based on max agents found
    fieldnames = [
        "filename",
        "task_name",
        "task_prompt",
        "model_used",
        "mechanics",
        "dag",
        "num_agents",
    ]

    # Add agent secret columns
    for i in range(MAX_AGENTS):
        fieldnames.append(f"agent{i}_secret")

    fieldnames.extend(["category", "scene_id"])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tasks to '{output_file}'")
    return 0


if __name__ == "__main__":
    exit(main())
