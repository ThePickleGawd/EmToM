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


def extract_mechanic_bindings(task: Dict[str, Any]) -> str:
    """Extract full mechanic bindings as JSON string."""
    bindings = task.get("mechanic_bindings", [])
    if not bindings:
        return "none"
    return json.dumps(bindings, separators=(",", ":"))


def extract_items_used(task: Dict[str, Any]) -> str:
    """Extract items used in the task."""
    items = task.get("items", [])
    if not items:
        return "none"
    # Format: item_id (inside location)
    # Backward-compat: accept legacy hidden_in field from older tasks.
    item_strs = []
    for item in items:
        item_id = item.get("item_id", "unknown")
        location = item.get("inside") or item.get("hidden_in") or "unknown"
        item_strs.append(f"{item_id} (in {location})")
    return "; ".join(item_strs)


def extract_success_criteria(task: Dict[str, Any]) -> str:
    """Extract success criteria for all DAG nodes (subtasks)."""
    subtasks = task.get("subtasks", [])
    if not subtasks:
        return "none"

    criteria_strs = []
    for subtask in subtasks:
        subtask_id = subtask.get("id", "unknown")
        condition = subtask.get("success_condition", {})
        if condition:
            entity = condition.get("entity", "?")
            prop = condition.get("property", "?")
            target = condition.get("target", "")
            if target:
                cond_str = f"{subtask_id}: {entity}.{prop}({target})"
            else:
                cond_str = f"{subtask_id}: {entity}.{prop}"
            criteria_strs.append(cond_str)
        else:
            criteria_strs.append(f"{subtask_id}: no_condition")

    return "; ".join(criteria_strs)


def build_judge_prompt(task: Dict[str, Any]) -> str:
    """
    Build the judge evaluation prompt for the task.

    This constructs the prompt that would be sent to the judge LLM
    to evaluate the task quality.
    """
    category = task.get("category", "cooperative")
    task_json = json.dumps(task, indent=2)

    # Simplified prompt template (full version is in emtom/task_gen/judge.py)
    prompt = f"""You are an expert evaluator for multi-agent tasks.

## Task Category: {category.upper()}

## Task to Evaluate

```json
{task_json}
```

## Evaluation Criteria

Score each criterion from 0.0 to 1.0:
1. Agent Necessity - Is every agent essential?
2. Secret Relevance - Are agent secrets useful and required?
3. Narrative Consistency - Does task description match subtasks?
4. Subtask Relevance - Does every subtask contribute to the goal?
5. Mechanic Utilization - Are listed mechanics essential?
"""

    if category == "cooperative":
        prompt += "6. Task Interdependence - Do agents genuinely need each other?\n"
    elif category == "competitive":
        prompt += "6. Goal Opposition - Do teams have mutually exclusive win conditions?\n"
    elif category == "mixed":
        prompt += "6. Subgoal Tension - Do hidden subgoals create meaningful conflict?\n"

    prompt += """
Respond with JSON containing scores, reasoning, and required fixes.
"""
    return prompt


def find_judgment_file(task: Dict[str, Any], judgments_dir: Path) -> Tuple[str, str, str]:
    """
    Find and extract judge data from judgment files.

    Returns: (judge_prompt, judge_required_fixes, judge_score)

    Note: Judgment files must contain a 'task_id' field that matches the task.
    Without this linking, we cannot match judgments to tasks.
    """
    if not judgments_dir.exists():
        return ("", "", "")

    task_id = task.get("task_id", "")
    if not task_id:
        return ("", "", "")

    # Look for judgment files
    judgment_files = list(judgments_dir.glob("tom_judgment_*.json"))

    # Try to find a matching judgment file by task_id
    for jfile in sorted(judgment_files, reverse=True):  # Most recent first
        try:
            with open(jfile, "r") as f:
                judgment = json.load(f)

            # Only match if judgment has explicit task_id
            judgment_task_id = judgment.get("task_id", "")
            if judgment_task_id and judgment_task_id == task_id:
                score = judgment.get("overall_score", "")
                required_fixes = judgment.get("required_fixes", [])
                required_fixes_str = "; ".join(required_fixes) if required_fixes else ""
                return ("", required_fixes_str, str(score) if score else "")

        except Exception:
            continue

    # No matching judgment found
    return ("", "", "")


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


def process_task_file(filepath: Path, judgments_dir: Path = None) -> Dict[str, Any]:
    """Process a single task JSON file and extract relevant fields."""
    with open(filepath, "r") as f:
        task = json.load(f)

    row = {
        "filename": filepath.name,
        "task_name": task.get("title", task.get("task_id", "unknown")),
        "task_prompt": task.get("task", ""),
        "mechanics": extract_mechanics(task),
        "mechanic_bindings": extract_mechanic_bindings(task),
        "items_used": extract_items_used(task),
        "dag": extract_dag_compact(task),
        "success_criteria": extract_success_criteria(task),
        "num_agents": task.get("num_agents", len(task.get("agent_spawns", {}))),
        "category": task.get("category", "unknown"),
        "scene_id": task.get("scene_id", ""),
    }

    # Add separate agent secret columns
    agent_secrets = extract_agent_secrets_separate(task)
    row.update(agent_secrets)

    # Build judge prompt
    row["judge_prompt"] = build_judge_prompt(task)

    # Try to find judgment data
    if judgments_dir:
        _, judge_required_fixes, judge_score = find_judgment_file(task, judgments_dir)
        row["judge_required_fixes"] = judge_required_fixes
        row["judge_score"] = judge_score
    else:
        row["judge_required_fixes"] = ""
        row["judge_score"] = ""

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
        "--judgments-dir",
        type=str,
        default=None,
        help="Directory containing tom_judgment JSON files (default: <input>/tom_judgments)"
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

    # Set up judgments directory
    if args.judgments_dir:
        judgments_dir = Path(args.judgments_dir)
    else:
        judgments_dir = input_dir / "tom_judgments"

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
    if judgments_dir.exists():
        print(f"Looking for judgments in: {judgments_dir}")

    # Process all files
    rows = []
    for filepath in sorted(json_files):
        try:
            row = process_task_file(filepath, judgments_dir)
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
        "mechanics",
        "mechanic_bindings",
        "items_used",
        "dag",
        "success_criteria",
        "num_agents",
    ]

    # Add agent secret columns
    for i in range(MAX_AGENTS):
        fieldnames.append(f"agent{i}_secret")

    fieldnames.extend([
        "category",
        "scene_id",
        "judge_prompt",
        "judge_required_fixes",
        "judge_score",
    ])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tasks to '{output_file}'")
    return 0


if __name__ == "__main__":
    exit(main())
