#!/usr/bin/env python3
"""
Static task verifier for EMTOM tasks.

Validates task structure and golden trajectories without launching Habitat.
Useful for fast preflight checks in environments without GPU/EGL support.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from emtom.task_gen.spec_validator import validate_blocking_spec


VALID_ACTIONS = {
    "Navigate", "Open", "Close", "Pick", "Place",
    "UseItem", "Communicate", "Wait", "Clean", "Pour", "PowerOn", "PowerOff",
    "Fill", "FindObjectTool", "FindReceptacleTool", "FindRoomTool",
}

PLACE_RELATIONS = {"on", "on_top", "within", "inside", "in", "next_to"}
ID_PATTERN = re.compile(r"\b[a-z_]+_\d+\b")
ACTION_PATTERN = re.compile(r"(\w+)(?:\[(.*)\])?$")


@dataclass
class VerifyResult:
    task_path: str
    task_id: str
    valid: bool
    errors: List[str]
    warnings: List[str]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _task_entries(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    data = _load_json(path)
    if isinstance(data, dict) and "tasks" in data and isinstance(data["tasks"], list):
        entries: List[Tuple[str, Dict[str, Any]]] = []
        for idx, task in enumerate(data["tasks"]):
            if isinstance(task, dict):
                entries.append((f"{path}::tasks[{idx}]", task))
        return entries
    if isinstance(data, dict):
        return [(str(path), data)]
    return []


def _extract_defined_items(task: Dict[str, Any]) -> Set[str]:
    items = set()
    for item in task.get("items", []):
        if isinstance(item, dict):
            item_id = item.get("item_id")
            if isinstance(item_id, str) and item_id:
                items.add(item_id)
    return items


def _extract_known_task_ids(task: Dict[str, Any]) -> Set[str]:
    """
    Extract IDs referenced across known task fields.

    This is a best-effort fallback when scene object lists are unavailable.
    """
    ids: Set[str] = set()
    ids.update(_extract_defined_items(task))

    for subtask in task.get("subtasks", []):
        if not isinstance(subtask, dict):
            continue
        sc = subtask.get("success_condition", {})
        if not isinstance(sc, dict):
            continue
        for key in ("entity", "target", "value"):
            val = sc.get(key)
            if isinstance(val, str):
                ids.update(ID_PATTERN.findall(val))

    for container, key_item in task.get("locked_containers", {}).items():
        if isinstance(container, str):
            ids.update(ID_PATTERN.findall(container))
        if isinstance(key_item, str):
            ids.update(ID_PATTERN.findall(key_item))

    for obj_name in task.get("initial_states", {}).keys():
        if isinstance(obj_name, str):
            ids.update(ID_PATTERN.findall(obj_name))

    for binding in task.get("mechanic_bindings", []):
        if not isinstance(binding, dict):
            continue
        for key in ("trigger_object", "target_object"):
            val = binding.get(key)
            if isinstance(val, str):
                ids.update(ID_PATTERN.findall(val))

    return ids


def _build_valid_ids(task: Dict[str, Any], scene_data: Optional[Dict[str, Any]]) -> Set[str]:
    valid_ids: Set[str] = set()
    valid_ids.update(_extract_defined_items(task))

    if scene_data:
        for key in ("rooms", "furniture", "objects"):
            for value in scene_data.get(key, []):
                if isinstance(value, str):
                    valid_ids.add(value)

    # Fallback source if scene inventory was not provided.
    valid_ids.update(_extract_known_task_ids(task))
    return valid_ids


def _parse_action(action_str: str) -> Tuple[Optional[str], Optional[str]]:
    match = ACTION_PATTERN.match(action_str or "")
    if not match:
        return None, None
    action_name, args = match.group(1), match.group(2)
    if args == "":
        args = None
    return action_name, args


def _validate_supported_predicates(task: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    from emtom.evaluation import PARTNR_PREDICATES, EMTOM_PREDICATES
    from emtom.state.manager import GameStateManager

    supported = PARTNR_PREDICATES | EMTOM_PREDICATES | GameStateManager.GAME_STATE_PREDICATES

    def check_condition(condition: Dict[str, Any], scope: str) -> None:
        prop = condition.get("property")
        if not prop:
            errors.append(f"{scope} missing 'property'")
            return
        if prop not in supported:
            errors.append(f"{scope} uses unsupported predicate '{prop}'")

    success_condition = task.get("success_condition", {})
    if isinstance(success_condition, dict):
        req = success_condition.get("required_states", [])
        if isinstance(req, list):
            for idx, cond in enumerate(req):
                if isinstance(cond, dict):
                    check_condition(cond, f"success_condition.required_states[{idx}]")

    for idx, subtask in enumerate(task.get("subtasks", [])):
        if not isinstance(subtask, dict):
            continue
        condition = subtask.get("success_condition", {})
        if isinstance(condition, dict):
            check_condition(condition, f"subtasks[{idx}].success_condition")

    return errors


def _validate_golden_trajectory(
    task: Dict[str, Any],
    valid_ids: Set[str],
    strict_object_ids: bool,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    golden = task.get("golden_trajectory", [])
    if not golden:
        return ["No golden_trajectory found"], warnings

    num_agents = task.get("num_agents", 2)
    valid_agents = {f"agent_{i}" for i in range(num_agents)}

    # Lightweight action-state checks for trajectory consistency.
    held_by_agent: Dict[str, Optional[str]] = {a: None for a in valid_agents}
    held_object_owner: Dict[str, str] = {}

    for step_idx, step in enumerate(golden):
        actions = step.get("actions", []) if isinstance(step, dict) else []
        if not actions:
            errors.append(f"Step {step_idx}: No actions array")
            continue

        agents_in_step: Set[str] = set()
        for action_entry in actions:
            if not isinstance(action_entry, dict):
                errors.append(f"Step {step_idx}: action entry must be an object")
                continue

            agent = action_entry.get("agent", "")
            action_str = action_entry.get("action", "")
            if not isinstance(action_str, str) or not action_str:
                errors.append(f"Step {step_idx}: empty action for {agent or 'unknown agent'}")
                continue

            if agent not in valid_agents:
                errors.append(f"Step {step_idx}: Invalid agent '{agent}' (valid: {sorted(valid_agents)})")
            if agent in agents_in_step:
                errors.append(f"Step {step_idx}: Duplicate action for {agent}")
            agents_in_step.add(agent)

            action_name, args = _parse_action(action_str)
            if not action_name:
                errors.append(f"Step {step_idx}: Malformed action '{action_str}'")
                continue

            if action_name not in VALID_ACTIONS:
                errors.append(f"Step {step_idx}: Unknown action '{action_name}'")
                continue

            if action_name == "Wait":
                if args:
                    errors.append(f"Step {step_idx}: Wait must use empty args (use Wait[] or Wait)")
                continue

            if not args:
                if action_name != "Communicate":
                    errors.append(f"Step {step_idx}: {action_name} missing target in '{action_str}'")
                continue

            parts = [p.strip() for p in args.split(",")]
            skip_words = {"on", "on_top", "within", "inside", "in", "next_to", "none", "None", ""}

            if strict_object_ids:
                for part in parts:
                    if part in skip_words:
                        continue
                    if part.startswith("agent_"):
                        continue
                    if valid_ids and part not in valid_ids and not part.startswith("item_"):
                        errors.append(f"Step {step_idx}: Unknown object '{part}' in {action_str}")

            if action_name == "Place":
                if len(parts) < 3:
                    errors.append(f"Step {step_idx}: Place requires at least 3 args in '{action_str}'")
                    continue
                relation = parts[1].lower()
                if relation not in PLACE_RELATIONS:
                    errors.append(f"Step {step_idx}: Invalid Place relation '{parts[1]}' in '{action_str}'")
                obj = parts[0]
                if agent in held_by_agent:
                    held_obj = held_by_agent[agent]
                    if held_obj is None:
                        errors.append(f"Step {step_idx}: {agent} places '{obj}' but is not holding anything")
                    elif held_obj != obj:
                        errors.append(
                            f"Step {step_idx}: {agent} places '{obj}' while holding '{held_obj}'"
                        )
                    else:
                        held_by_agent[agent] = None
                        held_object_owner.pop(obj, None)
                continue

            if action_name == "Pick":
                obj = parts[0]
                if agent in held_by_agent and held_by_agent[agent] is not None:
                    errors.append(
                        f"Step {step_idx}: {agent} picks '{obj}' while already holding '{held_by_agent[agent]}'"
                    )
                    continue
                current_owner = held_object_owner.get(obj)
                if current_owner and current_owner != agent:
                    errors.append(
                        f"Step {step_idx}: {agent} picks '{obj}' but it is already held by {current_owner}"
                    )
                    continue
                held_by_agent[agent] = obj
                held_object_owner[obj] = agent

        missing_agents = valid_agents - agents_in_step
        if missing_agents:
            warnings.append(f"Step {step_idx}: Missing actions for {sorted(missing_agents)}")

    return errors, warnings


def _validate_success_condition_ids(
    task: Dict[str, Any],
    scene_data: Optional[Dict[str, Any]],
) -> List[str]:
    """Validate that success condition entity/target IDs exist in the scene."""
    if not scene_data:
        return []

    errors: List[str] = []

    # Build set of valid scene IDs
    scene_ids: Set[str] = set()
    for key in ("rooms", "furniture", "objects"):
        for value in scene_data.get(key, []):
            if isinstance(value, str):
                scene_ids.add(value)
    # Also allow defined items
    scene_ids.update(_extract_defined_items(task))
    # Allow agent IDs
    num_agents = task.get("num_agents", 2)
    scene_ids.update(f"agent_{i}" for i in range(num_agents))

    for idx, subtask in enumerate(task.get("subtasks", [])):
        if not isinstance(subtask, dict):
            continue
        sc = subtask.get("success_condition", {})
        if not isinstance(sc, dict):
            continue
        for field in ("entity", "target"):
            val = sc.get(field)
            if not isinstance(val, str) or not val:
                continue
            # Check concrete object IDs (pattern: word_number)
            refs = ID_PATTERN.findall(val)
            for ref in refs:
                if ref not in scene_ids and not ref.startswith("item_"):
                    errors.append(
                        f"subtasks[{idx}].success_condition.{field} references "
                        f"'{ref}' which is not in the scene"
                    )

    return errors


def verify_task(
    task_path_label: str,
    task: Dict[str, Any],
    scene_data: Optional[Dict[str, Any]],
    strict_object_ids: bool,
) -> VerifyResult:
    errors: List[str] = []
    warnings: List[str] = []

    task_id = task.get("task_id", "unknown")
    valid_ids = _build_valid_ids(task, scene_data)

    # Shared deterministic checks used in generation path.
    errors.extend(validate_blocking_spec(task, scene_data))

    # PDDL goal validation (new format)
    pddl_goal = task.get("pddl_goal")
    if isinstance(pddl_goal, str) and pddl_goal:
        try:
            from emtom.pddl.dsl import parse_goal_string, Literal, EpistemicFormula
            goal = parse_goal_string(pddl_goal)
            conjuncts = goal.flatten()

            # Extract inner Literal nodes from conjuncts (unwrap K/B wrappers)
            def _extract_literals(formula):
                """Unwrap epistemic layers to get leaf Literal nodes."""
                if isinstance(formula, Literal):
                    return [formula]
                if isinstance(formula, EpistemicFormula):
                    return _extract_literals(formula.inner)
                return []

            literals = []
            for c in conjuncts:
                literals.extend(_extract_literals(c))

            # Check goal predicate names are valid
            from emtom.evaluation import PARTNR_PREDICATES, EMTOM_PREDICATES
            from emtom.state.manager import GameStateManager
            supported = PARTNR_PREDICATES | EMTOM_PREDICATES | GameStateManager.GAME_STATE_PREDICATES
            for lit in literals:
                if lit.predicate not in supported:
                    errors.append(f"PDDL goal uses unsupported predicate '{lit.predicate}'")

            # Check object references
            if scene_data:
                scene_ids = set()
                for key in ("rooms", "furniture", "objects"):
                    for value in scene_data.get(key, []):
                        if isinstance(value, str):
                            scene_ids.add(value)
                scene_ids.update(_extract_defined_items(task))
                num_agents = task.get("num_agents", 2)
                scene_ids.update(f"agent_{i}" for i in range(num_agents))

                for lit in literals:
                    for arg in lit.args:
                        if arg.startswith("?"):
                            continue
                        if arg not in scene_ids and not arg.startswith("item_"):
                            errors.append(f"PDDL goal references unknown object '{arg}'")

        except Exception as e:
            errors.append(f"PDDL goal validation failed: {e}")

    # DAG checks (legacy format)
    elif isinstance(task.get("subtasks"), list) and task.get("subtasks"):
        try:
            from emtom.task_gen import Subtask
            from emtom.task_gen.dag import validate_dag

            subtasks = [Subtask.from_dict(s) for s in task["subtasks"] if isinstance(s, dict)]
            is_valid_dag, dag_errors = validate_dag(subtasks)
            if not is_valid_dag:
                for err in dag_errors:
                    errors.append(f"Invalid subtask DAG: {err}")
        except Exception as e:
            errors.append(f"DAG validation failed: {e}")

    errors.extend(_validate_supported_predicates(task))
    errors.extend(_validate_success_condition_ids(task, scene_data))
    golden_errors, golden_warnings = _validate_golden_trajectory(task, valid_ids, strict_object_ids)
    errors.extend(golden_errors)
    warnings.extend(golden_warnings)

    return VerifyResult(
        task_path=task_path_label,
        task_id=task_id,
        valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statically verify EMTOM task files without launching Habitat."
    )
    parser.add_argument("--task", type=str, default=None, help="Path to a task JSON file")
    parser.add_argument("--task-dir", type=str, default=None, help="Directory containing task JSON files")
    parser.add_argument(
        "--scene-data",
        type=str,
        default=None,
        help="Optional scene data JSON with rooms/furniture/objects for strict ID checks",
    )
    parser.add_argument(
        "--strict-object-ids",
        action="store_true",
        help="Fail when action IDs are not in known scene/task IDs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.task and not args.task_dir:
        print("Error: provide --task or --task-dir", file=sys.stderr)
        sys.exit(2)
    if args.task and args.task_dir:
        print("Error: use only one of --task or --task-dir", file=sys.stderr)
        sys.exit(2)

    scene_data = None
    if args.scene_data:
        scene_data = _load_json(Path(args.scene_data))

    task_paths: List[Path] = []
    if args.task:
        task_paths = [Path(args.task)]
    else:
        task_paths = sorted(Path(args.task_dir).glob("*.json"))

    if not task_paths:
        print("No task files found.")
        sys.exit(1)

    results: List[VerifyResult] = []
    for task_path in task_paths:
        try:
            entries = _task_entries(task_path)
        except Exception as e:
            results.append(
                VerifyResult(
                    task_path=str(task_path),
                    task_id="unknown",
                    valid=False,
                    errors=[f"Failed to load JSON: {e}"],
                    warnings=[],
                )
            )
            continue

        if not entries:
            results.append(
                VerifyResult(
                    task_path=str(task_path),
                    task_id="unknown",
                    valid=False,
                    errors=["No task object found in JSON"],
                    warnings=[],
                )
            )
            continue

        for label, task in entries:
            results.append(
                verify_task(
                    task_path_label=label,
                    task=task,
                    scene_data=scene_data,
                    strict_object_ids=args.strict_object_ids,
                )
            )

    output_payload = {
        "all_valid": all(r.valid for r in results),
        "total": len(results),
        "passed": sum(1 for r in results if r.valid),
        "failed": sum(1 for r in results if not r.valid),
        "results": [asdict(r) for r in results],
    }

    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(output_payload, f, indent=2)

    for result in results:
        status = "PASS" if result.valid else "FAIL"
        print(f"[{status}] {result.task_path} (task_id={result.task_id})")
        for err in result.errors[:10]:
            print(f"  - ERROR: {err}")
        for warning in result.warnings[:5]:
            print(f"  - WARNING: {warning}")

    print(
        f"\nSummary: {output_payload['passed']}/{output_payload['total']} passed "
        f"({output_payload['failed']} failed)"
    )
    if args.output:
        print(f"Saved JSON report: {args.output}")

    sys.exit(0 if output_payload["all_valid"] else 1)


if __name__ == "__main__":
    main()
