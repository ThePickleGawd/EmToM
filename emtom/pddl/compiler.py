"""
Task → PDDL compiler.

Converts a GeneratedTask + scene data into a PDDL Problem instance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from emtom.pddl.dsl import (
    Domain, Formula, Literal, Knows, Believes, Not, And, Or,
    Problem,
)
from emtom.pddl.problem_pddl import parse_problem_pddl, validate_problem_pddl_self_contained

if TYPE_CHECKING:
    from emtom.task_gen.task_generator import GeneratedTask

# Valid PDDL identifiers: letters, digits, underscores, hyphens
_VALID_PDDL_ID = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')


def _infer_object_types(formula: Formula, domain: Domain) -> Dict[str, str]:
    """Infer PDDL types for object IDs referenced in a formula.

    Uses domain predicate signatures to determine the most specific type
    for each object. Falls back to naming convention if no predicate
    constraint narrows the type beyond ``object``.
    """
    # Build predicate-name -> list of param types
    pred_types: Dict[str, List[str]] = {}
    for pred in domain.predicates:
        pred_types[pred.name] = [p.type for p in pred.params]

    # Collect (object_id -> set of required types) from goal literals
    constraints: Dict[str, Set[str]] = {}

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            ptypes = pred_types.get(node.predicate, [])
            for i, arg in enumerate(node.args):
                if arg.startswith("?"):
                    continue
                typ = ptypes[i] if i < len(ptypes) else "object"
                constraints.setdefault(arg, set()).add(typ)
        elif isinstance(node, (Knows, Believes)):
            _walk(node.inner)
        elif isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)
        elif isinstance(node, And):
            for op in node.operands:
                _walk(op)
        elif isinstance(node, Or):
            for op in node.operands:
                _walk(op)

    _walk(formula)

    # Resolve: pick most specific type (anything that isn't root "object")
    result: Dict[str, str] = {}
    for obj_id, types in constraints.items():
        specific = types - {"object"}
        if len(specific) == 1:
            result[obj_id] = specific.pop()
        elif specific:
            # Multiple non-root types — shouldn't happen in valid PDDL, pick first
            result[obj_id] = sorted(specific)[0]
        else:
            # Only "object" constraint — use naming convention
            result[obj_id] = _type_from_name(obj_id)
    return result


def _type_from_name(obj_id: str) -> str:
    """Guess PDDL type from naming convention."""
    if obj_id.startswith("agent_"):
        return "agent"
    if obj_id.startswith("item_") or obj_id.startswith("key_"):
        return "item"
    room_prefixes = (
        "room_", "kitchen_", "bedroom_", "bathroom_", "living_room_",
        "garage_", "hallway_", "lobby_", "office_", "closet_", "laundry_",
        "dining_", "pantry_", "porch_", "utility_",
    )
    if any(obj_id.startswith(p) for p in room_prefixes):
        return "room"
    # Default: most objects in emtom scenes are furniture
    return "furniture"


def compile_task(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
) -> Problem:
    """Compile a GeneratedTask into a PDDL Problem."""
    task_problem_pddl = getattr(task, "problem_pddl", None)
    if isinstance(task_problem_pddl, str) and task_problem_pddl.strip():
        parsed = parse_problem_pddl(task_problem_pddl)
        validation_errors = validate_problem_pddl_self_contained(
            parsed,
            num_agents=getattr(task, "num_agents", None),
        )
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        problem = parsed.to_problem()

        # Mechanics are the single authored source for runtime constraints.
        _ensure_room_restrictions(task, problem)
        # Add default init facts (e.g., furniture starts closed)
        _add_default_init_facts(problem)

        # Populate can_communicate if not already in init
        _ensure_can_communicate(task, problem)
        return problem

    raise ValueError(
        "Task must define non-empty problem_pddl. "
        "Legacy goal formats are no longer supported."
    )


def _add_default_init_facts(problem: Problem) -> None:
    """Add default init facts for the planner.

    In the real simulator, furniture starts closed and objects have default
    states. The planner needs these facts explicitly under CWA.
    """
    existing = {(l.predicate, l.args) for l in problem.init}

    for obj_id, typ in problem.objects.items():
        if typ == "furniture":
            # Furniture defaults to closed (unless explicitly open)
            if (("is_open", (obj_id,)) not in existing
                    and ("is_closed", (obj_id,)) not in existing):
                problem.init.append(Literal("is_closed", (obj_id,)))


def _binding_value(binding: Any, key: str) -> Any:
    if isinstance(binding, dict):
        return binding.get(key)
    return getattr(binding, key, None)


def _ensure_room_restrictions(task: "GeneratedTask", problem: Problem) -> None:
    """Populate is_restricted predicates from room_restriction mechanic bindings."""
    existing = {(l.predicate, l.args) for l in problem.init}

    for binding in getattr(task, "mechanic_bindings", []) or []:
        if _binding_value(binding, "mechanic_type") != "room_restriction":
            continue
        rooms = _binding_value(binding, "restricted_rooms") or []
        agents = _binding_value(binding, "for_agents") or []
        for agent in agents:
            if not isinstance(agent, str):
                continue
            for room in rooms:
                if not isinstance(room, str):
                    continue
                fact = ("is_restricted", (agent, room))
                if fact in existing:
                    continue
                problem.init.append(Literal("is_restricted", (agent, room)))
                existing.add(fact)


def _ensure_can_communicate(task: "GeneratedTask", problem: Problem) -> None:
    """Populate can_communicate predicates in problem init.

    - Default: all agent pairs can communicate (can_communicate a_i a_j for i!=j).
    - restricted_communication mechanic: only allowed pairs.
    - message_targets: same constraint source.

    Skips if can_communicate is already present in init (authored in problem_pddl).
    """
    existing = {(l.predicate, l.args) for l in problem.init}
    has_can_comm = any(pred == "can_communicate" for pred, _ in existing)
    if has_can_comm:
        return  # Already authored

    num_agents = task.num_agents
    all_agents = [f"agent_{i}" for i in range(num_agents)]

    # Check for restricted_communication mechanic
    restricted_targets = None
    for binding in task.mechanic_bindings:
        if _binding_value(binding, "mechanic_type") == "restricted_communication":
            allowed_targets = _binding_value(binding, "allowed_targets")
            if allowed_targets:
                restricted_targets = allowed_targets
            break

    # Check for message_targets
    message_targets = task.message_targets

    if restricted_targets is not None:
        # Only add allowed pairs from allowed_targets dict
        for agent, targets in restricted_targets.items():
            for target in targets:
                problem.init.append(Literal("can_communicate", (agent, target)))
    elif message_targets:
        # message_targets: agent -> list of targets
        for agent, targets in message_targets.items():
            for target in targets:
                problem.init.append(Literal("can_communicate", (agent, target)))
    else:
        # Default: all pairs
        for a in all_agents:
            for b in all_agents:
                if a != b:
                    problem.init.append(Literal("can_communicate", (a, b)))
