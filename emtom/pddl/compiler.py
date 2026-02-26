"""
Task → PDDL compiler.

Converts a GeneratedTask + scene data into a PDDL Problem instance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from emtom.pddl.dsl import (
    Domain, Formula, Literal, Knows, Believes, EpistemicFormula, Not, And, Or,
    Problem, parse_goal_string, collect_leaf_literals,
)
from emtom.pddl.epistemic import ObservabilityModel
from emtom.pddl.problem_pddl import parse_problem_pddl

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
    """
    Compile a GeneratedTask into a PDDL Problem.

    Args:
        task: Generated task with inline `problem_pddl` or legacy goal fields
        scene_data: Optional scene data with rooms/furniture/objects lists

    Returns:
        Problem instance ready for solver
    """
    # New single-format path: inline task-level problem PDDL.
    task_problem_pddl = getattr(task, "problem_pddl", None)
    if isinstance(task_problem_pddl, str) and task_problem_pddl.strip():
        parsed = parse_problem_pddl(task_problem_pddl)
        problem = parsed.to_problem()

        # Canonical grounding for inline problems:
        # keep authored goal/init, but normalize object inventory from
        # scene + task metadata so solver checks don't fail on omitted
        # declarations in :objects.
        if scene_data:
            for room in scene_data.get("rooms", []):
                if isinstance(room, str) and room:
                    problem.objects[room] = "room"
            for furn in scene_data.get("furniture", []):
                if isinstance(furn, str) and furn:
                    problem.objects[furn] = "furniture"
            for obj in scene_data.get("objects", []):
                if isinstance(obj, str) and obj:
                    problem.objects[obj] = "object"

        for item_def in (task.items or []):
            if isinstance(item_def, dict):
                item_id = item_def.get("item_id")
                if isinstance(item_id, str) and item_id and _VALID_PDDL_ID.match(item_id):
                    problem.objects[item_id] = "item"

        # Ensure declared agent cardinality is reflected in objects.
        for i in range(task.num_agents):
            problem.objects[f"agent_{i}"] = "agent"

        # Auto-register objects referenced in the goal but missing from :objects.
        # This handles legacy-migrated tasks and tasks where :objects is minimal.
        if problem.goal:
            from emtom.pddl.domain import EMTOM_DOMAIN
            inferred = _infer_object_types(problem.goal, EMTOM_DOMAIN)
            for obj_id, typ in inferred.items():
                if obj_id not in problem.objects:
                    problem.objects[obj_id] = typ

        # Add default init facts (e.g., furniture starts closed)
        _add_default_init_facts(problem)

        # Populate can_communicate if not already in init
        _ensure_can_communicate(task, problem)
        return problem

    objects: Dict[str, str] = {}
    init: List[Literal] = []
    epistemic_init: List[Knows] = []

    # --- Objects ---

    # Agents
    for i in range(task.num_agents):
        objects[f"agent_{i}"] = "agent"

    # Scene objects
    if scene_data:
        for room in scene_data.get("rooms", []):
            objects[room] = "room"
        for furn in scene_data.get("furniture", []):
            objects[furn] = "furniture"
        for obj in scene_data.get("objects", []):
            objects[obj] = "object"

    # Items
    for item_def in (task.items or []):
        if isinstance(item_def, dict):
            item_id = item_def.get("item_id")
            if item_id and _VALID_PDDL_ID.match(item_id):
                objects[item_id] = "item"

    # --- Init state ---

    # Initial states from task
    for obj_name, states in (task.initial_states or {}).items():
        if not isinstance(states, dict):
            continue
        for prop, val in states.items():
            if val is True:
                init.append(Literal(predicate=prop, args=(obj_name,)))
            elif val is False:
                init.append(Literal(predicate=prop, args=(obj_name,), negated=True))
            elif isinstance(val, str):
                init.append(Literal(predicate=prop, args=(obj_name, val)))

    # Mechanic bindings → init predicates
    for binding in task.mechanic_bindings:
        mtype = binding.mechanic_type
        trigger = binding.trigger_object
        target = binding.target_object

        if mtype == "inverse_state" and trigger:
            init.append(Literal("is_inverse", (trigger,)))

        elif mtype == "state_mirroring" and trigger and target:
            init.append(Literal("mirrors", (trigger, target)))

        elif mtype == "remote_control" and trigger and target:
            init.append(Literal("controls", (trigger, target)))

        elif mtype == "room_restriction":
            rooms = binding.restricted_rooms or []
            agents = binding.for_agents or []
            for agent in agents:
                for room in rooms:
                    init.append(Literal("is_restricted", (agent, room)))

        elif mtype == "conditional_unlock" and trigger:
            init.append(Literal("is_locked_permanent", (trigger,)))
            req_item = getattr(binding, 'requires_item', None)
            if req_item:
                init.append(Literal("requires_item", (trigger, req_item)))

    # Items in containers
    for item_def in (task.items or []):
        if isinstance(item_def, dict):
            item_id = item_def.get("item_id")
            container = item_def.get("inside") or item_def.get("hidden_in")
            if (item_id and container
                    and _VALID_PDDL_ID.match(item_id)
                    and _VALID_PDDL_ID.match(container)):
                init.append(Literal("is_inside", (item_id, container)))

    # Locked containers
    for container, key_item in (task.locked_containers or {}).items():
        init.append(Literal("is_locked_permanent", (container,)))
        if key_item:
            init.append(Literal("requires_item", (container, key_item)))

    # --- Epistemic init (from observability model) ---

    observability = ObservabilityModel.from_task(task)

    # Agents know about things in rooms they can observe
    # Agents DON'T know about things in rooms they're restricted from
    if observability.restricted_rooms:
        # For each agent with restrictions, they don't know the state of
        # objects in restricted rooms. This is modeled as absence of K() atoms.
        # The solver handles this via closed-world assumption on K().
        pass  # Epistemic init is empty by default = agents don't know anything
        # Knowledge is gained via communicate actions

    # --- Goal ---

    goal = None
    # Extract goal from problem_pddl if available
    task_problem_pddl = getattr(task, 'problem_pddl', None)
    if isinstance(task_problem_pddl, str) and task_problem_pddl.strip():
        try:
            from emtom.pddl.problem_pddl import extract_goal_from_problem_pddl
            goal = parse_goal_string(extract_goal_from_problem_pddl(task_problem_pddl))
        except Exception:
            pass

    if goal is not None:
        # Auto-register objects referenced in goal but not yet in objects dict
        for literal in collect_leaf_literals(goal):
            for arg in literal.args:
                if not arg.startswith("?") and arg not in objects:
                    # Infer type from naming convention
                    if any(arg.startswith(p) for p in ("agent_",)):
                        objects[arg] = "agent"
                    else:
                        objects[arg] = "object"

    # Auto-register objects from mechanic bindings not yet in objects dict
    for binding in task.mechanic_bindings:
        for attr in ("trigger_object", "target_object"):
            obj_id = getattr(binding, attr, None)
            if obj_id and obj_id not in objects:
                objects[obj_id] = "object"

    problem = Problem(
        name=f"task_{task.task_id}",
        domain_name="emtom",
        objects=objects,
        init=init,
        goal=goal,
        epistemic_init=epistemic_init,
    )

    # Add default init facts (e.g., furniture starts closed)
    _add_default_init_facts(problem)

    # Populate can_communicate
    _ensure_can_communicate(task, problem)
    return problem


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
        if binding.mechanic_type == "restricted_communication":
            if binding.allowed_targets:
                restricted_targets = binding.allowed_targets
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
