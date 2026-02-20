"""
Task → PDDL compiler.

Converts a GeneratedTask + scene data into a PDDL Problem instance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from emtom.pddl.dsl import Literal, Knows, EpistemicFormula, Problem, parse_goal_string
from emtom.pddl.epistemic import ObservabilityModel

if TYPE_CHECKING:
    from emtom.task_gen.task_generator import GeneratedTask


def compile_task(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
) -> Problem:
    """
    Compile a GeneratedTask into a PDDL Problem.

    Args:
        task: The generated task with pddl_goal
        scene_data: Optional scene data with rooms/furniture/objects lists

    Returns:
        Problem instance ready for solver
    """
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
            if item_id:
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
            if item_id and container:
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
    pddl_goal = getattr(task, 'pddl_goal', None)
    if pddl_goal:
        goal = parse_goal_string(pddl_goal)

        # Auto-register objects referenced in goal but not yet in objects dict
        for conjunct in goal.flatten():
            # Unwrap epistemic layers to get leaf literals
            node = conjunct
            while isinstance(node, EpistemicFormula):
                node = node.inner
            if not isinstance(node, Literal):
                continue
            for arg in node.args:
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

    return Problem(
        name=f"task_{task.task_id}",
        domain_name="emtom",
        objects=objects,
        init=init,
        goal=goal,
        epistemic_init=epistemic_init,
    )
