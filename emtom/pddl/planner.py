"""
Deterministic rule-based golden trajectory planner.

Given a PDDL goal and scene data, produces a golden trajectory that satisfies
the goal. Same spec + scene always produces the same trajectory.

This module is used by both agent.py (ReAct loop) and the CLI verify/submit
commands, ensuring a single source of truth for trajectory generation.

Usage:
    from emtom.pddl.planner import regenerate_golden_trajectory

    task_data = json.load(open("task.json"))
    scene_data = ...  # SceneData or dict with rooms/furniture/objects
    result = regenerate_golden_trajectory(task_data, scene_data, source="verify")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InformAction:
    """A parsed inform action from an FD epistemic plan."""
    receiver: str       # agent who receives knowledge
    fact_hash: str      # 8-char hex hash of the leaf fact
    sender: str         # agent who sends the information


# Regex for FD inform action names:
#   inform_knows_{receiver}_{hash}_from_{sender}[_tokN]
_INFORM_RE = re.compile(
    r"inform_knows_(agent_\d+)_([0-9a-f]{8})_from_(agent_\d+)(?:_tok\d+)?"
)


def parse_fd_inform_actions(fd_plan: List[str]) -> List[InformAction]:
    """Parse inform actions from an FD plan, preserving plan order.

    FD plan steps are strings like:
        "inform_knows_agent_0_abc12345_from_agent_1"
        "inform_knows_agent_0_abc12345_from_agent_1_tok1"

    Returns InformActions in plan order (preserves relay chain ordering).
    """
    results: List[InformAction] = []
    for step in fd_plan:
        m = _INFORM_RE.search(step)
        if m:
            results.append(InformAction(
                receiver=m.group(1),
                fact_hash=m.group(2),
                sender=m.group(3),
            ))
    return results


def _derive_communicate_steps(
    task_data: Dict[str, Any],
    scene_data: Any,
    pddl_goal: str,
    num_agents: int,
) -> Dict[str, Any]:
    """Derive Communicate trajectory steps from FD epistemic plan.

    Runs the epistemic compilation + FD solver pipeline, parses inform
    actions from the plan, and converts them to Communicate[] steps.

    Returns dict with "steps" (list of trajectory steps) and "notes" (list of str).
    """
    from emtom.pddl.describe import _literal_to_nl, goal_to_natural_language
    from emtom.pddl.domain import EMTOM_DOMAIN
    from emtom.pddl.dsl import Literal, Knows, Believes, parse_goal_string
    from emtom.pddl.epistemic import ObservabilityModel
    from emtom.pddl.epistemic_compiler import (
        compile_epistemic,
        _collect_k_goals,
        _collect_leaf_facts,
        _get_leaf_formula,
    )
    from emtom.pddl.compiler import compile_task
    from emtom.pddl.fd_solver import FastDownwardSolver, HAS_UP
    from emtom.task_gen.task_generator import GeneratedTask

    if not HAS_UP:
        raise RuntimeError(
            "unified-planning is required for epistemic trajectory derivation "
            "(install unified-planning and up-fast-downward)."
        )

    # Convert SceneData to dict if needed
    if isinstance(scene_data, dict):
        scene_dict = scene_data
    elif hasattr(scene_data, "to_dict"):
        scene_dict = scene_data.to_dict()
    else:
        scene_dict = {}

    # Build GeneratedTask + compile to PDDL Problem
    task = GeneratedTask.from_dict(task_data)
    problem = compile_task(task, scene_dict)
    obs = ObservabilityModel.from_task_with_scene(task, scene_dict)

    if not obs.object_rooms:
        raise ValueError(
            "Epistemic trajectory derivation requires scene object-room mapping "
            "(missing observability.object_rooms)."
        )

    # Parse the goal formula
    goal = parse_goal_string(pddl_goal)

    # Run epistemic compilation to check if belief_depth > 0
    compilation = compile_epistemic(goal, EMTOM_DOMAIN, problem, obs)
    if compilation.belief_depth == 0:
        return {
            "steps": [],
            "notes": ["belief_depth=0 (trivial K goals only); no communicate steps needed"],
            "communication_required": False,
        }

    # Solve with FD to get the plan
    solver = FastDownwardSolver()
    result = solver._solve_epistemic(EMTOM_DOMAIN, problem, obs, timeout=30.0, start=0.0)

    if not result.solvable or not result.plan:
        raise RuntimeError(f"FD epistemic solve failed: {result.error or 'no plan'}")

    # Parse inform actions from FD plan
    informs = parse_fd_inform_actions(result.plan)
    if not informs:
        raise RuntimeError(
            "FD epistemic plan has no inform actions despite non-trivial epistemic goal."
        )

    # Build hash → formula maps
    k_goals = _collect_k_goals(goal, obs)
    leaf_facts = _collect_leaf_facts(k_goals)  # hash → Formula

    # Build nested K map: fact_id → (outer_agent, inner_agent, leaf_formula)
    nested_k_map: Dict[str, tuple] = {}
    for kg in k_goals:
        if kg.depth >= 2 and isinstance(kg.inner, (Knows, Believes)):
            inner_agent = kg.inner.agent
            leaf = _get_leaf_formula(kg.inner.inner)
            if leaf is not None:
                nested_k_map[kg.fact_id] = (kg.agent, inner_agent, leaf)

    # Convert each inform action to a Communicate step
    steps: List[Dict[str, Any]] = []
    notes: List[str] = []

    for inform in informs:
        formula = leaf_facts.get(inform.fact_hash)
        nested = nested_k_map.get(inform.fact_hash)

        if formula is not None and isinstance(formula, Literal):
            message = _literal_to_nl(formula)
        elif formula is not None:
            message = goal_to_natural_language(formula)
        elif nested is not None:
            _outer_agent, inner_agent, leaf = nested
            if isinstance(leaf, Literal):
                inner_msg = _literal_to_nl(leaf)
            else:
                inner_msg = goal_to_natural_language(leaf)
            message = f"{inner_agent} confirmed: {inner_msg}"
        else:
            raise RuntimeError(
                f"Unknown fact_hash '{inform.fact_hash}' in inform action; "
                "cannot derive deterministic Communicate step."
            )

        # Build the parallel step: sender Communicates, others Wait
        step_actions: List[Dict[str, str]] = []
        for i in range(max(1, num_agents)):
            agent_id = f"agent_{i}"
            if agent_id == inform.sender:
                step_actions.append({
                    "agent": agent_id,
                    "action": f'Communicate["{message}", {inform.receiver}]',
                })
            else:
                step_actions.append({
                    "agent": agent_id,
                    "action": "Wait[]",
                })
        steps.append({"actions": step_actions})

    if not steps:
        raise RuntimeError(
            "Epistemic plan required communication but no Communicate steps were derived."
        )

    return {"steps": steps, "notes": notes, "communication_required": True}


def _has_epistemic_goal(formula: Any) -> bool:
    """Return True when goal formula contains K()/B() operators."""
    from emtom.pddl.dsl import And, Or, Not, Knows, Believes

    if isinstance(formula, (Knows, Believes)):
        return True
    if isinstance(formula, (And, Or)):
        return any(_has_epistemic_goal(op) for op in formula.operands)
    if isinstance(formula, Not) and formula.operand is not None:
        return _has_epistemic_goal(formula.operand)
    return False


def canonical_json(value: Any) -> str:
    """Canonical JSON serialization for stable hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_task_spec_hash(task_data: Dict[str, Any]) -> str:
    """Hash the authoritative task spec (excluding golden trajectory artifacts)."""
    spec_keys = [
        "task_id", "title", "category", "task",
        "scene_id", "episode_id", "num_agents",
        "active_mechanics", "mechanic_bindings",
        "agent_secrets", "agent_actions",
        "pddl_domain", "problem_pddl",
        "items", "locked_containers", "initial_states",
        "message_targets", "teams", "team_secrets",
        "agent_spawns",
    ]
    spec_payload = {k: task_data.get(k) for k in spec_keys}
    return hashlib.sha256(canonical_json(spec_payload).encode("utf-8")).hexdigest()


def extract_room_restrictions(task_data: Dict[str, Any]) -> Dict[str, set]:
    """Build agent -> restricted rooms map from room_restriction mechanics."""
    restrictions: Dict[str, set] = {}
    for binding in task_data.get("mechanic_bindings", []):
        if not isinstance(binding, dict):
            continue
        if binding.get("mechanic_type") != "room_restriction":
            continue
        rooms = binding.get("restricted_rooms") or []
        agents = binding.get("for_agents") or []
        for agent_id in agents:
            if not isinstance(agent_id, str):
                continue
            restrictions.setdefault(agent_id, set()).update(
                room for room in rooms if isinstance(room, str)
            )
    return restrictions


def build_target_to_room_map(scene_data: Optional[Any]) -> Dict[str, str]:
    """Map room/furniture/object IDs to room IDs."""
    if not scene_data:
        return {}

    def _get(field: str, default):
        if hasattr(scene_data, field):
            return getattr(scene_data, field)
        if isinstance(scene_data, dict):
            return scene_data.get(field, default)
        return default

    rooms = _get("rooms", []) or []
    furniture_in_rooms = _get("furniture_in_rooms", {}) or {}
    objects_on_furniture = _get("objects_on_furniture", {}) or {}

    target_to_room: Dict[str, str] = {}
    for room in rooms:
        if isinstance(room, str):
            target_to_room[room] = room

    furniture_to_room: Dict[str, str] = {}
    if isinstance(furniture_in_rooms, dict):
        for room, furns in furniture_in_rooms.items():
            if not isinstance(room, str) or not isinstance(furns, list):
                continue
            for furn in furns:
                if isinstance(furn, str):
                    furniture_to_room[furn] = room
                    target_to_room[furn] = room

    if isinstance(objects_on_furniture, dict):
        for furn, objs in objects_on_furniture.items():
            room = furniture_to_room.get(furn)
            if not room or not isinstance(objs, list):
                continue
            for obj in objs:
                if isinstance(obj, str):
                    target_to_room[obj] = room

    return target_to_room


def _agent_sort_key(agent_id: str, agent_loads: Dict[str, int]) -> tuple:
    """Stable deterministic tie-breaker: (load, numeric agent index)."""
    try:
        idx = int(agent_id.split("_", 1)[1])
    except Exception:
        idx = 0
    return (agent_loads.get(agent_id, 0), idx)


def _resolve_target_room(
    target_id: str,
    target_to_room: Dict[str, str],
    restrictions: Dict[str, set],
) -> Optional[str]:
    """Resolve target_id to its room, checking restriction maps as fallback."""
    room = target_to_room.get(target_id)
    if room is not None:
        return room
    if restrictions:
        restricted_rooms = {
            r for rooms in restrictions.values() for r in rooms if isinstance(r, str)
        }
        if target_id in restricted_rooms:
            return target_id
    return None


def _feasible_agents(
    target_rooms: List[Optional[str]],
    num_agents: int,
    restrictions: Dict[str, set],
) -> List[str]:
    """Return agents that can reach ALL of the given rooms."""
    feasible: List[str] = []
    for i in range(max(1, num_agents)):
        agent_id = f"agent_{i}"
        agent_restricted = restrictions.get(agent_id, set())
        if all(
            room is None or room not in agent_restricted
            for room in target_rooms
        ):
            feasible.append(agent_id)
    return feasible


def pick_agent_for_target(
    target_id: str,
    num_agents: int,
    target_to_room: Dict[str, str],
    restrictions: Dict[str, set],
    agent_loads: Optional[Dict[str, int]] = None,
) -> str:
    """
    Pick a deterministic feasible agent for interacting with a target.

    When agent_loads is provided, prefers the least-loaded feasible agent to
    reduce trajectories where one agent does all work while others wait.
    """
    target_room = _resolve_target_room(target_id, target_to_room, restrictions)

    if target_room is None and restrictions:
        raise ValueError(
            f"Cannot assign agent for target '{target_id}': missing room mapping "
            "while room_restriction mechanics are active. Provide scene_data with "
            "furniture/object room mappings."
        )

    feasible = _feasible_agents([target_room], num_agents, restrictions)

    if not feasible:
        raise ValueError(
            f"No agent can reach {target_id} (room={target_room}). "
            f"All agents are restricted. Check room_restriction mechanics."
        )
    if not agent_loads:
        return feasible[0]

    feasible.sort(key=lambda a: _agent_sort_key(a, agent_loads))
    return feasible[0]


def pick_agent_for_targets(
    target_ids: List[str],
    num_agents: int,
    target_to_room: Dict[str, str],
    restrictions: Dict[str, set],
    agent_loads: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """Pick an agent that can reach ALL targets. Returns None if impossible."""
    target_rooms = [
        _resolve_target_room(tid, target_to_room, restrictions)
        for tid in target_ids
    ]
    feasible = _feasible_agents(target_rooms, num_agents, restrictions)
    if not feasible:
        return None
    if not agent_loads:
        return feasible[0]
    feasible.sort(key=lambda a: _agent_sort_key(a, agent_loads))
    return feasible[0]


def find_handoff_furniture(
    agent_a: str,
    agent_b: str,
    restrictions: Dict[str, set],
    scene_data: Optional[Any],
) -> Optional[str]:
    """Find a furniture item in a room accessible by both agents for handoff.

    Returns the first furniture item in a shared room, or None if no shared
    room exists.
    """
    if not scene_data:
        return None

    if hasattr(scene_data, "furniture_in_rooms"):
        fir = scene_data.furniture_in_rooms
    elif isinstance(scene_data, dict):
        fir = scene_data.get("furniture_in_rooms", {})
    else:
        return None

    a_restricted = restrictions.get(agent_a, set())
    b_restricted = restrictions.get(agent_b, set())

    for room in sorted(fir.keys()):
        if room in a_restricted or room in b_restricted:
            continue
        furns = fir.get(room, [])
        if furns:
            return furns[0]
    return None


def wrap_parallel_step(num_agents: int, acting_agent: str, action: str) -> Dict[str, Any]:
    """Create one parallel step with one active agent and Wait for others."""
    actions = []
    for i in range(max(1, num_agents)):
        agent_id = f"agent_{i}"
        actions.append({
            "agent": agent_id,
            "action": action if agent_id == acting_agent else "Wait[]",
        })
    return {"actions": actions}


def extract_plannable_literals(goal_formula: Any) -> List[Any]:
    """
    Extract a deterministic list of literals to satisfy from a PDDL goal.

    For OR branches, picks the lexicographically first branch to keep outputs stable.
    Unwraps epistemic operators and plans against their inner world-state literals.
    """
    from emtom.pddl.dsl import Literal, And, Or, Not, Knows, Believes

    def _unwrap_epistemic(node: Any) -> Any:
        while isinstance(node, (Knows, Believes)):
            node = node.inner
        return node

    def _collect(node: Any) -> List[Literal]:
        node = _unwrap_epistemic(node)
        if isinstance(node, Literal):
            return [node]
        if isinstance(node, Not):
            inner = _unwrap_epistemic(node.operand)
            if isinstance(inner, Literal):
                return [Literal(predicate=inner.predicate, args=inner.args, negated=not inner.negated)]
            return _collect(inner)
        if isinstance(node, And):
            out: List[Literal] = []
            for op in node.operands:
                out.extend(_collect(op))
            return out
        if isinstance(node, Or):
            if not node.operands:
                return []
            chosen = sorted(node.operands, key=lambda op: op.to_pddl())[0]
            return _collect(chosen)
        return []

    literals = _collect(goal_formula)

    deduped = []
    seen = set()
    for lit in literals:
        key = (lit.predicate, lit.args, lit.negated)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(lit)
    return deduped


def apply_literal_ordering(literals: List[Any], ordering: List[Dict[str, str]]) -> List[Any]:
    """Apply topological ordering constraints when they reference extracted literals."""
    if not literals:
        return literals

    literal_strs = [lit.to_pddl() for lit in literals]
    idx_by_str = {s: i for i, s in enumerate(literal_strs)}
    n = len(literals)
    incoming = {i: set() for i in range(n)}
    outgoing = {i: set() for i in range(n)}

    for rule in ordering or []:
        if not isinstance(rule, dict):
            continue
        before = rule.get("before")
        after = rule.get("after")
        if before not in idx_by_str or after not in idx_by_str:
            continue
        bi, ai = idx_by_str[before], idx_by_str[after]
        if bi == ai or ai in outgoing[bi]:
            continue
        outgoing[bi].add(ai)
        incoming[ai].add(bi)

    ready = [i for i in range(n) if not incoming[i]]
    ready.sort()
    ordered_idx = []
    while ready:
        cur = ready.pop(0)
        ordered_idx.append(cur)
        for nxt in sorted(outgoing[cur]):
            incoming[nxt].discard(cur)
            if not incoming[nxt]:
                ready.append(nxt)
        ready.sort()

    if len(ordered_idx) != n:
        return literals
    return [literals[i] for i in ordered_idx]


def generate_deterministic_trajectory(
    task_data: Dict[str, Any],
    scene_data: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build a deterministic golden trajectory from task spec.

    This is a rule-based, non-LLM planner: same spec -> same trajectory.

    Args:
        task_data: Parsed task dict with problem_pddl, mechanic_bindings, etc.
        scene_data: SceneData object or dict with rooms/furniture/objects.

    Returns:
        Dict with keys: trajectory, planned_literals, ignored_literals, planner_notes.
    """
    num_agents = int(task_data.get("num_agents", 2) or 2)
    init_positive_literals = set()

    problem_pddl = task_data.get("problem_pddl")
    if not isinstance(problem_pddl, str) or not problem_pddl.strip():
        raise ValueError(
            "Cannot generate deterministic golden trajectory: missing problem_pddl."
        )
    from emtom.pddl.problem_pddl import parse_problem_pddl

    parsed_problem = parse_problem_pddl(problem_pddl)
    pddl_goal = parsed_problem.goal_pddl
    for init_lit in parsed_problem.init_literals:
        if not getattr(init_lit, "negated", False):
            init_positive_literals.add(
                (init_lit.predicate, tuple(init_lit.args))
            )

    goal = parsed_problem.goal_formula
    has_epistemic_goal = _has_epistemic_goal(goal)
    literals = extract_plannable_literals(goal)

    restrictions = extract_room_restrictions(task_data)
    target_to_room = build_target_to_room_map(scene_data)

    # Build literal-pddl -> owner mapping from :goal-owners.
    literal_owner: Dict[str, str] = {}
    for lit_str, owner in (parsed_problem.owners or {}).items():
        if isinstance(owner, str) and isinstance(lit_str, str):
            literal_owner[lit_str.strip()] = owner

    remote_trigger_by_target: Dict[str, str] = {}
    for binding in task_data.get("mechanic_bindings", []):
        if not isinstance(binding, dict):
            continue
        if binding.get("mechanic_type") != "remote_control":
            continue
        trigger = binding.get("trigger_object")
        target = binding.get("target_object")
        if isinstance(trigger, str) and isinstance(target, str):
            remote_trigger_by_target[target] = trigger

    locked_containers = task_data.get("locked_containers", {})
    if not isinstance(locked_containers, dict):
        locked_containers = {}

    item_location: Dict[str, str] = {}
    for item in task_data.get("items", []):
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id")
        container = item.get("inside") or item.get("hidden_in")
        if isinstance(item_id, str) and isinstance(container, str):
            item_location[item_id] = container

    trajectory: List[Dict[str, Any]] = []
    planned_literals: List[str] = []
    ignored_literals: List[str] = []
    agent_action_queues: Dict[str, List[str]] = {
        f"agent_{i}": [] for i in range(max(1, num_agents))
    }
    agent_loads: Dict[str, int] = {
        f"agent_{i}": 0 for i in range(max(1, num_agents))
    }

    def add_action(agent_id: str, action: str) -> None:
        if agent_id not in agent_action_queues:
            agent_action_queues[agent_id] = []
            agent_loads.setdefault(agent_id, 0)
        agent_action_queues[agent_id].append(action)
        agent_loads[agent_id] = agent_loads.get(agent_id, 0) + 1

    def get_agent_for_literal(lit, target_id: str) -> str:
        """Get agent for a literal, respecting owner field if set."""
        owner = literal_owner.get(lit.to_pddl().strip())
        if owner and owner.startswith("agent_"):
            return owner
        return pick_agent_for_target(
            target_id,
            num_agents,
            target_to_room,
            restrictions,
            agent_loads=agent_loads,
        )

    for lit in literals:
        pred = lit.predicate
        args = list(lit.args)
        lit_str = lit.to_pddl()

        # Skip planning if this positive literal already holds in :init.
        # This avoids pointless moves (and restriction violations) for stable facts,
        # especially K() goals whose inner literal is already true in init.
        if not lit.negated and (pred, tuple(args)) in init_positive_literals:
            planned_literals.append(lit_str)
            continue

        if pred in ("is_open", "is_closed") and args:
            target = args[0]
            agent_id = get_agent_for_literal(lit, target)
            add_action(agent_id, f"Navigate[{target}]")
            if pred == "is_open":
                add_action(agent_id, f"{'Close' if lit.negated else 'Open'}[{target}]")
            else:
                add_action(agent_id, f"{'Open' if lit.negated else 'Close'}[{target}]")
            planned_literals.append(lit_str)
            continue

        if pred == "is_unlocked" and args and not lit.negated:
            target = args[0]
            if target in remote_trigger_by_target:
                trigger = remote_trigger_by_target[target]
                agent_id = get_agent_for_literal(lit, trigger)
                add_action(agent_id, f"Navigate[{trigger}]")
                add_action(agent_id, f"Open[{trigger}]")
                planned_literals.append(lit_str)
                continue

            key_item = locked_containers.get(target)
            if isinstance(key_item, str):
                source = item_location.get(key_item)
                if source:
                    src_agent = get_agent_for_literal(lit, source)
                    add_action(src_agent, f"Navigate[{source}]")
                    add_action(src_agent, f"Open[{source}]")
                dst_agent = get_agent_for_literal(lit, target)
                add_action(dst_agent, f"Navigate[{target}]")
                add_action(dst_agent, f"UseItem[{key_item}, {target}]")
                planned_literals.append(lit_str)
                continue

            fallback_agent = get_agent_for_literal(lit, target)
            add_action(fallback_agent, f"Navigate[{target}]")
            add_action(fallback_agent, f"Open[{target}]")
            planned_literals.append(lit_str)
            continue

        if pred in ("is_on_top", "is_inside") and len(args) >= 2:
            obj, receptacle = args[0], args[1]
            if lit.negated:
                ignored_literals.append(lit_str)
                continue
            if obj.startswith("item_"):
                # Items are inventory-only in this benchmark variant.
                ignored_literals.append(lit_str)
                continue
            relation = "within" if pred == "is_inside" else "on"

            # Check owner override first
            owner = literal_owner.get(lit.to_pddl().strip())
            if owner and owner.startswith("agent_"):
                add_action(owner, f"Navigate[{obj}]")
                add_action(owner, f"Pick[{obj}]")
                add_action(owner, f"Navigate[{receptacle}]")
                add_action(owner, f"Place[{obj}, {relation}, {receptacle}, None, None]")
                planned_literals.append(lit_str)
                continue

            # Try to find a single agent that can reach BOTH obj and receptacle
            single = pick_agent_for_targets(
                [obj, receptacle], num_agents, target_to_room,
                restrictions, agent_loads=agent_loads,
            )
            if single is not None:
                add_action(single, f"Navigate[{obj}]")
                add_action(single, f"Pick[{obj}]")
                add_action(single, f"Navigate[{receptacle}]")
                add_action(single, f"Place[{obj}, {relation}, {receptacle}, None, None]")
                planned_literals.append(lit_str)
                continue

            # No single agent can reach both — use handoff via shared room
            picker = pick_agent_for_target(
                obj, num_agents, target_to_room, restrictions,
                agent_loads=agent_loads,
            )
            placer = pick_agent_for_target(
                receptacle, num_agents, target_to_room, restrictions,
                agent_loads=agent_loads,
            )
            handoff_furn = find_handoff_furniture(
                picker, placer, restrictions, scene_data,
            )
            if handoff_furn is None:
                raise ValueError(
                    f"Cannot plan {lit_str}: no agent can reach both "
                    f"'{obj}' and '{receptacle}', and no shared room exists "
                    "for handoff. Redesign room restrictions or goal."
                )
            # Picker: pick obj and place on handoff furniture
            add_action(picker, f"Navigate[{obj}]")
            add_action(picker, f"Pick[{obj}]")
            add_action(picker, f"Navigate[{handoff_furn}]")
            add_action(picker, f"Place[{obj}, on, {handoff_furn}, None, None]")
            # Placer: pick from handoff and place on final receptacle
            add_action(placer, f"Navigate[{handoff_furn}]")
            add_action(placer, f"Pick[{obj}]")
            add_action(placer, f"Navigate[{receptacle}]")
            add_action(placer, f"Place[{obj}, {relation}, {receptacle}, None, None]")
            planned_literals.append(lit_str)
            continue

        if pred == "agent_in_room" and len(args) >= 2 and not lit.negated:
            agent_str, room = args[0], args[1]
            # Verify agent is not restricted from this room
            if room in restrictions.get(agent_str, set()):
                raise ValueError(
                    f"Cannot plan {lit_str}: {agent_str} is restricted from "
                    f"{room}. Fix room_restriction bindings or goal."
                )
            # Navigate agent to a furniture item in the target room
            nav_target = room
            if isinstance(scene_data, dict):
                fir = scene_data.get("furniture_in_rooms", {})
            elif scene_data and hasattr(scene_data, "furniture_in_rooms"):
                fir = scene_data.furniture_in_rooms
            else:
                fir = {}
            room_furniture = fir.get(room, [])
            if room_furniture:
                nav_target = room_furniture[0]
            add_action(agent_str, f"Navigate[{nav_target}]")
            planned_literals.append(lit_str)
            continue

        if pred == "is_held_by" and len(args) >= 2 and not lit.negated:
            obj, agent_str = args[0], args[1]
            # Verify agent can reach the object's room
            obj_room = target_to_room.get(obj)
            if obj_room and obj_room in restrictions.get(agent_str, set()):
                raise ValueError(
                    f"Cannot plan {lit_str}: {agent_str} is restricted from "
                    f"{obj_room} (where '{obj}' is). Fix room_restriction "
                    "bindings or goal."
                )
            add_action(agent_str, f"Navigate[{obj}]")
            add_action(agent_str, f"Pick[{obj}]")
            planned_literals.append(lit_str)
            continue

        if pred in ("is_clean", "is_filled", "is_powered_on", "is_powered_off") and args:
            target = args[0]
            if lit.negated:
                ignored_literals.append(lit_str)
                continue
            agent_id = get_agent_for_literal(lit, target)
            add_action(agent_id, f"Navigate[{target}]")
            action_map = {
                "is_clean": "Clean",
                "is_filled": "Fill",
                "is_powered_on": "PowerOn",
                "is_powered_off": "PowerOff",
            }
            add_action(agent_id, f"{action_map[pred]}[{target}]")
            planned_literals.append(lit_str)
            continue

        ignored_literals.append(lit_str)

    # Emit parallel trajectory by interleaving per-agent queues.
    max_queue_len = max((len(q) for q in agent_action_queues.values()), default=0)
    if max_queue_len <= 0:
        trajectory = [wrap_parallel_step(num_agents, "agent_0", "Wait[]")]
    else:
        for turn_idx in range(max_queue_len):
            step_actions: List[Dict[str, str]] = []
            for i in range(max(1, num_agents)):
                agent_id = f"agent_{i}"
                queue = agent_action_queues.get(agent_id, [])
                action = queue[turn_idx] if turn_idx < len(queue) else "Wait[]"
                step_actions.append({"agent": agent_id, "action": action})
            trajectory.append({"actions": step_actions})

    # --- Append Communicate steps from FD epistemic compilation ---
    comm_steps: List[Dict[str, Any]] = []
    comm_notes: List[str] = []
    communication_derived = False

    if has_epistemic_goal:
        comm_result = _derive_communicate_steps(
            task_data, scene_data, pddl_goal, num_agents
        )
        comm_steps = comm_result.get("steps", [])
        comm_notes = comm_result.get("notes", [])
        communication_derived = bool(comm_steps)

    if comm_steps:
        trajectory.extend(comm_steps)

    planner_notes = [
        "Deterministic rule-based planner generated trajectory from problem_pddl.",
        "Epistemic wrappers are unwrapped to world-state literals.",
    ]
    if communication_derived:
        planner_notes.append(
            f"Appended {len(comm_steps)} Communicate step(s) from FD epistemic plan."
        )
    planner_notes.extend(comm_notes)

    return {
        "trajectory": trajectory,
        "planned_literals": planned_literals,
        "ignored_literals": ignored_literals,
        "planner_notes": planner_notes,
        "communication_derived": communication_derived,
    }


def regenerate_golden_trajectory(
    task_data: Dict[str, Any],
    scene_data: Optional[Any] = None,
    source: str = "unknown",
    task_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Regenerate golden trajectory deterministically and attach metadata.

    Mutates task_data in-place (sets golden_trajectory + golden_trajectory_metadata).
    Optionally persists to task_file.

    Args:
        task_data: Parsed task dict (mutated in-place).
        scene_data: SceneData object or dict.
        source: Label for what triggered regeneration (e.g. "verify", "submit").
        task_file: If provided, write updated task_data to this path.

    Returns:
        Dict with: spec_hash, trajectory_hash, num_steps, metadata.
    """
    plan_result = generate_deterministic_trajectory(task_data, scene_data)
    trajectory = plan_result["trajectory"]
    task_data["golden_trajectory"] = trajectory

    spec_hash = compute_task_spec_hash(task_data)
    trajectory_hash = hashlib.sha256(
        canonical_json(trajectory).encode("utf-8")
    ).hexdigest()

    metadata = task_data.get("golden_trajectory_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    communication_derived = plan_result.get("communication_derived", False)

    metadata.update({
        "planner": "deterministic_rule_based",
        "planner_version": "v4_strict_no_fallback",
        "source": source,
        "spec_hash": spec_hash,
        "trajectory_hash": trajectory_hash,
        "generated_at": datetime.now().isoformat(),
        "num_steps": len(trajectory),
        "communication_derived": communication_derived,
        "planned_literals": plan_result.get("planned_literals", []),
        "ignored_literals": plan_result.get("ignored_literals", []),
        "planner_notes": plan_result.get("planner_notes", []),
    })
    task_data["golden_trajectory_metadata"] = metadata

    if task_file:
        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)

    return {
        "spec_hash": spec_hash,
        "trajectory_hash": trajectory_hash,
        "num_steps": len(trajectory),
        "metadata": metadata,
    }
