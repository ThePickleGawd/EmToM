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
from datetime import datetime
from typing import Any, Dict, List, Optional


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
        "pddl_goal", "pddl_ordering", "pddl_owners",
        "items", "locked_containers", "initial_states",
        "message_targets", "teams", "team_secrets",
        "success_condition", "subtasks", "agent_spawns",
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


def pick_agent_for_target(
    target_id: str,
    num_agents: int,
    target_to_room: Dict[str, str],
    restrictions: Dict[str, set],
) -> str:
    """Pick a deterministic feasible agent for interacting with a target."""
    target_room = target_to_room.get(target_id)
    for i in range(max(1, num_agents)):
        agent_id = f"agent_{i}"
        if target_room and target_room in restrictions.get(agent_id, set()):
            continue
        return agent_id
    return "agent_0"


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
        task_data: Parsed task dict with pddl_goal, mechanic_bindings, etc.
        scene_data: SceneData object or dict with rooms/furniture/objects.

    Returns:
        Dict with keys: trajectory, planned_literals, ignored_literals, planner_notes.
    """
    num_agents = int(task_data.get("num_agents", 2) or 2)
    pddl_goal = task_data.get("pddl_goal")

    if not pddl_goal:
        existing = task_data.get("golden_trajectory")
        if isinstance(existing, list) and existing:
            return {
                "trajectory": existing,
                "planned_literals": [],
                "ignored_literals": [],
                "planner_notes": ["No pddl_goal; preserved existing golden_trajectory."],
            }
        return {
            "trajectory": [wrap_parallel_step(num_agents, "agent_0", "Wait[]")],
            "planned_literals": [],
            "ignored_literals": [],
            "planner_notes": ["No pddl_goal; emitted single Wait[] step."],
        }

    from emtom.pddl.dsl import parse_goal_string

    goal = parse_goal_string(pddl_goal)
    literals = extract_plannable_literals(goal)
    literals = apply_literal_ordering(literals, task_data.get("pddl_ordering", []))

    restrictions = extract_room_restrictions(task_data)
    target_to_room = build_target_to_room_map(scene_data)

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

    def add_action(agent_id: str, action: str) -> None:
        trajectory.append(wrap_parallel_step(num_agents, agent_id, action))

    for lit in literals:
        pred = lit.predicate
        args = list(lit.args)
        lit_str = lit.to_pddl()

        if pred in ("is_open", "is_closed") and args:
            target = args[0]
            agent_id = pick_agent_for_target(target, num_agents, target_to_room, restrictions)
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
                agent_id = pick_agent_for_target(trigger, num_agents, target_to_room, restrictions)
                add_action(agent_id, f"Navigate[{trigger}]")
                add_action(agent_id, f"Open[{trigger}]")
                planned_literals.append(lit_str)
                continue

            key_item = locked_containers.get(target)
            if isinstance(key_item, str):
                source = item_location.get(key_item)
                if source:
                    src_agent = pick_agent_for_target(source, num_agents, target_to_room, restrictions)
                    add_action(src_agent, f"Navigate[{source}]")
                    add_action(src_agent, f"Open[{source}]")
                dst_agent = pick_agent_for_target(target, num_agents, target_to_room, restrictions)
                add_action(dst_agent, f"Navigate[{target}]")
                add_action(dst_agent, f"UseItem[{key_item}, {target}]")
                planned_literals.append(lit_str)
                continue

            fallback_agent = pick_agent_for_target(target, num_agents, target_to_room, restrictions)
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
            agent_id = pick_agent_for_target(obj, num_agents, target_to_room, restrictions)
            add_action(agent_id, f"Navigate[{obj}]")
            add_action(agent_id, f"Pick[{obj}]")
            add_action(agent_id, f"Navigate[{receptacle}]")
            relation = "within" if pred == "is_inside" else "on"
            add_action(agent_id, f"Place[{obj}, {relation}, {receptacle}, None, None]")
            planned_literals.append(lit_str)
            continue

        ignored_literals.append(lit_str)

    if not trajectory:
        trajectory = [wrap_parallel_step(num_agents, "agent_0", "Wait[]")]

    return {
        "trajectory": trajectory,
        "planned_literals": planned_literals,
        "ignored_literals": ignored_literals,
        "planner_notes": [
            "Deterministic rule-based planner generated trajectory from pddl_goal.",
            "Epistemic wrappers are unwrapped to world-state literals.",
        ],
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

    metadata.update({
        "planner": "deterministic_rule_based",
        "planner_version": "v1",
        "source": source,
        "spec_hash": spec_hash,
        "trajectory_hash": trajectory_hash,
        "generated_at": datetime.now().isoformat(),
        "num_steps": len(trajectory),
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
