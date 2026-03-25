from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _scene_to_dict(scene_data: Any) -> Dict[str, Any]:
    if isinstance(scene_data, dict):
        return scene_data
    if hasattr(scene_data, "to_dict"):
        return scene_data.to_dict()
    return {}


def _resolve_agent_room(scene: Dict[str, Any], rooms: List[str], agent_id: str, agent_idx: int) -> str | None:
    agent_spawns = scene.get("agent_spawns") or {}
    spawn = agent_spawns.get(agent_id)
    if isinstance(spawn, str) and spawn in rooms:
        return spawn
    if rooms:
        return rooms[agent_idx % len(rooms)]
    return None


def build_scene_bootstrap_problem_pddl(
    scene_data: Any,
    num_agents: int,
    *,
    problem_name: str = "task_problem",
) -> str:
    scene = _scene_to_dict(scene_data)
    rooms = _ordered_unique(scene.get("rooms") or [])
    articulated = set(_ordered_unique(scene.get("articulated_furniture") or []))

    furniture_to_room: Dict[str, str] = {}
    ordered_furniture: List[str] = []
    for room_id in rooms:
        room_furniture = (scene.get("furniture_in_rooms") or {}).get(room_id) or []
        for furniture_id in _ordered_unique(room_furniture):
            if furniture_id in furniture_to_room:
                continue
            furniture_to_room[furniture_id] = room_id
            ordered_furniture.append(furniture_id)

    object_to_furniture: Dict[str, str] = {}
    ordered_objects: List[str] = []
    for furniture_id in ordered_furniture:
        furniture_objects = (scene.get("objects_on_furniture") or {}).get(furniture_id) or []
        for object_id in _ordered_unique(furniture_objects):
            if object_id in object_to_furniture:
                continue
            object_to_furniture[object_id] = furniture_id
            ordered_objects.append(object_id)

    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    agent_rooms = {
        agent_id: _resolve_agent_room(scene, rooms, agent_id, idx)
        for idx, agent_id in enumerate(agent_ids)
    }

    lines = [
        f"(define (problem {problem_name})",
        "  (:domain emtom)",
        "  (:objects",
        f"    {' '.join(agent_ids)} - agent",
    ]
    if rooms:
        lines.append(f"    {' '.join(rooms)} - room")
    if ordered_objects:
        lines.append(f"    {' '.join(ordered_objects)} - object")
    if ordered_furniture:
        lines.append(f"    {' '.join(ordered_furniture)} - furniture")
    lines.extend(
        [
            "  )",
            "  (:init",
        ]
    )

    for agent_id in agent_ids:
        room_id = agent_rooms.get(agent_id)
        if room_id:
            lines.append(f"    (agent_in_room {agent_id} {room_id})")

    for furniture_id in ordered_furniture:
        room_id = furniture_to_room[furniture_id]
        lines.append(f"    (is_in_room {furniture_id} {room_id})")

    for object_id in ordered_objects:
        furniture_id = object_to_furniture[object_id]
        room_id = furniture_to_room.get(furniture_id)
        if room_id:
            lines.append(f"    (is_in_room {object_id} {room_id})")
        lines.append(f"    (is_on_top {object_id} {furniture_id})")

    goal_literals: List[str] = []
    if ordered_objects and ordered_furniture:
        goal_object = ordered_objects[0]
        current_parent = object_to_furniture.get(goal_object)
        target_furniture = next(
            (furniture_id for furniture_id in ordered_furniture if furniture_id != current_parent),
            ordered_furniture[0],
        )
        goal_literals.append(f"(is_on_top {goal_object} {target_furniture})")

    open_target = next((furniture_id for furniture_id in ordered_furniture if furniture_id in articulated), None)
    if open_target:
        goal_literals.append(f"(is_open {open_target})")

    if not goal_literals:
        fallback_agent = agent_ids[0] if agent_ids else None
        fallback_room = agent_rooms.get(fallback_agent) if fallback_agent else None
        if fallback_agent and fallback_room:
            goal_literals.append(f"(agent_in_room {fallback_agent} {fallback_room})")
        elif rooms:
            goal_literals.append(f"(is_in_room {ordered_furniture[0]} {rooms[0]})" if ordered_furniture else f"(agent_in_room agent_0 {rooms[0]})")
        else:
            goal_literals.append("(and)")

    if len(goal_literals) == 1:
        goal_pddl = goal_literals[0]
    else:
        goal_pddl = "(and " + " ".join(goal_literals) + ")"

    lines.extend(
        [
            "  )",
            f"  (:goal {goal_pddl})",
            ")",
        ]
    )
    return "\n".join(lines)

