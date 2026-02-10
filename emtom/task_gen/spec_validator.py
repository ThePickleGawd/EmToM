"""
Shared deterministic task-spec validation used by generation and CLI verification.

Phase 1 scope:
- Template placeholder detection
- Mechanic field/schema checks
- Mechanic binding completeness checks
- Mechanic binding scene-reference checks
- Category/schema consistency checks
- Basic golden-trajectory structural checks
- Room-restriction vs Navigate trajectory checks
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from emtom.mechanics.handlers import MECHANIC_INFO


_ACTION_PATTERN = re.compile(r"(\w+)(?:\[(.*)\])?$")
_PLACEHOLDER_PATTERNS = (
    "REPLACE_WITH_",
    "REPLACE_CONTAINER",
    "REPLACE_ITEM",
    "EXAMPLE_",
    "ACTION_NAME[TARGET]",
)

_VALID_ACTIONS = {
    "Navigate", "Open", "Close", "Pick", "Place",
    "UseItem", "Communicate", "Wait", "Clean", "Pour", "PowerOn", "PowerOff",
    "Fill", "FindObjectTool", "FindReceptacleTool", "FindRoomTool",
}


def _get_scene_list(scene_data: Optional[Any], key: str) -> List[str]:
    if not scene_data:
        return []
    if isinstance(scene_data, dict):
        value = scene_data.get(key, [])
    else:
        value = getattr(scene_data, key, [])
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


def _get_scene_dict(scene_data: Optional[Any], key: str) -> Dict[str, List[str]]:
    if not scene_data:
        return {}
    if isinstance(scene_data, dict):
        value = scene_data.get(key, {})
    else:
        value = getattr(scene_data, key, {})
    if not isinstance(value, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in value.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [x for x in v if isinstance(x, str)]
    return out


def _collect_placeholder_hits(obj: Any, path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(obj, str):
        if any(p in obj for p in _PLACEHOLDER_PATTERNS):
            hits.append(path or "$")
        return hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if any(p in k for p in _PLACEHOLDER_PATTERNS):
                key_path = f"{path}.{k}" if path else k
                hits.append(f"{key_path} (key)")
            child = f"{path}.{k}" if path else k
            hits.extend(_collect_placeholder_hits(v, child))
        return hits
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            child = f"{path}[{i}]" if path else f"[{i}]"
            hits.extend(_collect_placeholder_hits(v, child))
    return hits


def _parse_action(action_str: str) -> Tuple[Optional[str], Optional[str]]:
    m = _ACTION_PATTERN.match(action_str or "")
    if not m:
        return None, None
    action_name, args = m.group(1), m.group(2)
    if args == "":
        args = None
    return action_name, args


def _extract_defined_items(task_data: Dict[str, Any]) -> Set[str]:
    item_ids: Set[str] = set()
    for item in task_data.get("items", []):
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id")
        if isinstance(item_id, str) and item_id:
            item_ids.add(item_id)
    return item_ids


def _extract_room_restrictions(
    task_data: Dict[str, Any],
) -> Dict[str, Set[str]]:
    restrictions: Dict[str, Set[str]] = {}

    def add_binding(binding: Dict[str, Any]) -> None:
        if not isinstance(binding, dict):
            return
        if binding.get("mechanic_type") != "room_restriction":
            return
        rooms = binding.get("restricted_rooms", [])
        agents = binding.get("for_agents", [])
        if not isinstance(rooms, list) or not isinstance(agents, list):
            return
        room_set = {r for r in rooms if isinstance(r, str) and r}
        for agent in agents:
            if isinstance(agent, str) and agent:
                restrictions.setdefault(agent, set()).update(room_set)

    for binding in task_data.get("mechanic_bindings", []):
        if isinstance(binding, dict):
            add_binding(binding)

    # Be tolerant of swapped-fields legacy tasks: dicts in active_mechanics.
    for mech in task_data.get("active_mechanics", []):
        if isinstance(mech, dict):
            add_binding(mech)

    return restrictions


def _build_target_to_room(scene_data: Optional[Any]) -> Dict[str, str]:
    target_to_room: Dict[str, str] = {}
    if not scene_data:
        return target_to_room

    rooms = set(_get_scene_list(scene_data, "rooms"))
    furniture_in_rooms = _get_scene_dict(scene_data, "furniture_in_rooms")
    objects_on_furniture = _get_scene_dict(scene_data, "objects_on_furniture")

    for room in rooms:
        target_to_room[room] = room

    furniture_to_room: Dict[str, str] = {}
    for room, furns in furniture_in_rooms.items():
        for furn in furns:
            furniture_to_room[furn] = room
            target_to_room[furn] = room

    for furn, objs in objects_on_furniture.items():
        room = furniture_to_room.get(furn)
        if not room:
            continue
        for obj in objs:
            target_to_room[obj] = room

    return target_to_room


def validate_room_restriction_trajectory(
    task_data: Dict[str, Any],
    scene_data: Optional[Any],
    golden: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    errors: List[str] = []
    restrictions = _extract_room_restrictions(task_data)
    if not restrictions:
        return errors
    if not scene_data:
        return errors

    if golden is None:
        golden = task_data.get("golden_trajectory", [])
    if not isinstance(golden, list):
        return errors

    target_to_room = _build_target_to_room(scene_data)
    if not target_to_room:
        return errors

    for step_idx, step in enumerate(golden):
        if not isinstance(step, dict):
            continue
        actions = step.get("actions", [])
        if not isinstance(actions, list):
            continue
        for action_entry in actions:
            if not isinstance(action_entry, dict):
                continue
            agent = action_entry.get("agent", "")
            action_str = action_entry.get("action", "")
            if not isinstance(agent, str) or not isinstance(action_str, str):
                continue
            action_name, args = _parse_action(action_str)
            if action_name != "Navigate" or not args:
                continue
            target = args.split(",")[0].strip()
            target_room = target_to_room.get(target)
            if not target_room:
                continue
            if target_room in restrictions.get(agent, set()):
                errors.append(
                    f"Step {step_idx}: {agent} navigates to '{target}' "
                    f"(in {target_room}) but is restricted from {target_room}."
                )
    return errors


def validate_blocking_spec(
    task_data: Dict[str, Any],
    scene_data: Optional[Any] = None,
) -> List[str]:
    errors: List[str] = []

    # ------------------------------------------------------------------
    # Basic schema/category sanity
    # ------------------------------------------------------------------
    required_fields = [
        "task_id", "title", "task", "episode_id",
        "mechanic_bindings", "agent_secrets", "agent_actions",
    ]
    missing = [f for f in required_fields if f not in task_data]
    if missing:
        errors.append(f"Missing required fields: {missing}")
        # Continue so callers can still get other deterministic errors when possible.

    category = task_data.get("category")
    if isinstance(category, str) and category:
        if category not in {"cooperative", "competitive", "mixed"}:
            errors.append(
                "category must be one of ['cooperative', 'competitive', 'mixed']"
            )
        if category == "cooperative":
            if task_data.get("teams"):
                errors.append("cooperative task should not include 'teams'")
            if task_data.get("team_secrets"):
                errors.append("cooperative task should not include 'team_secrets'")

    # ------------------------------------------------------------------
    # Template placeholder artifacts
    # ------------------------------------------------------------------
    placeholder_hits = _collect_placeholder_hits(task_data)
    if placeholder_hits:
        shown = placeholder_hits[:12]
        more = "" if len(placeholder_hits) <= 12 else f" (+{len(placeholder_hits)-12} more)"
        errors.append(
            f"Unfilled template placeholders in: {shown}{more}. "
            "Replace all REPLACE_WITH_* and template placeholders."
        )

    # ------------------------------------------------------------------
    # Agent ID consistency
    # ------------------------------------------------------------------
    num_agents = task_data.get("num_agents", 2)
    if not isinstance(num_agents, int) or num_agents <= 0:
        errors.append("num_agents must be a positive integer")
        num_agents = 2
    valid_agent_ids = {f"agent_{i}" for i in range(num_agents)}

    for field_name in ("agent_actions", "agent_secrets"):
        field_val = task_data.get(field_name, {})
        if not isinstance(field_val, dict):
            errors.append(f"{field_name} must be a dict")
            continue
        for agent_id in field_val.keys():
            if agent_id not in valid_agent_ids:
                errors.append(
                    f"{field_name} contains invalid agent ID '{agent_id}'. "
                    f"Valid IDs: {sorted(valid_agent_ids)}"
                )

    # ------------------------------------------------------------------
    # Scene inventory
    # ------------------------------------------------------------------
    rooms = set(_get_scene_list(scene_data, "rooms"))
    furniture = set(_get_scene_list(scene_data, "furniture"))
    objects = set(_get_scene_list(scene_data, "objects"))
    articulated = set(_get_scene_list(scene_data, "articulated_furniture"))
    item_ids = _extract_defined_items(task_data)
    scene_known_ids = rooms | furniture | objects | item_ids

    # ------------------------------------------------------------------
    # Mechanics schema and binding checks
    # ------------------------------------------------------------------
    active_mechanics = task_data.get("active_mechanics", [])
    mechanic_bindings = task_data.get("mechanic_bindings", [])

    if active_mechanics is not None and not isinstance(active_mechanics, list):
        errors.append("active_mechanics must be a list")
        active_mechanics = []
    if mechanic_bindings is not None and not isinstance(mechanic_bindings, list):
        errors.append("mechanic_bindings must be a list")
        mechanic_bindings = []

    if isinstance(active_mechanics, list):
        dict_like = sum(1 for x in active_mechanics if isinstance(x, dict))
        if dict_like > 0:
            if not mechanic_bindings:
                errors.append(
                    "Detected swapped mechanic fields: binding dicts found in active_mechanics while mechanic_bindings is empty."
                )

    for i, binding in enumerate(mechanic_bindings if isinstance(mechanic_bindings, list) else []):
        if not isinstance(binding, dict):
            errors.append(f"mechanic_bindings[{i}] must be an object")
            continue

        mechanic_type = binding.get("mechanic_type")
        if not isinstance(mechanic_type, str) or not mechanic_type:
            errors.append(f"mechanic_bindings[{i}] missing mechanic_type")
            continue

        info = MECHANIC_INFO.get(mechanic_type)
        if not info:
            errors.append(f"mechanic_bindings[{i}] has unknown mechanic_type '{mechanic_type}'")
            continue

        # Required keys by mechanic.
        missing_required: List[str] = []
        if mechanic_type == "conditional_unlock":
            if not binding.get("trigger_object"):
                missing_required.append("trigger_object")
            if not (binding.get("prerequisite_object") or binding.get("requires_item")):
                missing_required.append("prerequisite_object|requires_item")
        else:
            required_keys = info.get("setup_keys", [])
            for key in required_keys:
                val = binding.get(key)
                if val is None or (isinstance(val, str) and not val):
                    missing_required.append(key)

        if missing_required:
            errors.append(
                f"mechanic_bindings[{i}] ({mechanic_type}) missing required fields: {missing_required}"
            )

        # Validate room_restriction structure.
        if mechanic_type == "room_restriction":
            rr = binding.get("restricted_rooms")
            fa = binding.get("for_agents")
            if not isinstance(rr, list) or not rr:
                errors.append(
                    f"mechanic_bindings[{i}] (room_restriction) requires non-empty restricted_rooms list"
                )
            if not isinstance(fa, list) or not fa:
                errors.append(
                    f"mechanic_bindings[{i}] (room_restriction) requires non-empty for_agents list"
                )
            if isinstance(rr, list) and rooms:
                unknown_rooms = [r for r in rr if isinstance(r, str) and r not in rooms]
                if unknown_rooms:
                    errors.append(
                        f"mechanic_bindings[{i}] (room_restriction) unknown restricted_rooms: {unknown_rooms}"
                    )
            if isinstance(fa, list):
                unknown_agents = [a for a in fa if isinstance(a, str) and a not in valid_agent_ids]
                if unknown_agents:
                    errors.append(
                        f"mechanic_bindings[{i}] (room_restriction) unknown for_agents: {unknown_agents}"
                    )

        # Validate binding object references against scene (when available).
        if scene_known_ids:
            for key in ("trigger_object", "target_object", "prerequisite_object"):
                val = binding.get(key)
                if isinstance(val, str) and val and val not in scene_known_ids:
                    errors.append(
                        f"mechanic_bindings[{i}] ({mechanic_type}): {key}='{val}' not found in scene."
                    )
            req_item = binding.get("requires_item")
            if isinstance(req_item, str) and req_item and req_item not in item_ids:
                errors.append(
                    f"mechanic_bindings[{i}] ({mechanic_type}): requires_item='{req_item}' not found in items."
                )

    # ------------------------------------------------------------------
    # Golden trajectory structural checks
    # ------------------------------------------------------------------
    golden = task_data.get("golden_trajectory", [])
    if not isinstance(golden, list) or not golden:
        errors.append("No golden_trajectory found in task. Add a golden_trajectory field.")
        return errors

    for step_idx, step in enumerate(golden):
        if not isinstance(step, dict):
            errors.append(f"golden_trajectory[{step_idx}] must be an object")
            continue
        actions = step.get("actions", [])
        if not isinstance(actions, list) or not actions:
            errors.append(f"golden_trajectory[{step_idx}] missing non-empty actions list")
            continue

        seen_agents: Set[str] = set()
        for action_idx, action_entry in enumerate(actions):
            if not isinstance(action_entry, dict):
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] must be an object"
                )
                continue
            agent = action_entry.get("agent")
            action_str = action_entry.get("action")
            if not isinstance(agent, str) or agent not in valid_agent_ids:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] has invalid agent '{agent}'"
                )
            if isinstance(agent, str):
                if agent in seen_agents:
                    errors.append(f"golden_trajectory[{step_idx}] has duplicate action for {agent}")
                seen_agents.add(agent)

            if not isinstance(action_str, str) or not action_str:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] missing action string"
                )
                continue
            action_name, args = _parse_action(action_str)
            if not action_name:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] has malformed action '{action_str}'"
                )
                continue
            if action_name not in _VALID_ACTIONS:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] unknown action '{action_name}'"
                )
                continue

            # Open/Close on non-articulated furniture is always invalid.
            if action_name in {"Open", "Close"} and args and furniture and articulated:
                target = args.split(",")[0].strip()
                if target in furniture and target not in articulated:
                    errors.append(
                        f"golden_trajectory[{step_idx}] uses {action_name}[{target}] "
                        f"but {target} is not articulated/openable."
                    )

    # Room restriction consistency against trajectory Navigate actions.
    errors.extend(validate_room_restriction_trajectory(task_data, scene_data, golden))

    return errors
