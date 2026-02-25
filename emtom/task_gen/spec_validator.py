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


def _has_ordering_cycle(ordering: List[Dict[str, Any]]) -> bool:
    """
    Check if pddl_ordering constraints form a cycle.

    Uses DFS with recursion stack (same pattern as dag._has_cycle).
    """
    # Build adjacency list
    graph: Dict[str, Set[str]] = {}
    nodes: Set[str] = set()
    for constraint in ordering:
        if not isinstance(constraint, dict):
            continue
        before = constraint.get("before", "")
        after = constraint.get("after", "")
        if before and after:
            graph.setdefault(before, set()).add(after)
            nodes.add(before)
            nodes.add(after)

    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True  # Back edge = cycle
        rec_stack.discard(node)
        return False

    for node in nodes:
        if node not in visited:
            if dfs(node):
                return True
    return False


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
    from emtom.pddl.planner import build_target_to_room_map
    return build_target_to_room_map(scene_data)


def _iter_formula_nodes(formula: Any):
    """Depth-first walk of DSL formulas."""
    if formula is None:
        return
    yield formula
    inner = getattr(formula, "inner", None)
    if inner is not None:
        yield from _iter_formula_nodes(inner)
    operand = getattr(formula, "operand", None)
    if operand is not None:
        yield from _iter_formula_nodes(operand)
    operands = getattr(formula, "operands", None)
    if isinstance(operands, tuple):
        for op in operands:
            yield from _iter_formula_nodes(op)


def _collect_literals(formula: Any) -> List[Any]:
    literals: List[Any] = []
    for node in _iter_formula_nodes(formula):
        if node.__class__.__name__ == "Literal":
            literals.append(node)
    return literals


def _collect_epistemic_goals(formula: Any) -> List[Any]:
    goals: List[Any] = []
    for node in _iter_formula_nodes(formula):
        if node.__class__.__name__ in {"Knows", "Believes"}:
            goals.append(node)
    return goals


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
        if category == "competitive":
            teams = task_data.get("teams")
            if not isinstance(teams, dict) or len(teams) < 2:
                errors.append("competitive task must define at least two teams in 'teams'.")
            else:
                assigned: Set[str] = set()
                for team_id, members in teams.items():
                    if not isinstance(team_id, str) or not team_id:
                        errors.append("teams must have non-empty string team IDs")
                        continue
                    if not isinstance(members, list) or not members:
                        errors.append(f"teams['{team_id}'] must be a non-empty list of agent IDs")
                        continue
                    for member in members:
                        if not isinstance(member, str):
                            errors.append(f"teams['{team_id}'] contains non-string member '{member}'")
                            continue
                        if member in assigned:
                            errors.append(f"Agent '{member}' appears in multiple teams")
                        assigned.add(member)

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
    if category == "competitive":
        teams = task_data.get("teams")
        if isinstance(teams, dict):
            for team_id, members in teams.items():
                if not isinstance(members, list):
                    continue
                for member in members:
                    if isinstance(member, str) and member not in valid_agent_ids:
                        errors.append(
                            f"teams['{team_id}'] contains invalid agent ID '{member}'. "
                            f"Valid IDs: {sorted(valid_agent_ids)}"
                        )

    # ------------------------------------------------------------------
    # Message targets validation
    # ------------------------------------------------------------------
    raw_mt = task_data.get("message_targets")
    if raw_mt is not None:
        if not isinstance(raw_mt, dict):
            errors.append("message_targets must be a dict")
        else:
            for mt_agent, mt_targets in raw_mt.items():
                if mt_agent not in valid_agent_ids:
                    errors.append(
                        f"message_targets key '{mt_agent}' is not a valid agent ID. "
                        f"Valid: {sorted(valid_agent_ids)}"
                    )
                if not isinstance(mt_targets, list):
                    errors.append(f"message_targets['{mt_agent}'] must be a list of agent IDs")
                    continue
                for target_id in mt_targets:
                    if not isinstance(target_id, str) or target_id not in valid_agent_ids:
                        errors.append(
                            f"message_targets['{mt_agent}'] contains invalid agent ID '{target_id}'. "
                            f"Valid: {sorted(valid_agent_ids)}"
                        )
                    elif target_id == mt_agent:
                        errors.append(
                            f"message_targets['{mt_agent}'] contains self-reference"
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

        # Validate limited_bandwidth structure.
        if mechanic_type == "limited_bandwidth":
            ml = binding.get("message_limits")
            if not isinstance(ml, dict) or not ml:
                errors.append(
                    f"mechanic_bindings[{i}] (limited_bandwidth) requires non-empty message_limits dict"
                )
            elif isinstance(ml, dict):
                for agent_id, limit in ml.items():
                    if agent_id not in valid_agent_ids:
                        errors.append(
                            f"mechanic_bindings[{i}] (limited_bandwidth) unknown agent '{agent_id}' in message_limits"
                        )
                    if not isinstance(limit, (int, float)) or limit < 1:
                        errors.append(
                            f"mechanic_bindings[{i}] (limited_bandwidth) message_limits[{agent_id}] must be a positive integer, got {limit}"
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

    mechanic_types = {
        b.get("mechanic_type")
        for b in (mechanic_bindings or [])
        if isinstance(b, dict) and isinstance(b.get("mechanic_type"), str)
    }
    if raw_mt and "restricted_communication" not in mechanic_types:
        errors.append(
            "message_targets is set but restricted_communication mechanic is missing. "
            "Add restricted_communication so messaging constraints are simulator-enforced."
        )
    if category in {"cooperative", "mixed"} and num_agents > 1:
        asymmetry_mechanics = {
            "room_restriction",
            "restricted_communication",
            "unreliable_communication",
            "conditional_unlock",
            "remote_control",
        }
        if mechanic_types and not (mechanic_types & asymmetry_mechanics):
            errors.append(
                "cooperative/mixed tasks need at least one asymmetry mechanic "
                "(room_restriction, restricted_communication, unreliable_communication, "
                "conditional_unlock, or remote_control)."
            )

    # ------------------------------------------------------------------
    # PDDL validation (single-format + transitional formats)
    # ------------------------------------------------------------------
    problem_pddl = task_data.get("problem_pddl")
    goals = task_data.get("goals")
    pddl_goal = task_data.get("pddl_goal")
    has_problem_pddl = isinstance(problem_pddl, str) and bool(problem_pddl.strip())
    has_goals = isinstance(goals, list) and bool(goals)
    has_pddl_goal = isinstance(pddl_goal, str) and bool(pddl_goal)
    parsed_problem = None

    if has_problem_pddl:
        # Canonical format should not be mixed with legacy goal fields.
        mixed_fields = []
        if has_goals:
            mixed_fields.append("goals")
        if has_pddl_goal:
            mixed_fields.append("pddl_goal")
        if mixed_fields:
            errors.append(
                "problem_pddl cannot be combined with legacy goal fields: "
                f"{mixed_fields}"
            )

        try:
            from emtom.pddl.domain import EMTOM_DOMAIN
            from emtom.pddl.goal_spec import GoalSpec
            from emtom.pddl.problem_pddl import parse_problem_pddl

            parsed_problem = parse_problem_pddl(problem_pddl)
            declared_domain = task_data.get("pddl_domain")
            if isinstance(declared_domain, str) and declared_domain:
                if parsed_problem.domain_name != declared_domain:
                    errors.append(
                        "problem_pddl domain mismatch: "
                        f":domain is '{parsed_problem.domain_name}' but pddl_domain is '{declared_domain}'"
                    )
            if parsed_problem.domain_name != EMTOM_DOMAIN.name:
                errors.append(
                    f"Unsupported problem domain '{parsed_problem.domain_name}'. "
                    f"Expected '{EMTOM_DOMAIN.name}'."
                )

            spec = GoalSpec.from_legacy(parsed_problem.goal_pddl, [], {})
            spec_errors = spec.validate(EMTOM_DOMAIN, valid_agent_ids)
            errors.extend(spec_errors)

            # Category-level structural lint for competitive goals.
            goal_lower = parsed_problem.goal_pddl.lower()
            has_or = any(n.__class__.__name__ == "Or" for n in _iter_formula_nodes(parsed_problem.goal_formula))
            has_not = "(not" in goal_lower

            # has_most/has_at_least are not fully modeled in deterministic PDDL planning.
            if "has_most" in goal_lower or "has_at_least" in goal_lower:
                errors.append(
                    "problem_pddl goal uses has_most/has_at_least, which are not supported for "
                    "deterministic PDDL solvability checks. Use explicit object-state opposition "
                    "with concrete literals (e.g., OR of mutually exclusive branches)."
                )

            if category == "competitive":
                if not has_or:
                    errors.append(
                        "competitive problem_pddl goal must encode opposed win branches "
                        "(e.g., (or (and team_0_win (not team_1_win)) "
                        "(and team_1_win (not team_0_win))))."
                    )
                if has_or and not has_not:
                    errors.append(
                        "competitive OR goal lacks explicit exclusivity. Add (not ...) constraints "
                        "so exactly one team can satisfy a winning branch."
                    )

            # Epistemic-goal backing: K/B goals benefit from observation
            # barriers (room_restriction, communication mechanics, etc.) but
            # this is evaluated by the judge, not enforced as a hard error
            # here.  Hard-blocking was removed because it created a catch-22
            # with agent necessity in cooperative tasks.

            # Mixed tasks must have per-agent personal objectives in :goal-owners
            if category == "mixed" and not parsed_problem.owners:
                errors.append(
                    "mixed task problem_pddl is missing :goal-owners section. "
                    "Each agent needs a personal objective for credit assignment."
                )
        except Exception as e:
            errors.append(f"problem_pddl validation error: {e}")

    elif has_goals:
        try:
            from emtom.pddl.goal_spec import GoalSpec
            from emtom.pddl.domain import EMTOM_DOMAIN
            spec = GoalSpec.from_goals_array(goals)

            # Validate against domain and agents
            spec_errors = spec.validate(EMTOM_DOMAIN, valid_agent_ids)
            errors.extend(spec_errors)

            # Warn if ordering is empty with multi-goal specs
            if len(spec.entries) > 1 and all(not e.after for e in spec.entries):
                errors.append(
                    "goals: All entries have empty 'after' but there are multiple goals. "
                    "Add ordering constraints to define dependencies between goals."
                )

            # Warn if owners empty for competitive/mixed tasks
            if isinstance(category, str) and category in ("competitive", "mixed"):
                has_owners = any(e.owner for e in spec.entries)
                if not has_owners:
                    errors.append(
                        f"goals: No entries have 'owner' set for {category} task. "
                        f"Assign goals to teams/agents via the 'owner' field."
                    )
        except ValueError as e:
            errors.append(f"Invalid goals array: {e}")
        except Exception as e:
            errors.append(f"Goals validation error: {e}")

    elif has_pddl_goal:
        try:
            from emtom.pddl.dsl import parse_goal_string, validate_goal_predicates, Knows, Believes
            from emtom.pddl.domain import EMTOM_DOMAIN
            goal = parse_goal_string(pddl_goal)
            conjuncts = goal.flatten()
            if not conjuncts:
                errors.append("pddl_goal parsed but contains no goal conjuncts")

            # Validate predicate arities against domain
            arity_errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
            errors.extend(arity_errors)

            # Validate K/B agent names reference valid agent IDs
            def _check_epistemic_agents(formula, path="pddl_goal"):
                if isinstance(formula, (Knows, Believes)):
                    if formula.agent not in valid_agent_ids:
                        errors.append(
                            f"{path}: epistemic operator references invalid agent '{formula.agent}'. "
                            f"Valid IDs: {sorted(valid_agent_ids)}"
                        )
                    _check_epistemic_agents(formula.inner, path)
                elif hasattr(formula, 'operands'):
                    for op in formula.operands:
                        _check_epistemic_agents(op, path)
                elif hasattr(formula, 'operand') and formula.operand is not None:
                    _check_epistemic_agents(formula.operand, path)
            _check_epistemic_agents(goal)

            # Validate ordering references valid goal conjuncts
            conjunct_strs = {c.to_pddl() for c in conjuncts}
            pddl_ordering = task_data.get("pddl_ordering", [])
            if isinstance(pddl_ordering, list):
                for i, constraint in enumerate(pddl_ordering):
                    if not isinstance(constraint, dict):
                        errors.append(f"pddl_ordering[{i}] must be an object")
                        continue
                    for key in ("before", "after"):
                        ref = constraint.get(key)
                        if ref and ref not in conjunct_strs:
                            errors.append(
                                f"pddl_ordering[{i}].{key} references '{ref}' "
                                f"which is not a goal conjunct"
                            )

                # Check for cycles in ordering
                if pddl_ordering and _has_ordering_cycle(pddl_ordering):
                    errors.append(
                        "pddl_ordering contains a cycle. "
                        "Ordering constraints must form a DAG."
                    )

                # Warn if ordering is empty with multi-conjunct goals
                if len(conjuncts) > 1 and not pddl_ordering:
                    errors.append(
                        "pddl_ordering is empty but goal has multiple conjuncts. "
                        "Add ordering constraints to define dependencies between goals."
                    )

            # Validate pddl_owners references valid goal conjuncts
            # (mixed tasks allow supplementary goals not in :goal)
            pddl_owners = task_data.get("pddl_owners", {})
            if isinstance(pddl_owners, dict):
                for literal_str, owner in pddl_owners.items():
                    if literal_str.startswith("_"):
                        continue  # Skip comment keys
                    if literal_str not in conjunct_strs and category != "mixed":
                        errors.append(
                            f"pddl_owners key '{literal_str}' is not a goal conjunct"
                        )

            # Warn if owners empty for competitive/mixed tasks
            if isinstance(category, str) and category in ("competitive", "mixed"):
                real_owners = {k: v for k, v in (pddl_owners or {}).items()
                              if not k.startswith("_")}
                if not real_owners:
                    errors.append(
                        f"pddl_owners is empty for {category} task. "
                        f"Assign goals to teams/agents via pddl_owners."
                    )

        except Exception as e:
            errors.append(f"Invalid pddl_goal syntax: {e}")

    # ------------------------------------------------------------------
    # Golden trajectory structural checks (optional derived artifact)
    # ------------------------------------------------------------------
    golden = task_data.get("golden_trajectory")
    if golden is None:
        return errors
    if not isinstance(golden, list):
        errors.append("golden_trajectory must be a list when provided.")
        return errors
    if not golden:
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
            allowed_by_agent = task_data.get("agent_actions", {}).get(agent)
            if isinstance(allowed_by_agent, list) and action_name not in allowed_by_agent:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] uses '{action_name}' "
                    f"for {agent}, but it is not in agent_actions[{agent}]"
                )

            # Open/Close on non-articulated furniture is always invalid.
            if action_name in {"Open", "Close"} and args and furniture and articulated:
                target = args.split(",")[0].strip()
                if target in furniture and target not in articulated:
                    errors.append(
                        f"golden_trajectory[{step_idx}] uses {action_name}[{target}] "
                        f"but {target} is not articulated/openable."
                    )

    # Multi-agent quality guard: avoid trajectories where one agent does all work.
    if len(valid_agent_ids) > 1:
        non_wait_counts: Dict[str, int] = {agent_id: 0 for agent_id in valid_agent_ids}
        for step in golden:
            actions = step.get("actions", []) if isinstance(step, dict) else []
            if not isinstance(actions, list):
                continue
            for entry in actions:
                if not isinstance(entry, dict):
                    continue
                agent = entry.get("agent")
                action_str = entry.get("action")
                if (
                    isinstance(agent, str)
                    and agent in non_wait_counts
                    and isinstance(action_str, str)
                    and action_str != "Wait[]"
                ):
                    non_wait_counts[agent] += 1
        active_agents = [a for a, c in non_wait_counts.items() if c > 0]
        if len(active_agents) <= 1:
            errors.append(
                "golden_trajectory has only one active agent (others only Wait[]). "
                "Distribute required actions across multiple agents."
            )

    # Room restriction consistency against trajectory Navigate actions.
    errors.extend(validate_room_restriction_trajectory(task_data, scene_data, golden))

    return errors
