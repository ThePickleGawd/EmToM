"""
Tool Wrapper for EMTOM.

Wraps Habitat tools with condition checks (locks, prerequisites, etc.)
without modifying their internal implementations.

Usage:
    wrap_habitat_tools(agent, game_manager)
    # Now agent.tools["Open"] checks lock conditions before calling Habitat
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# Type for condition check functions (pre-action)
# Returns (blocked: bool, message: str or None)
ConditionCheck = Callable[[str, Any], Tuple[bool, Optional[str]]]

# Type for post-action hooks
# Takes (target, result, game_manager, agent_id) -> modified result
# Result is (action_output, observation_string)
PostActionHook = Callable[[str, Tuple[Any, str], Any, str], Tuple[Any, str]]


def _did_action_succeed(tool_name: str, tool: Any, result: Tuple[Any, str]) -> bool:
    """
    Heuristic success check for running post-action hooks safely.

    For motor skills (notably Open), process_high_level_action is called multiple
    times while the skill is still in progress. We only want to run post-hooks
    after a confirmed success signal, not on the first in-progress tick.
    """
    action_output, observation = result
    obs_text = (observation or "").lower()

    # Explicit failure signals should never trigger post-hooks.
    failure_tokens = ("failed", "cannot", "too far", "occluded", "error")
    if any(token in obs_text for token in failure_tokens):
        return False

    # Explicit success signals from skills/planners.
    if "successful execution" in obs_text or "successfully" in obs_text:
        return True

    # Instant actions: no low-level action and no failure message.
    if action_output is None:
        return bool(observation)

    # Open is a multi-step motor skill. Defer until skill reports success.
    if tool_name == "Open":
        skill = getattr(tool, "skill", None)
        if skill is None:
            return False

        if getattr(skill, "failed", False):
            return False

        # Open/Close skills expose this signal once interaction succeeded.
        if bool(getattr(skill, "was_successful", False)):
            # Oracle open toggles _is_action_issued while action is in-flight.
            issued = getattr(skill, "_is_action_issued", None)
            if issued is not None:
                try:
                    if bool(issued[0]):
                        return False
                except Exception:
                    pass
            return True

    return False


# =============================================================================
# CONDITION CHECKS
# =============================================================================

def check_not_locked(target: str, game_manager) -> Tuple[bool, Optional[str]]:
    """
    Check if target is locked.

    Returns:
        (True, message) if blocked (target is locked)
        (False, None) if allowed (not locked or no lock state)
    """
    state = game_manager.get_state()
    # If an explicit is_unlocked flag exists, honor it.
    is_unlocked = state.get_object_property(target, "is_unlocked", None)
    if is_unlocked is True:
        return False, None

    is_locked = state.get_object_property(target, "is_locked", False)

    if is_locked:
        return True, f"{target} is locked. You need a key to open it."
    return False, None


def check_has_required_key(target: str, game_manager) -> Tuple[bool, Optional[str]]:
    """
    Check if agent has the required key for a locked target.

    This is used by Use[key, target] to verify the key matches.
    """
    # This check is handled in the Use action itself
    return False, None


# =============================================================================
# POST-ACTION HOOKS
# =============================================================================

def reveal_items_inside(
    target: str,
    result: Tuple[Any, str],
    game_manager,
    agent_id: str,
) -> Tuple[Any, str]:
    """
    Post-hook for Open: add items inside the container to agent's inventory.

    Items are NOT physical objects - they go directly to inventory when
    the container is opened. The observation tells the agent what they found.

    Task config example:
        "items": [{"item_id": "item_radio_1", "inside": "cabinet_45"}]
    """
    action_output, observation = result

    state = game_manager.get_state()
    items_inside = state.get_object_property(target, "items_inside", [])

    if not items_inside:
        return result

    # Add items directly to agent's inventory
    revealed_names = []
    for item_id in items_inside:
        # Add to agent's inventory via grant_item
        state, success, msg = game_manager.grant_item(
            agent_id, item_id, source=f"Open:{target}", state=state
        )
        # Show item_id directly for clarity
        revealed_names.append(item_id)

    # Clear items_inside (they've been collected)
    state = state.set_object_property(target, "items_inside", [])
    game_manager.set_state(state)

    # Append to observation
    if len(revealed_names) == 1:
        observation = f"{observation} Inside you find: {revealed_names[0]}. It's now in your inventory."
    else:
        observation = f"{observation} Inside you find: {', '.join(revealed_names)}. They're now in your inventory."

    return action_output, observation


# =============================================================================
# REGISTRIES
# =============================================================================

# Map action names to their condition checks (pre-action)
# Conditions are checked in order; first failure blocks the action
ACTION_CONDITIONS: Dict[str, List[ConditionCheck]] = {
    "Open": [check_not_locked],
    # Add more as needed:
    # "Pick": [check_not_sealed],
    # "Navigate": [check_room_unlocked],
}

# Map action names to their post-action hooks
# Hooks run after successful action; can modify the result
POST_ACTION_HOOKS: Dict[str, List[PostActionHook]] = {
    "Open": [reveal_items_inside],
    # Add more as needed:
    # "Close": [check_trap_trigger],
}


# =============================================================================
# TOOL WRAPPER
# =============================================================================

def wrap_tool(tool, game_manager, agent_uid: int = 0) -> None:
    """
    Wrap a single Habitat tool with EMTOM condition checks and post-hooks.

    Modifies the tool in-place by replacing process_high_level_action.

    Args:
        tool: The Habitat tool to wrap
        game_manager: GameStateManager for checking conditions
        agent_uid: The agent's UID (for inventory checks, etc.)
    """
    tool_name = getattr(tool, 'name', tool.__class__.__name__)
    conditions = ACTION_CONDITIONS.get(tool_name, [])
    post_hooks = POST_ACTION_HOOKS.get(tool_name, [])

    if not conditions and not post_hooks:
        # No conditions or hooks for this tool, skip wrapping
        return

    # Store original method
    original_method = tool.process_high_level_action
    agent_id = f"agent_{agent_uid}"

    def wrapped_process(input_query: str, observations: Any) -> Tuple[Optional[Any], str]:
        """Wrapped version with pre-checks and post-hooks."""
        # PRE-ACTION: Check all conditions
        for check in conditions:
            blocked, message = check(input_query, game_manager)
            if blocked:
                return None, message

        # Execute original action
        result = original_method(input_query, observations)

        # POST-ACTION: Run hooks only after a confirmed success signal.
        if _did_action_succeed(tool_name, tool, result):
            for hook in post_hooks:
                result = hook(input_query, result, game_manager, agent_id)

        return result

    # Replace the method
    tool.process_high_level_action = wrapped_process
    tool._emtom_wrapped = True  # Mark as wrapped


def wrap_habitat_tools(
    agent,
    game_manager,
    allowed_actions: Optional[List[str]] = None,
) -> None:
    """
    Wrap all relevant Habitat tools for an agent and filter by allowed actions.

    Args:
        agent: The Agent instance with tools dict
        game_manager: GameStateManager for condition checks
        allowed_actions: Optional list of allowed action names. If provided,
                        tools not in this list will be removed.
    """
    agent_uid = getattr(agent, 'uid', 0)

    # If allowed_actions specified, remove tools not in the list
    if allowed_actions is not None:
        tools_to_remove = [
            name for name in agent.tools.keys()
            if name not in allowed_actions
        ]
        for name in tools_to_remove:
            del agent.tools[name]

    for tool_name, tool in agent.tools.items():
        # Skip already wrapped tools
        if getattr(tool, '_emtom_wrapped', False):
            continue

        # Skip EMTOM custom actions (they handle their own logic)
        if hasattr(tool, '_game_manager'):
            continue

        wrap_tool(tool, game_manager, agent_uid)


# =============================================================================
# LOCK/UNLOCK HELPERS
# =============================================================================

def lock_container(game_manager, container: str, key_type: str = "small_key") -> None:
    """
    Mark a container as locked, requiring a specific key type.

    Args:
        game_manager: GameStateManager
        container: The container to lock
        key_type: The type of key required ("small_key" or "big_key")
    """
    state = game_manager.get_state()
    state = state.set_object_property(container, "is_locked", True)
    state = state.set_object_property(container, "required_key", key_type)
    game_manager.set_state(state)


def unlock_container(game_manager, container: str) -> None:
    """
    Unlock a container.

    Args:
        game_manager: GameStateManager
        container: The container to unlock
    """
    state = game_manager.get_state()
    state = state.set_object_property(container, "is_locked", False)
    game_manager.set_state(state)


def try_unlock_with_key(
    game_manager,
    agent_id: str,
    key_id: str,
    target: str,
) -> Tuple[bool, str]:
    """
    Try to unlock a target using a key.

    Uses the item's can_unlock() method and consumable property
    instead of hardcoded key type checks.

    Args:
        game_manager: GameStateManager
        agent_id: The agent trying to unlock
        key_id: The key item ID
        target: The container to unlock

    Returns:
        (success, message) tuple
    """
    state = game_manager.get_state()

    # Check if target is locked
    is_locked = state.get_object_property(target, "is_locked", False)
    if not is_locked:
        return False, f"{target} is not locked."

    # Check if agent has the key
    if not game_manager.agent_has_item(agent_id, key_id):
        return False, f"You don't have {key_id}."

    # Get the item instance
    item = game_manager.get_item(key_id)
    if not item:
        return False, f"Unknown item: {key_id}"

    # Check if key type matches using item's can_unlock() method
    from emtom.state.item_registry import ItemRegistry
    required_key = state.get_object_property(target, "required_key", "item_small_key")
    required_key_base = ItemRegistry.get_base_id(required_key)
    key_base = ItemRegistry.get_base_id(key_id)

    # First check base type (small_key vs big_key), then ask item for specific lock
    if key_base != required_key_base:
        return False, f"This lock requires a {required_key_base}, not a {key_base}."

    if not item.can_unlock(target):
        return False, f"This key cannot unlock {target}."

    # Unlock the container
    unlock_container(game_manager, target)

    # Use the item (handles consumption via item's on_use and consumable flag)
    success, use_message = game_manager.use_item(agent_id, key_id, [target])

    key_name = item.name
    if item.consumable:
        return True, f"You unlock {target} with the {key_name}. {use_message}"
    else:
        return True, f"You unlock {target} with the {key_name}."
