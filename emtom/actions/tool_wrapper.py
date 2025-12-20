"""
Tool Wrapper for EMTOM.

Wraps Habitat tools with condition checks (locks, prerequisites, etc.)
without modifying their internal implementations.

Usage:
    wrap_habitat_tools(agent, game_manager)
    # Now agent.tools["Open"] checks lock conditions before calling Habitat
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# Type for condition check functions
# Returns (blocked: bool, message: str or None)
ConditionCheck = Callable[[str, Any], Tuple[bool, Optional[str]]]


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
# CONDITION REGISTRY
# =============================================================================

# Map action names to their condition checks
# Conditions are checked in order; first failure blocks the action
ACTION_CONDITIONS: Dict[str, List[ConditionCheck]] = {
    "Open": [check_not_locked],
    # Add more as needed:
    # "Pick": [check_not_sealed],
    # "Navigate": [check_room_unlocked],
}


# =============================================================================
# TOOL WRAPPER
# =============================================================================

def wrap_tool(tool, game_manager, agent_uid: int = 0) -> None:
    """
    Wrap a single Habitat tool with EMTOM condition checks.

    Modifies the tool in-place by replacing process_high_level_action.

    Args:
        tool: The Habitat tool to wrap
        game_manager: GameStateManager for checking conditions
        agent_uid: The agent's UID (for inventory checks, etc.)
    """
    tool_name = getattr(tool, 'name', tool.__class__.__name__)
    conditions = ACTION_CONDITIONS.get(tool_name, [])

    if not conditions:
        # No conditions for this tool, skip wrapping
        return

    # Store original method
    original_method = tool.process_high_level_action

    def wrapped_process(input_query: str, observations: Any) -> Tuple[Optional[Any], str]:
        """Wrapped version that checks conditions first."""
        # Check all conditions
        for check in conditions:
            blocked, message = check(input_query, game_manager)
            if blocked:
                return None, message

        # All conditions passed, call original
        return original_method(input_query, observations)

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

    # Check if key type matches
    required_key = state.get_object_property(target, "required_key", "small_key")
    key_def = game_manager.get_item_definition(key_id)

    # Extract base key type from instance ID (e.g., "small_key_1" -> "small_key")
    key_base = key_id.rsplit("_", 1)[0] if "_" in key_id else key_id

    if key_base != required_key:
        return False, f"This lock requires a {required_key}, not a {key_base}."

    # Unlock the container
    unlock_container(game_manager, target)

    # Consume the key (small keys are single-use)
    if key_base == "small_key":
        game_manager.remove_item(agent_id, key_id)
        key_name = key_def.name if key_def else key_id
        return True, f"You unlock {target} with the {key_name}. The key crumbles to dust."
    else:
        # Big key is not consumed
        key_name = key_def.name if key_def else key_id
        return True, f"You unlock {target} with the {key_name}."
