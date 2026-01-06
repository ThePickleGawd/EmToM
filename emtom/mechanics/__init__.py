"""
EMTOM Mechanics Module.

Provides stateless mechanic handlers for transforming actions.
All state lives in EMTOMGameState, mechanics are pure functions.

Usage:
    from emtom.mechanics import apply_mechanics, list_mechanics, get_mechanic_info

    # Apply mechanics to an action
    result = apply_mechanics("open", "agent_0", "door_1", game_state)

    # List available mechanics
    mechanics = list_mechanics()

    # Get info about a mechanic
    info = get_mechanic_info("inverse_state")
"""

from emtom.mechanics.handlers import (
    # Core functions
    apply_mechanics,
    get_handler,
    list_mechanics,
    get_mechanic_info,
    get_mechanics_for_task_generation,
    # Types
    HandlerResult,
    MechanicHandler,
    # Info
    MECHANIC_INFO,
    MECHANIC_HANDLERS,
    # Individual handlers (for direct use if needed)
    handle_inverse_state,
    handle_remote_control,
    handle_conditional_unlock,
    handle_state_mirroring,
    # Theory of Mind handlers
    handle_location_change,
    handle_container_swap,
    handle_state_change_unseen,
    handle_delayed_information,
    # Communication handlers
    handle_limited_bandwidth,
    handle_delayed_messages,
    handle_noisy_channel,
    # Coordination handlers
    handle_hidden_agenda,
    handle_simultaneous_action,
)

__all__ = [
    # Core functions
    "apply_mechanics",
    "get_handler",
    "list_mechanics",
    "get_mechanic_info",
    "get_mechanics_for_task_generation",
    # Types
    "HandlerResult",
    "MechanicHandler",
    # Info
    "MECHANIC_INFO",
    "MECHANIC_HANDLERS",
    # Individual handlers
    "handle_inverse_state",
    "handle_remote_control",
    "handle_conditional_unlock",
    "handle_state_mirroring",
    # Theory of Mind handlers
    "handle_location_change",
    "handle_container_swap",
    "handle_state_change_unseen",
    "handle_delayed_information",
    # Communication handlers
    "handle_limited_bandwidth",
    "handle_delayed_messages",
    "handle_noisy_channel",
    # Coordination handlers
    "handle_hidden_agenda",
    "handle_simultaneous_action",
]
