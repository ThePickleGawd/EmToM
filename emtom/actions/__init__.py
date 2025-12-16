"""
EMTOM Custom Actions.

These actions extend the standard partnr tools with mechanics-aware behaviors.
They can be affected by EMTOM mechanics (inverse_state, remote_control, etc.)

To add a new action:
1. Create a class that extends EMTOMAction in custom_actions.py
2. Decorate it with @register_action("ActionName")
3. The action will automatically be available everywhere
"""

from emtom.actions.registry import ActionRegistry, register_action
from emtom.actions.custom_actions import (
    ActionResult,
    EMTOMAction,
    EMTOMActionExecutor,
    UseAction,
    InspectAction,
    EMTOM_ACTIONS,
    get_all_actions,
    get_emtom_tools,
)

__all__ = [
    # Registry
    "ActionRegistry",
    "register_action",
    # Base classes
    "ActionResult",
    "EMTOMAction",
    "EMTOMActionExecutor",
    # Actions (auto-registered)
    "UseAction",
    "InspectAction",
    # Helpers
    "EMTOM_ACTIONS",
    "get_all_actions",
    "get_emtom_tools",
]
