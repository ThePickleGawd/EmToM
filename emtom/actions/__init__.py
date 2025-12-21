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
    DynamicItemTool,
    UseAction,
    SearchAction,
    InspectAction,
    EMTOM_ACTIONS,
    get_all_actions,
    get_emtom_tools,
)
from emtom.actions.schema import (
    tool,
    ToolSchema,
    ToolRegistry,
    get_tool_schemas,
    get_openai_tools,
    schemas_to_prompt,
    get_global_registry,
)

__all__ = [
    # Registry
    "ActionRegistry",
    "register_action",
    # Base classes
    "ActionResult",
    "EMTOMAction",
    "EMTOMActionExecutor",
    "DynamicItemTool",
    # Actions (auto-registered)
    "UseAction",
    "SearchAction",
    "InspectAction",
    # Helpers
    "EMTOM_ACTIONS",
    "get_all_actions",
    "get_emtom_tools",
    # Tool schema generation (ARE-style)
    "tool",
    "ToolSchema",
    "ToolRegistry",
    "get_tool_schemas",
    "get_openai_tools",
    "schemas_to_prompt",
    "get_global_registry",
]
