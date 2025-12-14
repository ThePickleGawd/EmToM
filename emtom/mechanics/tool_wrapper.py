"""
Unified tool wrapper for applying mechanics to Habitat tools.

This wrapper intercepts tool calls and applies mechanics transformations,
providing consistent behavior for both exploration and benchmark phases.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from emtom.core.mechanic import Mechanic, ActionResult, create_default_effect

if TYPE_CHECKING:
    pass


def _apply_mechanics(
    mechanics: List[Mechanic],
    action_name: str,
    agent_id: str,
    target: str,
    world_state: Any,
) -> ActionResult | None:
    """
    Apply mechanics to an action and return the result.

    Args:
        mechanics: List of active mechanics
        action_name: Name of the action (e.g., "open", "close")
        agent_id: ID of the agent performing the action
        target: Target object of the action
        world_state: World state object

    Returns:
        ActionResult if a mechanic applies, None otherwise
    """
    action_lower = action_name.lower()

    for mechanic in mechanics:
        if mechanic.applies_to(action_lower, target, world_state):
            intended = create_default_effect(action_lower, target, world_state)
            result = mechanic.transform_effect(
                action_lower, agent_id, target, intended, world_state
            )
            return result

    return None


class MechanicToolWrapper:
    """
    Wraps a Habitat tool to apply EMTOM mechanics.

    This wrapper intercepts `process_high_level_action` calls and applies
    mechanic transformations. It works with both exploration and benchmark phases.

    Usage:
        mechanics = MechanicRegistry.instantiate_from_bindings(bindings)
        wrapped_open = MechanicToolWrapper(original_open, mechanics, "open", world_adapter)
        agent.tools["Open"] = wrapped_open
    """

    def __init__(
        self,
        original_tool,
        mechanics: List[Mechanic],
        action_type: str,
        world_state_adapter: Any = None,
    ):
        """
        Args:
            original_tool: The original Habitat tool instance
            mechanics: List of active mechanics
            action_type: Type of action ("open", "close", etc.)
            world_state_adapter: Adapter providing world state interface for mechanics
        """
        self.original_tool = original_tool
        self.mechanics = mechanics
        self.action_type = action_type.lower()
        self.world_state_adapter = world_state_adapter

        # Copy attributes from original tool for compatibility
        self.__dict__.update({
            k: v for k, v in original_tool.__dict__.items()
            if not k.startswith('_') and k not in (
                'original_tool', 'mechanics', 'action_type', 'world_state_adapter'
            )
        })

    def __getattr__(self, name):
        """Delegate attribute access to original tool."""
        return getattr(self.original_tool, name)

    def process_high_level_action(self, target_string: str, observations: Any):
        """
        Process a high-level action with mechanic transformations.

        This is the method called by the Habitat planner to execute motor skills.
        """
        target = target_string
        agent_id = f"agent_{getattr(self.original_tool, 'agent_uid', 0)}"

        # Apply mechanics
        result = _apply_mechanics(
            mechanics=self.mechanics,
            action_name=self.action_type,
            agent_id=agent_id,
            target=target,
            world_state=self.world_state_adapter,
        )

        if result is not None:
            # Mechanic transformed the action
            obs_text = result.observations.get(agent_id, "")

            # If mechanic blocked the action (empty effects), return None action
            if not result.effects and not result.success:
                print(f"[MechanicToolWrapper] BLOCKED: {target} - {obs_text}")
                return None, obs_text

            # If mechanic has a surprise trigger, log it
            if result.surprise_triggers.get(agent_id):
                print(f"[MechanicToolWrapper] SURPRISE: {result.surprise_triggers[agent_id]}")

            # Execute the original action but return modified observation
            low_level_action, original_msg = self.original_tool.process_high_level_action(
                target_string, observations
            )

            # Combine messages
            if obs_text:
                final_msg = f"{original_msg} {obs_text}" if original_msg else obs_text
            else:
                final_msg = original_msg

            return low_level_action, final_msg

        # No mechanic applies - execute normally
        return self.original_tool.process_high_level_action(target_string, observations)

    def __call__(self, *args, **kwargs):
        """Fallback for direct calls."""
        return self.original_tool(*args, **kwargs)


def wrap_tools_with_mechanics(
    agent_tools: Dict[str, Any],
    mechanics: List[Mechanic],
    world_state_adapter: Any = None,
    tool_action_map: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Wrap relevant agent tools with mechanic-aware versions.

    Args:
        agent_tools: Dictionary of agent tools
        mechanics: List of active mechanics
        world_state_adapter: Adapter for world state queries
        tool_action_map: Mapping of tool names to action types
            Default: {"Open": "open", "Close": "close"}

    Returns:
        Updated tools dictionary with wrapped tools
    """
    if tool_action_map is None:
        tool_action_map = {
            "Open": "open",
            "Close": "close",
            # Add more mappings as needed
        }

    wrapped_tools = dict(agent_tools)

    for tool_name, action_type in tool_action_map.items():
        if tool_name in wrapped_tools:
            wrapped_tools[tool_name] = MechanicToolWrapper(
                original_tool=agent_tools[tool_name],
                mechanics=mechanics,
                action_type=action_type,
                world_state_adapter=world_state_adapter,
            )
            print(f"[MechanicToolWrapper] Wrapped {tool_name} with mechanics")

    return wrapped_tools
