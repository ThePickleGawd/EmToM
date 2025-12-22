"""
Action registry for EMTOM benchmark.

Provides a decorator-based registration system for custom actions,
enabling plug-and-play action selection via configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from emtom.actions.custom_actions import EMTOMAction

# Global registry of action classes
_ACTION_REGISTRY: Dict[str, Type["EMTOMAction"]] = {}

# Standard habitat_llm tools that EMTOM uses (not custom actions, but included for completeness)
# These are described here so prompts can reference all available actions uniformly
STANDARD_ACTIONS: Dict[str, str] = {
    "Navigate": "Navigate[target]: Move to a location (room or furniture). Almost always needed for task completion.",
    "Open": "Open[target]: Open a piece of furniture like a cabinet, drawer, or fridge. Use when task involves accessing contents.",
    "Close": "Close[target]: Close a piece of furniture. Use when task requires closing things after opening.",
    "Pick": "Pick[target]: Pick up an object. Use when task involves moving objects around.",
    "Place": "Place[target, receptacle]: Place an object on a receptacle. Use when task involves putting objects somewhere.",
    "Wait": "Wait[]: Do nothing for a turn. Use for coordination or waiting for delayed effects.",
    "Communicate": "Communicate[message]: Send a message to other agents. Essential for multi-agent coordination.",
}


def register_action(name: Optional[str] = None):
    """
    Decorator to register a custom action class.

    Usage:
        @register_action("Hide")
        class HideAction(EMTOMAction):
            ...

        # Or use the class's name attribute:
        @register_action()
        class HideAction(EMTOMAction):
            name = "Hide"
    """

    def decorator(cls: Type["EMTOMAction"]) -> Type["EMTOMAction"]:
        action_name = name or getattr(cls, "name", cls.__name__)
        if action_name in _ACTION_REGISTRY:
            raise ValueError(
                f"Action '{action_name}' is already registered "
                f"(by {_ACTION_REGISTRY[action_name].__name__})"
            )
        _ACTION_REGISTRY[action_name] = cls
        return cls

    return decorator


class ActionRegistry:
    """
    Central registry for all available custom actions.

    Provides methods to query, instantiate, and compose actions.
    """

    @staticmethod
    def get(name: str) -> Type["EMTOMAction"]:
        """Get an action class by name."""
        if name not in _ACTION_REGISTRY:
            available = ", ".join(sorted(_ACTION_REGISTRY.keys()))
            raise KeyError(
                f"Unknown action: '{name}'. Available: {available}"
            )
        return _ACTION_REGISTRY[name]

    @staticmethod
    def list_all() -> List[str]:
        """List all registered action names."""
        return sorted(_ACTION_REGISTRY.keys())

    @staticmethod
    def is_registered(name: str) -> bool:
        """Check if an action is registered."""
        return name in _ACTION_REGISTRY

    @staticmethod
    def instantiate(name: str, **params) -> "EMTOMAction":
        """Create an instance of an action."""
        cls = ActionRegistry.get(name)
        return cls(**params)

    @staticmethod
    def instantiate_all() -> Dict[str, "EMTOMAction"]:
        """Instantiate all registered actions."""
        return {name: cls() for name, cls in _ACTION_REGISTRY.items()}

    @staticmethod
    def get_info(name: str) -> Dict[str, Any]:
        """Get information about a registered action."""
        cls = ActionRegistry.get(name)
        return {
            "name": name,
            "description": getattr(cls, "description", ""),
            "class": cls.__name__,
        }

    @staticmethod
    def describe_all() -> str:
        """Get a human-readable description of all registered actions."""
        lines = ["Registered Actions:", "=" * 40]
        for name in sorted(_ACTION_REGISTRY.keys()):
            info = ActionRegistry.get_info(name)
            desc = info["description"][:60] + "..." if len(info["description"]) > 60 else info["description"]
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def get_action_descriptions(include_standard: bool = False) -> str:
        """
        Get action descriptions formatted for system prompts.

        Args:
            include_standard: If True, include standard habitat_llm tools

        Returns a string with each action on a line:
        - ActionName: Description of the action
        """
        lines = []

        # Include standard actions if requested
        if include_standard:
            for name in sorted(STANDARD_ACTIONS.keys()):
                lines.append(f"- {STANDARD_ACTIONS[name]}")

        # Include registered custom actions
        for name in sorted(_ACTION_REGISTRY.keys()):
            cls = _ACTION_REGISTRY[name]
            desc = getattr(cls, "action_description", getattr(cls, "description", ""))
            lines.append(f"- {name}: {desc}")

        return "\n".join(lines)

    @staticmethod
    def get_all_action_descriptions() -> str:
        """
        Get all action descriptions (standard + custom) for prompts.

        Returns a formatted string suitable for injection into system prompts.
        """
        return ActionRegistry.get_action_descriptions(include_standard=True)


def clear_registry() -> None:
    """Clear all registered actions (useful for testing)."""
    _ACTION_REGISTRY.clear()


def get_registry() -> Dict[str, Type["EMTOMAction"]]:
    """Get the raw registry dict (useful for testing)."""
    return _ACTION_REGISTRY
