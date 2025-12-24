"""
Item registry for EMTOM inventory system.

Provides a decorator-based registration system for items,
enabling plug-and-play item creation via configuration.

Item naming convention:
- All item IDs use "item_" prefix (e.g., "item_small_key")
- Instance IDs add a number suffix (e.g., "item_small_key_1")
- This distinguishes items from Habitat scene objects (e.g., "cup_1")

Usage:
    @register_item("item_small_key")
    class SmallKey(BaseItem):
        name = "Small Key"
        description = "A small brass key. Use[item_small_key_N, container] to unlock."
        item_type = ItemType.KEY
        consumable = True
        use_args = ["container"]

        def can_unlock(self, target_id: str) -> bool:
            return True
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from emtom.state.items import BaseItem

# Global registry of item classes
_ITEM_REGISTRY: Dict[str, Type["BaseItem"]] = {}


def register_item(item_id: str):
    """
    Decorator to register an item class.

    Args:
        item_id: Base ID for this item type (e.g., "small_key", "radio")

    Usage:
        @register_item("small_key")
        class SmallKey(BaseItem):
            name = "Small Key"
            ...
    """
    def decorator(cls: Type["BaseItem"]) -> Type["BaseItem"]:
        if item_id in _ITEM_REGISTRY:
            raise ValueError(
                f"Item '{item_id}' is already registered "
                f"(by {_ITEM_REGISTRY[item_id].__name__})"
            )
        # Store the base_id on the class
        cls.base_id = item_id
        _ITEM_REGISTRY[item_id] = cls
        return cls

    return decorator


class ItemRegistry:
    """
    Central registry for all available items.

    Provides methods to query, instantiate, and describe items.
    All methods are static - no instance needed.
    """

    @staticmethod
    def get(item_id: str) -> Type["BaseItem"]:
        """
        Get an item class by its base ID.

        Args:
            item_id: Base item ID (e.g., "small_key")

        Returns:
            The item class

        Raises:
            KeyError: If item not registered
        """
        if item_id not in _ITEM_REGISTRY:
            available = ", ".join(sorted(_ITEM_REGISTRY.keys()))
            raise KeyError(
                f"Unknown item: '{item_id}'. Available: {available}"
            )
        return _ITEM_REGISTRY[item_id]

    @staticmethod
    def get_by_instance_id(instance_id: str) -> Optional[Type["BaseItem"]]:
        """
        Get an item class from an instance ID like "small_key_1".

        Args:
            instance_id: Instance ID (e.g., "small_key_1", "big_key_1")

        Returns:
            The item class or None if not found
        """
        base_id = ItemRegistry.get_base_id(instance_id)
        if base_id in _ITEM_REGISTRY:
            return _ITEM_REGISTRY[base_id]
        return None

    @staticmethod
    def get_base_id(instance_id: str) -> str:
        """
        Extract base ID from an instance ID.

        Examples:
            "item_small_key_1" -> "item_small_key"
            "item_big_key_1" -> "item_big_key"
            "item_radio_1" -> "item_radio"

        Args:
            instance_id: The instance ID

        Returns:
            The base ID
        """
        # Split on last underscore and check if suffix is numeric
        if "_" in instance_id:
            parts = instance_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
        return instance_id

    @staticmethod
    def list_all() -> List[str]:
        """List all registered item base IDs."""
        return sorted(_ITEM_REGISTRY.keys())

    @staticmethod
    def is_registered(item_id: str) -> bool:
        """Check if an item is registered by base ID."""
        return item_id in _ITEM_REGISTRY

    @staticmethod
    def instantiate(item_id: str, instance_num: int = 1) -> "BaseItem":
        """
        Create an instance of an item with a unique instance ID.

        Args:
            item_id: Base item ID (e.g., "small_key")
            instance_num: Instance number for unique ID (default 1)

        Returns:
            Item instance with instance_id like "small_key_1"
        """
        cls = ItemRegistry.get(item_id)
        instance = cls()
        instance.instance_id = f"{item_id}_{instance_num}"
        return instance

    @staticmethod
    def instantiate_multiple(item_id: str, count: int) -> List["BaseItem"]:
        """
        Create multiple instances of an item.

        Args:
            item_id: Base item ID
            count: Number of instances to create

        Returns:
            List of item instances with unique IDs
        """
        return [
            ItemRegistry.instantiate(item_id, i + 1)
            for i in range(count)
        ]

    @staticmethod
    def get_info(item_id: str) -> Dict[str, Any]:
        """Get information about a registered item."""
        cls = ItemRegistry.get(item_id)
        return {
            "item_id": item_id,
            "name": getattr(cls, "name", item_id),
            "description": getattr(cls, "description", ""),
            "item_type": getattr(cls, "item_type", None),
            "consumable": getattr(cls, "consumable", False),
            "class": cls.__name__,
        }

    @staticmethod
    def get_item_descriptions() -> str:
        """
        Get item descriptions formatted for system prompts.

        Returns:
            String with each item on a line:
            - ItemName: Description
        """
        lines = []
        for item_id in sorted(_ITEM_REGISTRY.keys()):
            cls = _ITEM_REGISTRY[item_id]
            name = getattr(cls, "name", item_id)
            desc = getattr(cls, "description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def describe_all() -> str:
        """Get a human-readable description of all registered items."""
        lines = ["Registered Items:", "=" * 40]
        for item_id in sorted(_ITEM_REGISTRY.keys()):
            info = ItemRegistry.get_info(item_id)
            item_type = info["item_type"]
            type_str = item_type.value if item_type else "unknown"
            consumable = " (consumable)" if info["consumable"] else ""
            lines.append(f"  - {info['name']} [{type_str}]{consumable}: {info['description'][:50]}...")
        return "\n".join(lines)

    @staticmethod
    def get_items_for_task_generation() -> str:
        """
        Get comprehensive item information formatted for task generation prompts.

        Returns a detailed string that tells the LLM:
        - What items are available (with item_id for use in JSON)
        - How each item works (type, consumable, use_args)
        - How to use items in task definitions

        Returns:
            Formatted string for LLM prompts
        """
        from emtom.state.items import ItemType

        lines = []
        lines.append("Item IDs always use 'item_' prefix (e.g., item_small_key_1).")
        lines.append("This distinguishes them from scene objects (e.g., cup_1).")
        lines.append("")

        # Group by item type
        key_items = []
        tool_items = []

        for item_id in sorted(_ITEM_REGISTRY.keys()):
            cls = _ITEM_REGISTRY[item_id]
            use_args = getattr(cls, "use_args", []) or []
            info = {
                "item_id": item_id,
                "name": getattr(cls, "name", item_id),
                "description": getattr(cls, "description", ""),
                "item_type": getattr(cls, "item_type", ItemType.KEY),
                "consumable": getattr(cls, "consumable", False),
                "use_args": use_args,
                "grants_action": getattr(cls, "grants_action", None),
                "action_description": getattr(cls, "action_description", None),
            }

            if info["item_type"] == ItemType.TOOL:
                tool_items.append(info)
            else:
                key_items.append(info)

        # KEY items section
        if key_items:
            lines.append("KEY Items (unlock containers, satisfy possession goals):")
            for info in key_items:
                consumable_str = "consumable, single-use" if info["consumable"] else "reusable"
                lines.append(f"  - {info['item_id']}: {info['name']} ({consumable_str})")
                lines.append(f"    {info['description']}")
                # Show use_args
                if info["use_args"]:
                    usage = f"Use[{info['item_id']}_N, {', '.join(info['use_args'])}]"
                else:
                    usage = f"Use[{info['item_id']}_N]"
                lines.append(f"    Usage: {usage}")

        # TOOL items section
        if tool_items:
            if lines:
                lines.append("")
            lines.append("TOOL Items (grant new actions when obtained):")
            for info in tool_items:
                consumable_str = "consumable" if info["consumable"] else "unlimited uses"
                lines.append(f"  - {info['item_id']}: {info['name']} ({consumable_str})")
                lines.append(f"    {info['description']}")
                if info["grants_action"]:
                    lines.append(f"    Grants action: {info['grants_action']}")
                if info["action_description"]:
                    lines.append(f"    Action usage: {info['action_description']}")

        return "\n".join(lines)


def clear_registry() -> None:
    """Clear all registered items (useful for testing)."""
    _ITEM_REGISTRY.clear()


def get_registry() -> Dict[str, Type["BaseItem"]]:
    """Get the raw registry dict (useful for testing)."""
    return _ITEM_REGISTRY
