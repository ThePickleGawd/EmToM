"""
Item definitions for the EMTOM inventory system.

Items are abstract objects that exist only in game state (no Habitat correspondence).
They can be KEYs (possession-checkable, unlock things) or TOOLs (grant new actions).

To create a new item:
    from emtom.state.item_registry import register_item
    from emtom.state.items import BaseItem, ItemType

    @register_item("magic_wand")
    class MagicWand(BaseItem):
        name = "Magic Wand"
        description = "A wand that casts spells."
        item_type = ItemType.TOOL
        grants_action = "CastSpell"
        consumable = True
        uses = 3

        def on_use(self, game_manager, agent_id, target):
            return True, "You cast a spell!"
"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from emtom.state.manager import GameStateManager


class ItemType(Enum):
    """Type of inventory item."""
    KEY = "key"    # Possession-checkable (unlocks things, satisfies goals)
    TOOL = "tool"  # Grants new actions/capabilities when obtained


class BaseItem(ABC):
    """
    Base class for all EMTOM items.

    Subclass this and use @register_item decorator to create new items.

    Class Attributes (define in subclass):
        name: Display name (e.g., "Small Key")
        description: Description shown in inventory
        item_type: ItemType.KEY or ItemType.TOOL
        consumable: Whether item is consumed on use (default False)
        uses: Number of uses if consumable (default None = unlimited or single-use)

    For TOOL items:
        grants_action: Action name this item grants (e.g., "Communicate")
        action_description: Description of the granted action

    Instance Attributes (set at runtime):
        instance_id: Unique ID like "small_key_1"
        base_id: Base ID like "small_key" (set by @register_item)
    """

    # Class attributes - override in subclasses
    name: str = "Unknown Item"
    description: str = ""
    item_type: ItemType = ItemType.KEY
    consumable: bool = False
    uses: Optional[int] = None

    # TOOL-specific attributes
    grants_action: Optional[str] = None
    action_description: Optional[str] = None
    action_targets: Optional[List[str]] = None
    # Room restrictions: if set, tool only works in these rooms (great for ToM tasks)
    allowed_rooms: Optional[List[str]] = None

    # Instance attributes - set at runtime
    instance_id: str = ""
    base_id: str = ""  # Set by @register_item decorator

    # Runtime state
    uses_remaining: Optional[int] = None
    hidden_in: Optional[str] = None

    def __init__(self):
        """Initialize item instance."""
        # Copy class-level uses to instance
        if self.uses is not None:
            self.uses_remaining = self.uses

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        target: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Called when this item is used.

        Override in subclasses for custom behavior.

        Args:
            game_manager: The game state manager
            agent_id: ID of agent using the item
            target: Optional target of the use action

        Returns:
            (success, message) tuple
        """
        return True, f"You use the {self.name}."

    def on_acquire(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
    ) -> str:
        """
        Called when an agent acquires this item.

        Override in subclasses for custom behavior.

        Args:
            game_manager: The game state manager
            agent_id: ID of agent acquiring the item

        Returns:
            Message to show agent
        """
        return f"Obtained {self.name}!"

    def can_unlock(self, target_id: str) -> bool:
        """
        For KEY items: check if this key can unlock the target.

        Override in subclasses for custom unlock logic.

        Args:
            target_id: ID of the locked object

        Returns:
            True if this key can unlock the target
        """
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize item to dictionary for game state storage."""
        return {
            "instance_id": self.instance_id,
            "base_id": self.base_id,
            "name": self.name,
            "description": self.description,
            "item_type": self.item_type.value,
            "consumable": self.consumable,
            "uses": self.uses,
            "uses_remaining": self.uses_remaining,
            "grants_action": self.grants_action,
            "action_description": self.action_description,
            "action_targets": self.action_targets,
            "allowed_rooms": self.allowed_rooms,
            "hidden_in": self.hidden_in,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseItem":
        """
        Recreate item instance from dictionary.

        This creates an instance of the registered item class.
        """
        from emtom.state.item_registry import ItemRegistry

        base_id = data.get("base_id", "")
        instance_id = data.get("instance_id", "")

        # Get the registered class for this item type
        try:
            item_cls = ItemRegistry.get(base_id)
            instance = item_cls()
        except KeyError:
            # Fallback for unknown items
            instance = cls()
            instance.name = data.get("name", "Unknown")
            instance.description = data.get("description", "")
            item_type_str = data.get("item_type", "key")
            instance.item_type = ItemType(item_type_str)

        # Set instance attributes
        instance.instance_id = instance_id
        instance.base_id = base_id
        instance.uses_remaining = data.get("uses_remaining")
        instance.hidden_in = data.get("hidden_in")
        # Override class-level TOOL attributes if specified in data
        if data.get("action_targets"):
            instance.action_targets = data.get("action_targets")
        if data.get("allowed_rooms") is not None:
            instance.allowed_rooms = data.get("allowed_rooms")

        return instance


# Legacy compatibility - keep ItemDefinition for backward compatibility
# This can be removed once all code is migrated to BaseItem
class ItemDefinition:
    """
    Legacy item definition class.

    DEPRECATED: Use BaseItem with @register_item decorator instead.
    Kept for backward compatibility during migration.
    """

    def __init__(
        self,
        item_id: str,
        name: str,
        description: str,
        item_type: ItemType,
        unlocks: Optional[List[str]] = None,
        grants_action: Optional[str] = None,
        action_description: Optional[str] = None,
        action_targets: Optional[List[str]] = None,
        allowed_rooms: Optional[List[str]] = None,
        consumable: bool = False,
        uses_remaining: Optional[int] = None,
        hidden_in: Optional[str] = None,
        granted_by_mechanic: Optional[str] = None,
    ):
        self.item_id = item_id
        self.name = name
        self.description = description
        self.item_type = item_type
        self.unlocks = unlocks or []
        self.grants_action = grants_action
        self.action_description = action_description
        self.action_targets = action_targets or []
        self.allowed_rooms = allowed_rooms  # Room restrictions for TOOL items
        self.consumable = consumable
        self.uses_remaining = uses_remaining
        self.hidden_in = hidden_in
        self.granted_by_mechanic = granted_by_mechanic

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "description": self.description,
            "item_type": self.item_type.value,
            "unlocks": self.unlocks,
            "grants_action": self.grants_action,
            "action_description": self.action_description,
            "action_targets": self.action_targets,
            "allowed_rooms": self.allowed_rooms,
            "consumable": self.consumable,
            "uses_remaining": self.uses_remaining,
            "hidden_in": self.hidden_in,
            "granted_by_mechanic": self.granted_by_mechanic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ItemDefinition":
        """Deserialize from dictionary."""
        item_type_str = data.get("item_type", "key")
        item_type = ItemType(item_type_str) if isinstance(item_type_str, str) else item_type_str

        return cls(
            item_id=data["item_id"],
            name=data["name"],
            description=data.get("description", ""),
            item_type=item_type,
            unlocks=data.get("unlocks", []),
            grants_action=data.get("grants_action"),
            action_description=data.get("action_description"),
            action_targets=data.get("action_targets", []),
            allowed_rooms=data.get("allowed_rooms"),
            consumable=data.get("consumable", False),
            uses_remaining=data.get("uses_remaining"),
            hidden_in=data.get("hidden_in"),
            granted_by_mechanic=data.get("granted_by_mechanic"),
        )
