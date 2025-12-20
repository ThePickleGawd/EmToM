"""
Item definitions for the EMTOM inventory system.

Items are abstract objects that exist only in game state (no Habitat correspondence).
They can be KEYs (possession-checkable, unlock things) or TOOLs (grant new actions).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ItemType(Enum):
    """Type of inventory item."""
    KEY = "key"    # Possession-checkable (unlocks things, satisfies goals)
    TOOL = "tool"  # Grants new actions/capabilities when obtained


@dataclass
class ItemDefinition:
    """
    Definition of an abstract inventory item.

    Items have no physical correspondence in Habitat - they're pure game-state
    objects managed by GameStateManager.
    """
    item_id: str                        # Unique identifier (e.g., "key_1", "lockpick_1")
    name: str                           # Display name (e.g., "Golden Key", "Lockpick")
    description: str                    # Description shown in inventory
    item_type: ItemType                 # KEY or TOOL

    # KEY items: what this unlocks/enables
    unlocks: List[str] = field(default_factory=list)  # Object IDs this can unlock

    # TOOL items: the action this grants
    grants_action: Optional[str] = None               # Action name (e.g., "Lockpick")
    action_description: Optional[str] = None          # Description for the action
    action_targets: List[str] = field(default_factory=list)  # Valid target types
    consumable: bool = False                          # Whether tool is consumed on use
    uses_remaining: Optional[int] = None              # If consumable, how many uses

    # Acquisition metadata
    hidden_in: Optional[str] = None                   # Container where this item is hidden
    granted_by_mechanic: Optional[str] = None         # Mechanic type that grants this

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
            consumable=data.get("consumable", False),
            uses_remaining=data.get("uses_remaining"),
            hidden_in=data.get("hidden_in"),
            granted_by_mechanic=data.get("granted_by_mechanic"),
        )


@dataclass
class InventoryItem:
    """
    An item in an agent's inventory (runtime state).

    Tracks when/how the item was obtained.
    """
    item_id: str
    obtained_at_step: int
    obtained_from: Optional[str] = None  # What action/object granted it
    obtained_by: Optional[str] = None    # Agent who got it (for transfers)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "item_id": self.item_id,
            "obtained_at_step": self.obtained_at_step,
            "obtained_from": self.obtained_from,
            "obtained_by": self.obtained_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InventoryItem":
        """Deserialize from dictionary."""
        return cls(
            item_id=data["item_id"],
            obtained_at_step=data.get("obtained_at_step", 0),
            obtained_from=data.get("obtained_from"),
            obtained_by=data.get("obtained_by"),
        )
