"""
Item definitions for the EMTOM inventory system.

Items are abstract objects that exist only in game state (no Habitat correspondence).
All items share the same base properties and can optionally grant actions or have
passive effects.

Item naming convention:
- All item IDs use "item_" prefix (e.g., "item_small_key_1")
- This distinguishes them from Habitat scene objects (e.g., "cup_1")

To create a new item:
    from emtom.state.item_registry import register_item
    from emtom.state.items import BaseItem

    @register_item("item_magic_wand")
    class MagicWand(BaseItem):
        name = "Magic Wand"
        description = "A wand that casts spells."
        grants_action = "CastSpell"
        consumable = True
        uses = 3
        use_args = ["target"]  # Requires 1 arg
        passive_effects = {"magic_aura": True}

        def on_use(self, game_manager, agent_id, args):
            target = args[0] if args else None
            return True, f"You cast a spell on {target}!"
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from emtom.state.manager import GameStateManager


class BaseItem(ABC):
    """
    Base class for all EMTOM items.

    Subclass this and use @register_item decorator to create new items.

    Class Attributes (define in subclass):
        name: Display name (e.g., "Small Key")
        description: Description shown in inventory (include usage hint)
        consumable: Whether item is consumed on use (default False)
        uses: Number of uses if consumable (default None = unlimited or single-use)
        use_args: List of argument names for Use action (e.g., ["container"])
        grants_action: Action name this item grants when obtained (e.g., "Communicate")
        action_description: Description of the granted action
        passive_effects: Dict of passive effects this item provides (e.g., {"oracle_world_graph": True})

    Instance Attributes (set at runtime):
        instance_id: Unique ID like "item_small_key_1"
        base_id: Base ID like "item_small_key" (set by @register_item)
        task_info: Task-specific info set by task generator, shown to agent
    """

    # Class attributes - override in subclasses
    name: str = "Unknown Item"
    description: str = ""
    consumable: bool = False
    uses: Optional[int] = None
    use_args: List[str] = []  # Arguments for Use action (e.g., ["container"])

    # Granted action (optional)
    grants_action: Optional[str] = None
    action_description: Optional[str] = None
    action_targets: Optional[List[str]] = None
    # Room restrictions: if set, granted action only works in these rooms
    allowed_rooms: Optional[List[str]] = None

    # Passive effects (checked at action time)
    passive_effects: Dict[str, Any] = {}

    # Instance attributes - set at runtime
    instance_id: str = ""
    base_id: str = ""  # Set by @register_item decorator

    # Runtime state
    uses_remaining: Optional[int] = None
    hidden_in: Optional[str] = None

    # Task-specific info / clue (set by task generator, shown when acquired)
    # Use this for hints that guide emergent discovery
    task_info: Optional[str] = None

    def __init__(self):
        """Initialize item instance."""
        # Copy class-level uses to instance
        if self.uses is not None:
            self.uses_remaining = self.uses

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Called when this item is used.

        Override in subclasses for custom behavior.

        Args:
            game_manager: The game state manager
            agent_id: ID of agent using the item
            args: List of arguments passed to Use (validated against use_args)

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
            Message to show agent (includes task_info/clue if present)
        """
        msg = f"Obtained {self.name}!"
        if self.task_info:
            msg += f" {self.task_info}"
        return msg

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
            "consumable": self.consumable,
            "uses": self.uses,
            "uses_remaining": self.uses_remaining,
            "use_args": self.use_args,
            "grants_action": self.grants_action,
            "action_description": self.action_description,
            "action_targets": self.action_targets,
            "allowed_rooms": self.allowed_rooms,
            "passive_effects": self.passive_effects,
            "hidden_in": self.hidden_in,
            "task_info": self.task_info,
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

        # Set instance attributes
        instance.instance_id = instance_id
        instance.base_id = base_id
        instance.uses_remaining = data.get("uses_remaining")
        instance.hidden_in = data.get("hidden_in")
        instance.task_info = data.get("task_info")

        # Override class-level attributes if specified in data
        if data.get("action_targets"):
            instance.action_targets = data.get("action_targets")
        if data.get("allowed_rooms") is not None:
            instance.allowed_rooms = data.get("allowed_rooms")
        if data.get("passive_effects"):
            instance.passive_effects = data.get("passive_effects")

        return instance
