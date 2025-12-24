"""
Built-in items for the EMTOM inventory system.

These are the standard items available in EMTOM scenarios:
- SmallKey: Common key that unlocks small containers (consumed on use)
- BigKey: Rare key for final objectives (reusable)
- Radio: Enables inter-agent communication

Item naming convention:
- All item IDs use "item_" prefix (e.g., "item_small_key_1")
- This distinguishes them from Habitat scene objects (e.g., "cup_1")
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

from emtom.state.item_registry import register_item
from emtom.state.items import BaseItem, ItemType

if TYPE_CHECKING:
    from emtom.state.manager import GameStateManager


@register_item("item_small_key")
class SmallKey(BaseItem):
    """
    A small brass key that unlocks drawers and cabinets.

    Common item - can find multiples in a scenario.
    Single use - crumbles to dust after unlocking.

    Usage: Use[item_small_key_1, container_id]
    """

    name = "Small Key"
    description = "A small brass key. Use[item_small_key_N, container] to unlock."
    item_type = ItemType.KEY
    consumable = True  # Single use - consumed when used
    use_args = ["container"]  # Requires 1 arg: what to unlock

    def can_unlock(self, target_id: str) -> bool:
        """Small keys can unlock any container requiring a small_key."""
        return True

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Key crumbles to dust after unlocking."""
        return True, "The key crumbles to dust after unlocking."


@register_item("item_big_key")
class BigKey(BaseItem):
    """
    A large ornate key for important locks.

    Rare item - typically only one in a scenario.
    Reusable - not consumed when used.

    Usage: Use[item_big_key_1, container_id]
    """

    name = "Big Key"
    description = "A large ornate key. Use[item_big_key_N, container] to unlock."
    item_type = ItemType.KEY
    consumable = False  # Reusable
    use_args = ["container"]  # Requires 1 arg: what to unlock

    def can_unlock(self, target_id: str) -> bool:
        """Big keys can unlock any container requiring a big_key."""
        return True

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Big key remains intact after use."""
        return True, "The ornate key turns smoothly in the lock."


@register_item("item_radio")
class Radio(BaseItem):
    """
    A two-way radio for inter-agent communication.

    TOOL item - grants the Communicate action when obtained.
    Not consumable - can be used unlimited times.

    Usage: Communicate[message] (granted action, not Use)
    """

    name = "Two-Way Radio"
    description = "A handheld radio. Grants Communicate[message] action when obtained."
    item_type = ItemType.TOOL
    consumable = False
    use_args = []  # No args needed (uses granted Communicate action instead)

    # TOOL-specific: grants Communicate action
    grants_action = "Communicate"
    action_description = "Communicate[message]: Send a message to other agents."

    def on_acquire(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
    ) -> str:
        """Radio enables communication when obtained."""
        return "Obtained Two-Way Radio! You can now use Communicate[message]."

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Radio is used via the granted Communicate action."""
        return False, "Use Communicate[message] instead of Use for the radio."
