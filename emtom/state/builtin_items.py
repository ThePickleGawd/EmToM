"""
Built-in items for the EMTOM inventory system.

These are the standard items available in EMTOM scenarios:
- SmallKey: Common key that unlocks small containers (consumed on use)
- BigKey: Rare key for final objectives (reusable)
- Radio: Enables inter-agent communication
"""

from typing import Optional, Tuple, TYPE_CHECKING

from emtom.state.item_registry import register_item
from emtom.state.items import BaseItem, ItemType

if TYPE_CHECKING:
    from emtom.state.manager import GameStateManager


@register_item("small_key")
class SmallKey(BaseItem):
    """
    A small brass key that unlocks drawers and cabinets.

    Common item - can find multiples in a scenario.
    Single use - crumbles to dust after unlocking.
    """

    name = "Small Key"
    description = "A small brass key that unlocks drawers and cabinets."
    item_type = ItemType.KEY
    consumable = True  # Single use - consumed when used

    def can_unlock(self, target_id: str) -> bool:
        """Small keys can unlock any container requiring a small_key."""
        # Small keys are generic - they can unlock anything that needs them
        return True

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        target: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Key crumbles to dust after unlocking."""
        return True, "The key crumbles to dust after unlocking."


@register_item("big_key")
class BigKey(BaseItem):
    """
    A large ornate key for important locks.

    Rare item - typically only one in a scenario.
    Reusable - not consumed when used.
    """

    name = "Big Key"
    description = "A large ornate key. This looks important."
    item_type = ItemType.KEY
    consumable = False  # Reusable

    def can_unlock(self, target_id: str) -> bool:
        """Big keys can unlock any container requiring a big_key."""
        return True

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        target: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Big key remains intact after use."""
        return True, "The ornate key turns smoothly in the lock."


@register_item("radio")
class Radio(BaseItem):
    """
    A two-way radio for inter-agent communication.

    TOOL item - grants the Communicate action when obtained.
    Not consumable - can be used unlimited times.
    """

    name = "Two-Way Radio"
    description = "A handheld radio for communication with others."
    item_type = ItemType.TOOL
    consumable = False

    # TOOL-specific: grants Communicate action
    grants_action = "Communicate"
    action_description = "Communicate[message]: Send a message to other agents."

    def on_acquire(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
    ) -> str:
        """Radio enables communication when obtained."""
        return "Obtained Two-Way Radio! You can now communicate with other agents."

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        target: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Send a radio message."""
        if not target:
            return False, "You need to specify a message to send."
        return True, f'You speak into the radio: "{target}"'
