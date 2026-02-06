"""
Built-in items for the EMTOM inventory system.

These are the standard items available in EMTOM scenarios:
- SmallKey: Common key that unlocks small containers (consumed on use)
- BigKey: Rare key for final objectives (reusable)
- Radio: Grants Communicate action when obtained
- OracleCrystal: Passive effect grants full world observability

Item naming convention:
- All item IDs use "item_" prefix (e.g., "item_small_key_1")
- This distinguishes them from Habitat scene objects (e.g., "cup_1")
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

from emtom.state.item_registry import register_item
from emtom.state.items import BaseItem

if TYPE_CHECKING:
    from emtom.state.manager import GameStateManager


@register_item("item_small_key")
class SmallKey(BaseItem):
    """
    A small brass key that unlocks drawers and cabinets.

    Common item - can find multiples in a scenario.
    Single use - crumbles to dust after unlocking.

    Usage: UseItem[item_small_key_1, container_id]
    """

    name = "Small Key"
    description = "A small brass key. UseItem[item_small_key_N, container] to unlock."
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

    Usage: UseItem[item_big_key_1, container_id]
    """

    name = "Big Key"
    description = "A large ornate key. UseItem[item_big_key_N, container] to unlock."
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

    Grants the Communicate action when obtained.
    Not consumable - can be used unlimited times.

    Usage: Communicate["message", agent_X] (granted action, not Use)
    """

    name = "Two-Way Radio"
    description = 'A handheld radio. Grants Communicate["message", recipients] action when obtained. Useful for agents who DO NOT start with Communicate tool in their actions.'
    consumable = False
    use_args = []  # No args needed (uses granted Communicate action instead)

    # Grants Communicate action
    grants_action = "Communicate"
    action_description = 'Communicate["message", recipients]: Send a message to specific agents. Use agent IDs (agent_0, agent_1) or "all" to broadcast.'

    def on_acquire(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
    ) -> str:
        """Radio enables communication when obtained."""
        return 'Obtained Two-Way Radio! You can now use Communicate["message", agent_X].'

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Radio is used via the granted Communicate action."""
        return False, 'Use Communicate["message", agent_X] instead of Use for the radio.'


@register_item("item_stun_gun")
class StunGun(BaseItem):
    """
    A device that grants the Stun action for competitive advantage.

    Best used in competitive and mixed game modes where agents may have
    conflicting goals. Stunning an opponent can help you reach objectives
    first or prevent them from interfering with your plans.

    The stun effect:
    - Target agent's next action is skipped
    - Must be in the same room as target
    - Cannot stun yourself
    - Cannot stun an already stunned agent

    Example task setup:
        items: [
            {"item_id": "item_stun_gun_1", "base_id": "item_stun_gun", "inside": "drawer_5"}
        ]
    """

    name = "Stun Gun"
    description = (
        "A tactical device for competitive scenarios. Grants Stun[agent_id] action when held. "
        "Target agent skips their next turn, giving you a strategic advantage."
    )
    consumable = False
    use_args = []  # Uses granted action instead

    # Grants Stun action
    grants_action = "Stun"
    action_description = "Stun[agent_id]: Target agent skips their next turn. Useful for gaining advantage in competitive scenarios."

    def on_acquire(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
    ) -> str:
        """StunGun enables stun when obtained."""
        msg = "Obtained Stun Gun! You can now use Stun[agent_id] to make another agent skip their turn."
        if self.task_info:
            msg += f" {self.task_info}"
        return msg

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """StunGun is used via the granted Stun action."""
        return False, "Use Stun[agent_id] instead of UseItem for the stun gun."


@register_item("item_gold_coin")
class GoldCoin(BaseItem):
    """
    A collectible gold coin.

    Cannot be used directly - purely for collection and scoring.
    Placed inside containers, found via Open action.

    Common use case: Competitive tasks where agents/teams race to collect
    the most coins. Use has_most predicate for competitive win conditions,
    or has_at_least for cooperative collection goals.

    Example task setup:
        items: [
            {"item_id": "item_gold_coin_1", "base_id": "item_gold_coin", "inside": "cabinet_1"},
            {"item_id": "item_gold_coin_2", "base_id": "item_gold_coin", "inside": "drawer_3"},
        ]
        subtasks: [
            {"success_condition": {"entity": "team_0", "property": "has_most", "target": "item_gold_coin"}}
        ]
    """

    name = "Gold Coin"
    description = (
        "A shiny gold coin. Collect as many as you can! "
        "Commonly used in competitive tasks where the goal is to collect more coins than the opposing team."
    )
    consumable = False
    use_args = []  # No use action - just for collection

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Gold coins cannot be used directly."""
        return False, "Gold coins cannot be used directly. They're for collecting!"

    def on_acquire(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
    ) -> str:
        """Report coin acquisition with current count."""
        count = game_manager.count_items_by_type(agent_id, "item_gold_coin")
        msg = f"Found a Gold Coin! (You now have {count})"
        if self.task_info:
            msg += f" {self.task_info}"
        return msg


@register_item("item_oracle_crystal")
class OracleCrystal(BaseItem):
    """
    A glowing crystal that reveals all locations and objects.

    Passive effect: grants full world graph observability.
    When an agent has this item, they can "see" all entities in the
    environment regardless of partial observability settings.

    Not consumable - passive effect persists while in inventory.
    No direct Use action - the effect is always active.
    """

    name = "Oracle Crystal"
    description = "A glowing crystal that reveals all locations. Grants full observability while held."
    consumable = False
    use_args = []  # No direct usage

    # Passive effect: grants full world graph
    passive_effects = {"oracle_world_graph": True}

    def on_acquire(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
    ) -> str:
        """Crystal reveals all when obtained."""
        msg = "Obtained Oracle Crystal! The world's layout becomes clear in your mind."
        if self.task_info:
            msg += f" {self.task_info}"
        return msg

    def on_use(
        self,
        game_manager: "GameStateManager",
        agent_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Crystal's effect is passive, no active use needed."""
        return False, "The Oracle Crystal's power is always active while you hold it."
