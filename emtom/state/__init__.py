"""
EMTOM Game State Module.

Provides centralized state management for EMTOM mechanics and actions.

Usage:
    from emtom.state import EMTOMGameState, GameStateManager

    # Initialize from task
    manager = GameStateManager(env_interface)
    state = manager.initialize_from_task(task_data)

    # Game loop
    while not done:
        state = manager.sync_from_habitat()
        state, result = manager.apply_action("open", "agent_0", "door_1")
        state = manager.tick()
        completed = manager.check_goals()

Creating custom items:
    from emtom.state import register_item, BaseItem, ItemType

    @register_item("magic_wand")
    class MagicWand(BaseItem):
        name = "Magic Wand"
        description = "A wand that casts spells."
        item_type = ItemType.TOOL
        consumable = True
        uses = 3

        def on_use(self, game_manager, agent_id, target):
            return True, "You cast a spell!"
"""

from emtom.state.game_state import (
    EMTOMGameState,
    SpawnedItem,
    PendingEffect,
    ActionRecord,
    Goal,
    GoalStatus,
    AgentBelief,
)

from emtom.state.manager import (
    GameStateManager,
    ActionExecutionResult,
)

from emtom.state.items import (
    ItemType,
    BaseItem,
    ItemDefinition,  # Legacy, for backward compatibility
)

from emtom.state.item_registry import (
    register_item,
    ItemRegistry,
    clear_registry,
    get_registry,
)

# Import builtin items to trigger registration
import emtom.state.builtin_items  # noqa: F401

__all__ = [
    # Core state
    "EMTOMGameState",
    "SpawnedItem",
    "PendingEffect",
    "ActionRecord",
    "Goal",
    "GoalStatus",
    "AgentBelief",
    # Manager
    "GameStateManager",
    "ActionExecutionResult",
    # Items - new system
    "ItemType",
    "BaseItem",
    "register_item",
    "ItemRegistry",
    "clear_registry",
    "get_registry",
    # Items - legacy (deprecated)
    "ItemDefinition",
]
