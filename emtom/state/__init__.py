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
    from emtom.state import register_item, BaseItem

    @register_item("item_magic_wand")
    class MagicWand(BaseItem):
        name = "Magic Wand"
        description = "A wand that casts spells."
        grants_action = "CastSpell"
        consumable = True
        uses = 3
        passive_effects = {"magic_aura": True}

        def on_use(self, game_manager, agent_id, args):
            return True, "You cast a spell!"
"""

from emtom.state.game_state import (
    EMTOMGameState,
    SpawnedItem,
    ActionRecord,
    Goal,
    GoalStatus,
)

from emtom.state.manager import (
    GameStateManager,
    ActionExecutionResult,
)

from emtom.state.items import (
    BaseItem,
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
    "ActionRecord",
    "Goal",
    "GoalStatus",
    # Manager
    "GameStateManager",
    "ActionExecutionResult",
    # Items
    "BaseItem",
    "register_item",
    "ItemRegistry",
    "clear_registry",
    "get_registry",
]
