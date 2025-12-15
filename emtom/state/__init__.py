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
]
