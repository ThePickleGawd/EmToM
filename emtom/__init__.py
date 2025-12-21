"""
EMTOM: Embodied Theory of Mind Benchmark

A framework for testing theory of mind reasoning through mechanics
with "unexpected behaviors" that induce surprise and require mental modeling.

Usage:
    from emtom import GameStateManager, EMTOMGameState

    # Initialize
    manager = GameStateManager(env_interface)
    state = manager.initialize_from_task(task_data)

    # Game loop
    state = manager.sync_from_habitat()
    state, result = manager.apply_action("open", "agent_0", "door_1")
    state, triggered = manager.tick()
    completed = manager.check_goals()
"""

from emtom.state import EMTOMGameState, GameStateManager, ActionExecutionResult
from emtom.mechanics import (
    apply_mechanics,
    list_mechanics,
    get_mechanic_info,
    HandlerResult,
    MECHANIC_INFO,
)
from emtom.evaluation import (
    TaskEvaluator,
    EvaluationResult,
    TemporalConstraint,
    Proposition,
    evaluate_task,
)

__all__ = [
    # State (primary interface)
    "EMTOMGameState",
    "GameStateManager",
    "ActionExecutionResult",
    # Mechanics
    "apply_mechanics",
    "list_mechanics",
    "get_mechanic_info",
    "HandlerResult",
    "MECHANIC_INFO",
    # Evaluation (PARTNR-style)
    "TaskEvaluator",
    "EvaluationResult",
    "TemporalConstraint",
    "Proposition",
    "evaluate_task",
]
