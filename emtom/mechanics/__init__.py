"""
Mechanics library for EMTOM benchmark.

This module provides the unified mechanics system used by both exploration
and benchmark phases. Mechanics define "unexpected behaviors" that test
agents' theory of mind reasoning.

Usage:
    # From config (exploration)
    mechanics = MechanicRegistry.instantiate_from_config(config)
    for m in mechanics:
        m.bind_to_scene(world_state)

    # From task bindings (benchmark)
    bindings = [b.to_dict() for b in task.mechanic_bindings]
    mechanics = MechanicRegistry.instantiate_from_bindings(bindings)

    # Apply mechanics to an action
    result = apply_mechanics(mechanics, action, agent_id, target, world_state)
"""

from emtom.mechanics.registry import MechanicRegistry, register_mechanic

# Scene-aware mechanics (work with any Habitat scene)
from emtom.mechanics.inverse_state import InverseStateMechanic
from emtom.mechanics.remote_control import RemoteControlMechanic
from emtom.mechanics.counting_state import CountingStateMechanic
from emtom.mechanics.delayed_effect import DelayedEffectMechanic
from emtom.mechanics.conditional_unlock import ConditionalUnlockMechanic
from emtom.mechanics.state_mirroring import StateMirroringMechanic
from emtom.mechanics.decaying_state import DecayingStateMechanic
from emtom.mechanics.sequence_lock import SequenceLockMechanic

# Core types
from emtom.mechanics.mechanic import (
    Mechanic,
    SceneAwareMechanic,
    ActionResult,
    Effect,
    MechanicCategory,
    create_default_effect,
)

from typing import Any, Dict, List, Optional

# Tool wrapper for Habitat integration
from emtom.mechanics.tool_wrapper import MechanicToolWrapper, wrap_tools_with_mechanics


def apply_mechanics(
    mechanics: List[Mechanic],
    action_name: str,
    agent_id: str,
    target: str,
    world_state: Any,
    default_observation: str = "",
) -> Optional[ActionResult]:
    """
    Apply mechanics to an action and return the result.

    This is the unified entry point for applying mechanics in both
    exploration and benchmark phases.

    Args:
        mechanics: List of active mechanics
        action_name: Name of the action (e.g., "open", "close")
        agent_id: ID of the agent performing the action
        target: Target object of the action
        world_state: World state object (HabitatMechanicWorldState or similar)
        default_observation: Default observation if no mechanic applies

    Returns:
        ActionResult if a mechanic applies, None otherwise
    """
    # Normalize action name to lowercase
    action_lower = action_name.lower()

    for mechanic in mechanics:
        if mechanic.applies_to(action_lower, target, world_state):
            # Create the default/intended effect
            intended = create_default_effect(action_lower, target, world_state)
            # Transform with mechanic
            result = mechanic.transform_effect(
                action_lower, agent_id, target, intended, world_state
            )
            return result

    return None


__all__ = [
    # Registry
    "MechanicRegistry",
    "register_mechanic",
    # Scene-aware mechanics
    "InverseStateMechanic",
    "RemoteControlMechanic",
    "CountingStateMechanic",
    "DelayedEffectMechanic",
    "ConditionalUnlockMechanic",
    "StateMirroringMechanic",
    "DecayingStateMechanic",
    "SequenceLockMechanic",
    # Core types
    "Mechanic",
    "SceneAwareMechanic",
    "ActionResult",
    "Effect",
    "MechanicCategory",
    "create_default_effect",
    # Helper functions
    "apply_mechanics",
    # Tool wrapper
    "MechanicToolWrapper",
    "wrap_tools_with_mechanics",
]
