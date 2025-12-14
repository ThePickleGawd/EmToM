"""
EMTOM: Embodied Theory of Mind Benchmark

A framework for testing theory of mind reasoning through mechanics
with "unexpected behaviors" that induce surprise and require mental modeling.
"""

from emtom.mechanics.mechanic import Mechanic, MechanicCategory, Effect, ActionResult, SceneAwareMechanic
from emtom.mechanics.registry import MechanicRegistry, register_mechanic

__all__ = [
    "Mechanic",
    "MechanicCategory",
    "Effect",
    "ActionResult",
    "SceneAwareMechanic",
    "MechanicRegistry",
    "register_mechanic",
]
