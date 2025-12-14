"""
Conditional Unlock Mechanic.

Objects only work after a prerequisite action has been performed.
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to discover hidden prerequisites/dependencies.
Real-world analogy: need a key to unlock, must plug in before turning on.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from emtom.mechanics.mechanic import (
    ActionResult,
    Effect,
    MechanicCategory,
    SceneAwareMechanic,
)
from emtom.mechanics.object_selector import BINARY_STATES
from emtom.mechanics.registry import register_mechanic


@register_mechanic("conditional_unlock")
class ConditionalUnlockMechanic(SceneAwareMechanic):
    """
    Objects only work after a prerequisite action.

    Creates hidden dependencies between objects. Target objects won't
    respond to actions until a prerequisite object has been interacted with.

    Examples:
    - Drawer won't open until you've opened the desk
    - Lamp won't turn on until you've pressed the power strip
    - Fridge won't open until you've unlocked it (hidden lock)

    This mechanic tests whether agents can:
    1. Detect that an action failed for an unknown reason
    2. Explore to discover the prerequisite
    3. Remember and use the prerequisite knowledge
    4. Understand causal dependencies
    """

    name = "conditional_unlock"
    category = MechanicCategory.CONDITIONAL
    description = "Objects require prerequisite actions to work"

    required_affordance = None

    def __init__(
        self,
        num_dependencies: int = 2,
        seed: Optional[int] = None,
        require_state_change: bool = True,
    ):
        """
        Initialize the conditional unlock mechanic.

        Args:
            num_dependencies: Number of prerequisite->target dependencies to create.
            seed: Random seed for reproducible dependencies.
            require_state_change: If True, prerequisite must change state (not just interact).
        """
        super().__init__()
        self.num_dependencies = num_dependencies
        self.seed = seed
        self.require_state_change = require_state_change
        self._rng = random.Random(seed)

        # Mapping from locked_target_id -> (prerequisite_id, prerequisite_action)
        self._dependencies: Dict[str, Tuple[str, str]] = {}
        # Track which prerequisites have been satisfied
        self._unlocked: Set[str] = set()
        # Track which agents have discovered which dependencies
        self._discovered: Dict[str, Set[str]] = {}

    def bind_to_scene(self, world_state: Any) -> bool:
        """Discover objects and create random dependencies."""
        selector = self.get_selector()
        all_interactable = selector.select_interactable(world_state)

        if len(all_interactable) < 2:
            self._is_bound = False
            return False

        # Find objects with binary states
        objects_with_states: List[Tuple[Any, List[str]]] = []
        for entity in all_interactable:
            states = []
            for prop in entity.properties:
                if prop in BINARY_STATES:
                    states.append(prop)
            if entity.entity_type in {"door", "cabinet", "drawer", "fridge"}:
                if "is_open" not in states:
                    states.append("is_open")
            if entity.entity_type in {"light", "lamp", "tv", "fan"}:
                if "is_on" not in states:
                    states.append("is_on")
            if states:
                objects_with_states.append((entity, states))

        if len(objects_with_states) < 2:
            self._is_bound = False
            return False

        # Create random dependencies
        self._dependencies.clear()
        self._rng.shuffle(objects_with_states)

        num_pairs = min(self.num_dependencies, len(objects_with_states) // 2)
        prerequisites = objects_with_states[:num_pairs]
        targets = objects_with_states[num_pairs : num_pairs * 2]

        for (prereq, prereq_states), (target, _) in zip(prerequisites, targets):
            # Choose an action that affects the prerequisite
            prereq_state = self._rng.choice(prereq_states)
            prereq_action = self._state_to_action(prereq_state)
            self._dependencies[target.id] = (prereq.id, prereq_action)

        if not self._dependencies:
            self._is_bound = False
            return False

        self._bound_targets = list(self._dependencies.keys())
        self._is_bound = True
        return True

    def _state_to_action(self, state_name: str) -> str:
        """Map a state name to an action that would change it."""
        state_actions = {
            "is_open": "open",
            "is_on": "turn_on",
            "is_active": "activate",
            "is_pressed": "press",
            "is_filled": "fill",
            "is_locked": "unlock",
        }
        return state_actions.get(state_name, "interact")

    def applies_to(
        self, action_name: str, target: str, _world_state: Any
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if not self._is_bound:
            return False

        # Check if this is a locked target
        if target in self._dependencies and target not in self._unlocked:
            return True

        # Check if this is a prerequisite action (to track unlocking)
        for _, (prereq_id, prereq_action) in self._dependencies.items():
            if target == prereq_id and action_name == prereq_action:
                return True

        return False

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        _world_state: Any,
    ) -> ActionResult:
        """Handle the action - either block it or unlock dependencies."""

        # Check if this is a prerequisite action (unlocking something)
        for locked_id, (prereq_id, prereq_action) in self._dependencies.items():
            if target == prereq_id and action_name == prereq_action:
                # This action unlocks the dependent target
                self._unlocked.add(locked_id)

                # Normal effect happens
                observation = (
                    f"You {action_name.replace('_', ' ')} the {target}. "
                    f"You hear a click somewhere else."
                )

                return ActionResult(
                    success=True,
                    effects=[intended_effect],
                    pending_effects=[],
                    observations={actor_id: observation},
                    surprise_triggers={
                        actor_id: f"Something else was unlocked by {action_name} on {target}"
                    },
                )

        # Check if this is a locked target
        if target in self._dependencies and target not in self._unlocked:
            prereq_id, prereq_action = self._dependencies[target]

            # Record discovery attempt
            if actor_id not in self._discovered:
                self._discovered[actor_id] = set()
            self._discovered[actor_id].add(target)

            # Action is blocked
            observation = (
                f"You try to {action_name.replace('_', ' ')} the {target}, "
                f"but it doesn't respond. Something seems to be preventing it."
            )

            return ActionResult(
                success=False,
                effects=[],
                pending_effects=[],
                observations={actor_id: observation},
                surprise_triggers={
                    actor_id: f"{target} didn't respond to {action_name}"
                },
                error_message=f"{target} is locked by an unknown prerequisite",
            )

        # If unlocked, normal behavior (shouldn't reach here normally)
        return ActionResult(
            success=True,
            effects=[intended_effect],
            pending_effects=[],
            observations={actor_id: f"You {action_name.replace('_', ' ')} the {target}."},
            surprise_triggers={},
        )

    def is_unlocked(self, target: str) -> bool:
        """Check if a target has been unlocked."""
        return target in self._unlocked or target not in self._dependencies

    def get_prerequisite(self, target: str) -> Optional[Tuple[str, str]]:
        """Get the prerequisite for a target, if any."""
        return self._dependencies.get(target)

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        return f"{action_name} on {target} should work immediately"

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit dependencies.

        Args:
            bindings: List of binding dicts with keys:
                - trigger_object: The locked target object
                - prerequisite_object: The object that must be interacted with first
                - prerequisite_action: The action required on the prerequisite

        Returns:
            True if bindings were applied successfully
        """
        self._dependencies.clear()
        self._explicit_bindings = bindings

        for binding in bindings:
            locked = binding.get("trigger_object")
            prereq = binding.get("prerequisite_object")
            action = binding.get("prerequisite_action", "open")

            if locked and prereq:
                self._dependencies[locked] = (prereq, action)

        self._bound_targets = list(self._dependencies.keys())
        self._is_bound = len(self._dependencies) > 0
        return self._is_bound

    def reset(self) -> None:
        """Reset per-episode state."""
        super().reset()
        self._unlocked.clear()
        self._discovered.clear()
        self._dependencies.clear()
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "dependencies": {
                k: {"prerequisite": v[0], "action": v[1]}
                for k, v in self._dependencies.items()
            },
            "unlocked": list(self._unlocked),
            "discovered_by_agent": {k: list(v) for k, v in self._discovered.items()},
            "num_dependencies": self.num_dependencies,
        })
        return base_info
