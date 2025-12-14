"""
Sequence Lock Mechanic.

Objects must be interacted with in a specific order to unlock a target.
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to discover and follow procedural sequences.
Real-world analogy: combination locks, boot sequences, recipe steps, rituals.
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


@register_mechanic("sequence_lock")
class SequenceLockMechanic(SceneAwareMechanic):
    """
    A target object requires a specific sequence of interactions to unlock.

    The agent must interact with objects in the correct order. Wrong
    order resets the sequence. Correct order unlocks the final target.

    Examples:
    - Open drawer A, then cabinet B, then fridge unlocks
    - Press button 1, 2, 3 in order to open a door
    - Toggle light, open window, then TV works

    This mechanic tests whether agents can:
    1. Detect that a target is sequence-locked
    2. Discover the correct sequence through exploration
    3. Remember and reproduce the sequence
    4. Handle sequence resets gracefully
    """

    name = "sequence_lock"
    category = MechanicCategory.CONDITIONAL
    description = "Target requires specific interaction sequence to unlock"

    required_affordance = None

    def __init__(
        self,
        sequence_length: int = 3,
        seed: Optional[int] = None,
        reset_on_wrong: bool = True,
    ):
        """
        Initialize the sequence lock mechanic.

        Args:
            sequence_length: Number of steps in the required sequence.
            seed: Random seed for reproducible sequences.
            reset_on_wrong: If True, wrong interaction resets progress.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.seed = seed
        self.reset_on_wrong = reset_on_wrong
        self._rng = random.Random(seed)

        # The target object that gets unlocked
        self._locked_target: Optional[str] = None
        # Required sequence: list of (object_id, action) tuples
        self._required_sequence: List[Tuple[str, str]] = []
        # Current progress through the sequence
        self._current_step: int = 0
        # Whether the target is unlocked
        self._unlocked: bool = False
        # Track discovery attempts per agent
        self._attempts: Dict[str, List[List[Tuple[str, str]]]] = {}
        # All sequence objects (for binding)
        self._sequence_objects: Set[str] = set()

    def bind_to_scene(self, world_state: Any) -> bool:
        """Discover objects and create a random unlock sequence."""
        selector = self.get_selector()
        all_interactable = selector.select_interactable(world_state)

        if len(all_interactable) < self.sequence_length + 1:
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
            if entity.entity_type in {"button", "switch"}:
                if "is_pressed" not in states:
                    states.append("is_pressed")
            if states:
                objects_with_states.append((entity, states))

        if len(objects_with_states) < self.sequence_length + 1:
            self._is_bound = False
            return False

        # Shuffle and pick objects
        self._rng.shuffle(objects_with_states)

        # First object is the locked target
        locked_entity, locked_states = objects_with_states[0]
        self._locked_target = locked_entity.id

        # Remaining objects form the sequence
        self._required_sequence = []
        self._sequence_objects = set()

        for i in range(1, self.sequence_length + 1):
            if i >= len(objects_with_states):
                break
            seq_entity, seq_states = objects_with_states[i]
            # Pick a state and corresponding action
            state = self._rng.choice(seq_states)
            action = self._state_to_action(state)
            self._required_sequence.append((seq_entity.id, action))
            self._sequence_objects.add(seq_entity.id)

        if len(self._required_sequence) < 2:
            self._is_bound = False
            return False

        # All sequence objects plus the locked target are bound targets
        self._bound_targets = list(self._sequence_objects) + [self._locked_target]
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
        self, action_name: str, target: str, world_state: Any
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if not self._is_bound:
            return False

        # Check if acting on the locked target (before unlock)
        if target == self._locked_target and not self._unlocked:
            return True

        # Check if acting on a sequence object
        if target in self._sequence_objects:
            return True

        return False

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: Any,
    ) -> ActionResult:
        """Handle sequence interactions or block locked target."""

        # Initialize attempt tracking for this agent
        if actor_id not in self._attempts:
            self._attempts[actor_id] = [[]]

        # Check if trying to use the locked target
        if target == self._locked_target:
            if self._unlocked:
                # Already unlocked, normal behavior
                return ActionResult(
                    success=True,
                    effects=[intended_effect],
                    pending_effects=[],
                    observations={actor_id: f"You {action_name.replace('_', ' ')} the {target}."},
                    surprise_triggers={},
                )
            else:
                # Still locked
                observation = (
                    f"You try to {action_name.replace('_', ' ')} the {target}, "
                    f"but it's locked. It seems to require a specific sequence of actions."
                )
                return ActionResult(
                    success=False,
                    effects=[],
                    pending_effects=[],
                    observations={actor_id: observation},
                    surprise_triggers={
                        actor_id: f"{target} is locked by an unknown sequence"
                    },
                    error_message=f"{target} is sequence-locked",
                )

        # Check if this is a sequence object
        if target in self._sequence_objects:
            # Record the attempt
            self._attempts[actor_id][-1].append((target, action_name))

            # Check if this matches the next required step
            if self._current_step < len(self._required_sequence):
                required_obj, required_action = self._required_sequence[self._current_step]

                if target == required_obj and action_name == required_action:
                    # Correct step!
                    self._current_step += 1

                    if self._current_step >= len(self._required_sequence):
                        # Sequence complete! Unlock the target
                        self._unlocked = True
                        observation = (
                            f"You {action_name.replace('_', ' ')} the {target}. "
                            f"You hear a satisfying click - something unlocked!"
                        )
                        return ActionResult(
                            success=True,
                            effects=[intended_effect],
                            pending_effects=[],
                            observations={actor_id: observation},
                            surprise_triggers={
                                actor_id: f"Completed sequence! {self._locked_target} is now unlocked"
                            },
                        )
                    else:
                        # Partial progress
                        observation = (
                            f"You {action_name.replace('_', ' ')} the {target}. "
                            f"You hear a small click."
                        )
                        return ActionResult(
                            success=True,
                            effects=[intended_effect],
                            pending_effects=[],
                            observations={actor_id: observation},
                            surprise_triggers={},
                        )
                else:
                    # Wrong step
                    if self.reset_on_wrong and self._current_step > 0:
                        self._current_step = 0
                        self._attempts[actor_id].append([])  # Start new attempt
                        observation = (
                            f"You {action_name.replace('_', ' ')} the {target}. "
                            f"You hear a discordant sound - the sequence reset!"
                        )
                        return ActionResult(
                            success=True,
                            effects=[intended_effect],
                            pending_effects=[],
                            observations={actor_id: observation},
                            surprise_triggers={
                                actor_id: "The sequence was reset by a wrong step"
                            },
                        )
                    else:
                        # No progress, but no reset
                        observation = f"You {action_name.replace('_', ' ')} the {target}."
                        return ActionResult(
                            success=True,
                            effects=[intended_effect],
                            pending_effects=[],
                            observations={actor_id: observation},
                            surprise_triggers={},
                        )

        # Default: normal action
        return ActionResult(
            success=True,
            effects=[intended_effect],
            pending_effects=[],
            observations={actor_id: f"You {action_name.replace('_', ' ')} the {target}."},
            surprise_triggers={},
        )

    def get_progress(self) -> Tuple[int, int]:
        """Get current sequence progress as (current_step, total_steps)."""
        return (self._current_step, len(self._required_sequence))

    def is_unlocked(self) -> bool:
        """Check if the target has been unlocked."""
        return self._unlocked

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        if target == self._locked_target:
            return f"{action_name} on {target} should work immediately"
        return f"{action_name} on {target} should only affect {target}"

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit sequence.

        Args:
            bindings: List with a single binding dict containing:
                - trigger_object: The locked target object
                - sequence: List of {"object": str, "action": str} dicts

        Returns:
            True if bindings were applied successfully
        """
        if not bindings:
            self._is_bound = False
            return False

        binding = bindings[0]
        self._locked_target = binding.get("trigger_object")
        sequence_data = binding.get("sequence", [])

        self._required_sequence = []
        self._sequence_objects = set()

        for step in sequence_data:
            obj = step.get("object")
            action = step.get("action", "open")
            if obj:
                self._required_sequence.append((obj, action))
                self._sequence_objects.add(obj)

        self._explicit_bindings = bindings
        self._bound_targets = list(self._sequence_objects)
        if self._locked_target:
            self._bound_targets.append(self._locked_target)

        self._is_bound = len(self._required_sequence) >= 2 and self._locked_target is not None
        return self._is_bound

    def reset(self) -> None:
        """Reset per-episode state."""
        super().reset()
        self._current_step = 0
        self._unlocked = False
        self._attempts.clear()
        self._locked_target = None
        self._required_sequence = []
        self._sequence_objects = set()
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "locked_target": self._locked_target,
            "required_sequence": [
                {"object": obj, "action": act}
                for obj, act in self._required_sequence
            ],
            "current_step": self._current_step,
            "unlocked": self._unlocked,
            "sequence_length": self.sequence_length,
            "reset_on_wrong": self.reset_on_wrong,
            "attempts_by_agent": {
                k: [[(o, a) for o, a in attempt] for attempt in v]
                for k, v in self._attempts.items()
            },
        })
        return base_info
