"""
Counting State Mechanic.

An object only changes state after being interacted with N times.
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to discover hidden activation conditions.
"""

import random
from typing import Any, Dict, List, Optional, Set

from emtom.mechanics.mechanic import (
    ActionResult,
    Effect,
    MechanicCategory,
    SceneAwareMechanic,
)
from emtom.mechanics.registry import register_mechanic


# Actions that can be counted
COUNTABLE_ACTIONS: Set[str] = {
    "press", "toggle", "open", "close", "turn_on", "turn_off",
    "activate", "interact", "touch", "push", "pull",
}


@register_mechanic("counting_state")
class CountingStateMechanic(SceneAwareMechanic):
    """
    Objects require multiple interactions to change state.

    Unlike CountingTriggerMechanic which only works on buttons, this
    mechanic works with ANY interactable object. At episode start,
    it discovers objects in the scene and randomly selects which
    require multiple interactions.

    Examples:
    - A cabinet might only open after being tried 3 times
    - A lamp might only turn on after being toggled 5 times
    - A drawer might only close after being pushed twice

    The agent must discover this hidden threshold through experimentation.

    This mechanic tests whether agents can:
    1. Detect that an action has no immediate effect
    2. Persist in trying the same action
    3. Discover the threshold through trial and error
    4. Remember and apply this knowledge
    """

    name = "counting_state"
    category = MechanicCategory.CONDITIONAL
    description = "Objects require multiple interactions to change state"

    # We discover targets at runtime
    required_affordance = None

    def __init__(
        self,
        required_count: int = 3,
        count_range: Optional[tuple] = None,
        max_targets: int = 2,
        show_progress: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the counting state mechanic.

        Args:
            required_count: Default number of interactions needed.
            count_range: If set (min, max), randomly pick count per target.
            max_targets: Maximum number of objects to affect.
            show_progress: If True, give hints about progress toward activation.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.required_count = required_count
        self.count_range = count_range
        self.max_targets = max_targets
        self.show_progress = show_progress
        self.seed = seed
        self._rng = random.Random(seed)

        # Per-target required counts (may vary if count_range is set)
        self._target_thresholds: Dict[str, int] = {}
        # Track interaction counts per target per action
        self._interaction_counts: Dict[str, Dict[str, int]] = {}
        # Track which targets have been "activated" (changed state)
        self._state_changed: Dict[str, Set[str]] = {}  # target -> set of states changed

    def bind_to_scene(self, world_state: Any) -> bool:
        """
        Discover interactable objects and select targets.

        Returns True if at least one target was found.
        """
        selector = self.get_selector()

        # Find interactable objects
        candidates = selector.select_interactable(world_state)

        if not candidates:
            self._is_bound = False
            return False

        # Select random targets
        self._rng.shuffle(candidates)
        selected = candidates[:min(self.max_targets, len(candidates))]

        self._bound_targets = []
        self._target_thresholds.clear()
        self._interaction_counts.clear()
        self._state_changed.clear()

        for entity in selected:
            self._bound_targets.append(entity.id)

            # Set threshold for this target
            if self.count_range:
                threshold = self._rng.randint(self.count_range[0], self.count_range[1])
            else:
                threshold = self.required_count
            self._target_thresholds[entity.id] = threshold
            self._interaction_counts[entity.id] = {}
            self._state_changed[entity.id] = set()

        self._is_bound = True
        return True

    def applies_to(
        self, action_name: str, target: str, world_state: Any
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if not self._is_bound:
            return False

        # Must be a countable action
        if action_name not in COUNTABLE_ACTIONS:
            return False

        # Must be acting on one of our targets
        return target in self._bound_targets

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: Any,
    ) -> ActionResult:
        """Transform action with counting requirement."""
        threshold = self._target_thresholds.get(target, self.required_count)

        # Get or create counts for this target
        if target not in self._interaction_counts:
            self._interaction_counts[target] = {}

        # Increment count for this action type
        counts = self._interaction_counts[target]
        count = counts.get(action_name, 0) + 1
        counts[action_name] = count

        # Get the state this action would change
        state_name = intended_effect.property_changed

        # Check if this state has already been changed
        changed_states = self._state_changed.get(target, set())
        if state_name in changed_states:
            observation = (
                f"You {action_name.replace('_', ' ')} {target}. "
                f"It's already in the desired state."
            )
            return ActionResult(
                success=True,
                effects=[],
                pending_effects=[],
                observations={actor_id: observation},
                surprise_triggers={},
            )

        # Check if threshold reached
        if count < threshold:
            # Not yet activated
            if self.show_progress:
                remaining = threshold - count
                observation = (
                    f"You {action_name.replace('_', ' ')} {target}. "
                    f"Nothing happens, but you notice slight movement. "
                    f"({count}/{threshold})"
                )
            else:
                observation = (
                    f"You {action_name.replace('_', ' ')} {target}. "
                    f"Nothing seems to happen."
                )

            surprise_triggers = {
                actor_id: f"Expected {target} to respond, but nothing happened"
            }

            return ActionResult(
                success=True,
                effects=[],
                pending_effects=[],
                observations={actor_id: observation},
                surprise_triggers=surprise_triggers,
            )

        # Threshold reached - state changes!
        if target not in self._state_changed:
            self._state_changed[target] = set()
        self._state_changed[target].add(state_name)

        # Apply the intended effect
        effect = Effect(
            target=intended_effect.target,
            property_changed=intended_effect.property_changed,
            old_value=intended_effect.old_value,
            new_value=intended_effect.new_value,
            visible_to={actor_id},
            description=f"{target} {state_name} changed after {count} attempts",
        )

        state_desc = self._describe_state_change(state_name, intended_effect.new_value)
        observation = (
            f"You {action_name.replace('_', ' ')} {target} for the {self._ordinal(count)} time. "
            f"Finally, it {state_desc}!"
        )

        # This is surprising because it finally worked
        surprise_triggers = {
            actor_id: (
                f"After {count} attempts, {target} finally responded! "
                f"It seems to require {threshold} tries."
            )
        }

        return ActionResult(
            success=True,
            effects=[effect],
            pending_effects=[],
            observations={actor_id: observation},
            surprise_triggers=surprise_triggers,
        )

    def _describe_state_change(self, state_name: str, new_value: Any) -> str:
        """Describe a state change in natural language."""
        descriptions = {
            ("is_open", True): "opens",
            ("is_open", False): "closes",
            ("is_on", True): "turns on",
            ("is_on", False): "turns off",
            ("is_active", True): "activates",
            ("is_active", False): "deactivates",
            ("is_filled", True): "fills up",
            ("is_filled", False): "empties",
            ("is_locked", True): "locks",
            ("is_locked", False): "unlocks",
        }
        key = (state_name, new_value)
        if key in descriptions:
            return descriptions[key]
        return f"changes {state_name} to {new_value}"

    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal string (1st, 2nd, 3rd, etc.)."""
        if 11 <= n % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        return f"{action_name} on {target} should work immediately"

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "target_thresholds": self._target_thresholds.copy(),
            "interaction_counts": {
                k: v.copy() for k, v in self._interaction_counts.items()
            },
            "state_changed": {k: list(v) for k, v in self._state_changed.items()},
            "required_count": self.required_count,
            "count_range": self.count_range,
            "show_progress": self.show_progress,
        })
        return base_info

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit target mappings.

        Args:
            bindings: List of binding dicts with keys:
                - trigger_object: Object that requires multiple interactions
                - count: Number of interactions needed (default: 3)

        Returns:
            True if bindings were applied successfully
        """
        self._explicit_bindings = bindings
        self._bound_targets = []
        self._target_thresholds.clear()
        self._interaction_counts.clear()
        self._state_changed.clear()

        for binding in bindings:
            trigger = binding.get("trigger_object")
            count = binding.get("count", self.required_count)

            if trigger:
                self._bound_targets.append(trigger)
                self._target_thresholds[trigger] = count
                self._interaction_counts[trigger] = {}
                self._state_changed[trigger] = set()

        self._is_bound = len(self._bound_targets) > 0
        return self._is_bound

    def reset(self) -> None:
        """Reset per-episode state."""
        super().reset()
        self._target_thresholds.clear()
        self._interaction_counts.clear()
        self._state_changed.clear()
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def get_interaction_count(self, target: str, action: str = "press") -> int:
        """Get the current interaction count for a target (for testing)."""
        counts = self._interaction_counts.get(target, {})
        return counts.get(action, 0)

    def has_state_changed(self, target: str, state_name: str) -> bool:
        """Check if a target's state has been changed (for testing)."""
        changed = self._state_changed.get(target, set())
        return state_name in changed
