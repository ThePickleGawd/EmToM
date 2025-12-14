"""
Delayed Effect Mechanic.

Actions take effect after a delay of N steps.
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to understand temporal cause-effect relationships.
Real-world analogy: slow-closing doors, timer-based lights, delayed reactions.
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


# Actions that can be delayed
DELAYABLE_ACTIONS: Dict[str, tuple] = {
    "open": ("is_open", True),
    "close": ("is_open", False),
    "turn_on": ("is_on", True),
    "turn_off": ("is_on", False),
    "toggle": ("is_on", None),
    "activate": ("is_active", True),
    "deactivate": ("is_active", False),
    "lock": ("is_locked", True),
    "unlock": ("is_locked", False),
}


@register_mechanic("delayed_effect")
class DelayedEffectMechanic(SceneAwareMechanic):
    """
    Actions take effect after a delay.

    When an agent performs an action on a bound target, nothing happens
    immediately. The effect manifests after N steps.

    Examples:
    - Opening a door: door stays closed, opens 3 steps later
    - Turning on a light: light stays off, turns on after delay
    - Activating a device: nothing happens, then it activates

    This mechanic tests whether agents can:
    1. Detect that their action had no immediate effect
    2. Understand there's a delay mechanism
    3. Plan around the delay (e.g., act early)
    4. Remember pending effects
    """

    name = "delayed_effect"
    category = MechanicCategory.TIME_DELAYED
    description = "Actions take effect after a delay"

    required_affordance = None

    def __init__(
        self,
        delay_steps: int = 3,
        allowed_states: Optional[List[str]] = None,
        max_targets: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Initialize the delayed effect mechanic.

        Args:
            delay_steps: Number of steps before effect manifests.
            allowed_states: List of state names that can be delayed.
            max_targets: Maximum number of objects to affect.
            seed: Random seed for reproducible target selection.
        """
        super().__init__()
        self.delay_steps = delay_steps
        self.allowed_states: Optional[Set[str]] = (
            set(allowed_states) if allowed_states else None
        )
        self.max_targets = max_targets
        self.seed = seed
        self._rng = random.Random(seed)

        # Track pending effects: target_id -> list of (steps_remaining, effect)
        self._pending_effects: Dict[str, List[tuple]] = {}
        # Track which agents triggered which delays
        self._triggered_by: Dict[str, str] = {}  # effect_key -> agent_id

    def bind_to_scene(self, world_state: Any) -> bool:
        """Discover objects with binary states and select targets."""
        return self.bind_to_entities_with_state(
            world_state,
            state_names=list(self.allowed_states) if self.allowed_states else None,
            max_targets=self.max_targets,
            random_select=True,
        )

    def applies_to(
        self, action_name: str, target: str, world_state: Any
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if not self._is_bound:
            return False

        if action_name not in DELAYABLE_ACTIONS:
            return False

        if target not in self._bound_targets:
            return False

        # Check if action affects the bound state
        state_name, _ = DELAYABLE_ACTIONS[action_name]
        bound_state = self._bound_states.get(target)
        if bound_state is not None and state_name != bound_state:
            return False

        return True

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: Any,
    ) -> ActionResult:
        """Transform the action to have a delayed effect."""
        state_name, intended_value = DELAYABLE_ACTIONS[action_name]
        current_state = world_state.get_property(target, state_name, False)

        # Calculate what would normally happen
        if intended_value is None:
            normal_result = not current_state
        else:
            normal_result = intended_value

        # Create the delayed effect
        delayed_effect = Effect(
            target=target,
            property_changed=state_name,
            old_value=current_state,
            new_value=normal_result,
            visible_to={actor_id},
            delay_steps=self.delay_steps,
            description=f"{target} will become {self._get_state_adjective(state_name, normal_result)} in {self.delay_steps} steps",
        )

        # Store in pending effects
        if target not in self._pending_effects:
            self._pending_effects[target] = []
        self._pending_effects[target].append((self.delay_steps, delayed_effect))

        # Track who triggered it
        effect_key = f"{target}_{state_name}_{len(self._pending_effects[target])}"
        self._triggered_by[effect_key] = actor_id

        # Observation - nothing happens immediately
        expected_adj = self._get_state_adjective(state_name, normal_result)
        observation = (
            f"You {action_name.replace('_', ' ')} the {target}, "
            f"but nothing seems to happen immediately."
        )

        # Surprise trigger
        surprise_triggers = {
            actor_id: f"Expected {target} to become {expected_adj}, but nothing happened"
        }

        return ActionResult(
            success=True,
            effects=[],  # No immediate effects
            pending_effects=[delayed_effect],
            observations={actor_id: observation},
            surprise_triggers=surprise_triggers,
        )

    def tick(self, world_state: Any) -> List[Effect]:
        """
        Advance time and return any effects that should now trigger.

        Called by the environment each step to process delayed effects.

        Returns:
            List of effects that should now be applied.
        """
        triggered = []

        for target, pending_list in list(self._pending_effects.items()):
            remaining = []
            for steps_left, effect in pending_list:
                steps_left -= 1
                if steps_left <= 0:
                    triggered.append(effect)
                else:
                    remaining.append((steps_left, effect))

            if remaining:
                self._pending_effects[target] = remaining
            else:
                del self._pending_effects[target]

        return triggered

    def get_pending_count(self) -> int:
        """Get the number of pending delayed effects."""
        return sum(len(v) for v in self._pending_effects.values())

    def _get_state_adjective(self, state_name: str, value: bool) -> str:
        """Convert a state name and value to a human-readable adjective."""
        adjectives = {
            "is_open": ("open", "closed"),
            "is_on": ("on", "off"),
            "is_active": ("active", "inactive"),
            "is_locked": ("locked", "unlocked"),
        }
        if state_name in adjectives:
            true_adj, false_adj = adjectives[state_name]
            return true_adj if value else false_adj
        return f"{state_name}={value}"

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        if action_name in DELAYABLE_ACTIONS:
            state_name, intended_value = DELAYABLE_ACTIONS[action_name]
            if intended_value is None:
                return f"{target} should toggle its {state_name.replace('is_', '')} state immediately"
            adj = self._get_state_adjective(state_name, intended_value)
            return f"{target} should become {adj} immediately"
        return f"{action_name} on {target} should have immediate effect"

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit target mappings.

        Args:
            bindings: List of binding dicts with keys:
                - trigger_object: Object to delay effects on
                - target_state: State to delay (default: "is_open")
                - delay_steps: Optional override for delay (default: uses instance delay)

        Returns:
            True if bindings were applied successfully
        """
        self._explicit_bindings = bindings
        self._bound_targets = []
        self._bound_states = {}

        for binding in bindings:
            trigger = binding.get("trigger_object")
            state = binding.get("target_state", "is_open")

            # Allow per-binding delay override
            if "delay_steps" in binding:
                self.delay_steps = binding["delay_steps"]

            if trigger:
                self._bound_targets.append(trigger)
                self._bound_states[trigger] = state

        self._is_bound = len(self._bound_targets) > 0
        return self._is_bound

    def reset(self) -> None:
        """Reset per-episode state."""
        super().reset()
        self._pending_effects.clear()
        self._triggered_by.clear()
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "delay_steps": self.delay_steps,
            "pending_effects": {
                k: [(s, e.to_dict()) for s, e in v]
                for k, v in self._pending_effects.items()
            },
            "pending_count": self.get_pending_count(),
            "allowed_states": list(self.allowed_states) if self.allowed_states else None,
            "max_targets": self.max_targets,
        })
        return base_info
