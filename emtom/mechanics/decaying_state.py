"""
Decaying State Mechanic.

States automatically revert to their original value after N steps.
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to plan around timing constraints.
Real-world analogy: self-closing doors, auto-shutoff lights, timers.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from emtom.mechanics.mechanic import (
    ActionResult,
    Effect,
    MechanicCategory,
    SceneAwareMechanic,
)
from emtom.mechanics.registry import register_mechanic


# Actions that change states
STATE_ACTIONS: Dict[str, tuple] = {
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


@register_mechanic("decaying_state")
class DecayingStateMechanic(SceneAwareMechanic):
    """
    States automatically revert after a time period.

    When an agent changes the state of a bound object, the action
    succeeds normally, but the state automatically reverts after N steps.

    Examples:
    - Door opens, then closes itself after 3 steps
    - Light turns on, then turns off automatically
    - Lock stays unlocked only briefly before re-locking

    This mechanic tests whether agents can:
    1. Detect that states don't persist
    2. Plan actions within the time window
    3. Coordinate multi-step tasks under time pressure
    4. Anticipate state reversions
    """

    name = "decaying_state"
    category = MechanicCategory.TIME_DELAYED
    description = "States automatically revert after a delay"

    required_affordance = None

    def __init__(
        self,
        decay_steps: int = 3,
        allowed_states: Optional[List[str]] = None,
        max_targets: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Initialize the decaying state mechanic.

        Args:
            decay_steps: Number of steps before state reverts.
            allowed_states: List of state names that can decay.
            max_targets: Maximum number of objects to affect.
            seed: Random seed for reproducible target selection.
        """
        super().__init__()
        self.decay_steps = decay_steps
        self.allowed_states: Optional[Set[str]] = (
            set(allowed_states) if allowed_states else None
        )
        self.max_targets = max_targets
        self.seed = seed
        self._rng = random.Random(seed)

        # Track pending reversions: target_id -> (steps_remaining, state_name, revert_value)
        self._pending_reversions: Dict[str, Tuple[int, str, Any]] = {}
        # Track original states before any actions
        self._original_states: Dict[str, Dict[str, Any]] = {}

    def bind_to_scene(self, world_state: Any) -> bool:
        """Discover objects with binary states and select targets."""
        result = self.bind_to_entities_with_state(
            world_state,
            state_names=list(self.allowed_states) if self.allowed_states else None,
            max_targets=self.max_targets,
            random_select=True,
        )

        # Store original states for targets
        if result:
            for target_id in self._bound_targets:
                state_name = self._bound_states.get(target_id, "is_open")
                current_val = world_state.get_property(target_id, state_name, False)
                if target_id not in self._original_states:
                    self._original_states[target_id] = {}
                self._original_states[target_id][state_name] = current_val

        return result

    def applies_to(
        self, action_name: str, target: str, world_state: Any
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if not self._is_bound:
            return False

        if action_name not in STATE_ACTIONS:
            return False

        if target not in self._bound_targets:
            return False

        # Check if action affects the bound state
        state_name, _ = STATE_ACTIONS[action_name]
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
        """Transform the action to schedule a reversion."""
        state_name, intended_value = STATE_ACTIONS[action_name]
        current_state = world_state.get_property(target, state_name, False)

        # Calculate new state
        if intended_value is None:
            new_state = not current_state
        else:
            new_state = intended_value

        # Action succeeds normally
        effect = Effect(
            target=target,
            property_changed=state_name,
            old_value=current_state,
            new_value=new_state,
            visible_to={actor_id},
            description=f"{target} becomes {self._get_state_adjective(state_name, new_state)}",
        )

        # Schedule reversion (revert to the state before this action)
        self._pending_reversions[target] = (self.decay_steps, state_name, current_state)

        # Create pending effect for the reversion
        revert_effect = Effect(
            target=target,
            property_changed=state_name,
            old_value=new_state,
            new_value=current_state,
            visible_to=set(),
            delay_steps=self.decay_steps,
            description=f"{target} will revert to {self._get_state_adjective(state_name, current_state)}",
        )

        # Normal observation (agent doesn't know about decay initially)
        state_adj = self._get_state_adjective(state_name, new_state)
        observation = f"You {action_name.replace('_', ' ')} the {target}. It becomes {state_adj}."

        return ActionResult(
            success=True,
            effects=[effect],
            pending_effects=[revert_effect],
            observations={actor_id: observation},
            surprise_triggers={},  # No immediate surprise - surprise comes when it reverts
        )

    def tick(self, world_state: Any) -> List[Effect]:
        """
        Advance time and return any reversion effects that should trigger.

        Called by the environment each step to process decaying states.

        Returns:
            List of effects that should now be applied.
        """
        triggered = []

        for target, (steps_left, state_name, revert_value) in list(self._pending_reversions.items()):
            steps_left -= 1
            if steps_left <= 0:
                # Time to revert
                current_state = True  # Placeholder, actual comes from world_state
                effect = Effect(
                    target=target,
                    property_changed=state_name,
                    old_value=not revert_value,  # Approximate
                    new_value=revert_value,
                    visible_to=set(),  # Will be determined by environment
                    description=f"{target} reverts to {self._get_state_adjective(state_name, revert_value)}",
                )
                triggered.append(effect)
                del self._pending_reversions[target]
            else:
                self._pending_reversions[target] = (steps_left, state_name, revert_value)

        return triggered

    def get_pending_reversions(self) -> Dict[str, int]:
        """Get targets with pending reversions and their remaining steps."""
        return {k: v[0] for k, v in self._pending_reversions.items()}

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
        if action_name in STATE_ACTIONS:
            state_name, intended_value = STATE_ACTIONS[action_name]
            if intended_value is None:
                return f"{target} should toggle and stay in that state"
            adj = self._get_state_adjective(state_name, intended_value)
            return f"{target} should become {adj} and stay that way"
        return f"{action_name} on {target} should have permanent effect"

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit target mappings.

        Args:
            bindings: List of binding dicts with keys:
                - trigger_object: Object with decaying state
                - target_state: State that decays (default: "is_open")
                - decay_steps: Optional override for decay time

        Returns:
            True if bindings were applied successfully
        """
        self._explicit_bindings = bindings
        self._bound_targets = []
        self._bound_states = {}

        for binding in bindings:
            trigger = binding.get("trigger_object")
            state = binding.get("target_state", "is_open")

            # Allow per-binding decay override
            if "decay_steps" in binding:
                self.decay_steps = binding["decay_steps"]

            if trigger:
                self._bound_targets.append(trigger)
                self._bound_states[trigger] = state

        self._is_bound = len(self._bound_targets) > 0
        return self._is_bound

    def reset(self) -> None:
        """Reset per-episode state."""
        super().reset()
        self._pending_reversions.clear()
        self._original_states.clear()
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "decay_steps": self.decay_steps,
            "pending_reversions": {
                k: {"steps_remaining": v[0], "state": v[1], "revert_to": v[2]}
                for k, v in self._pending_reversions.items()
            },
            "original_states": self._original_states,
            "allowed_states": list(self.allowed_states) if self.allowed_states else None,
            "max_targets": self.max_targets,
        })
        return base_info
