"""
Inverse State Mechanic.

Actions that change any binary state have the opposite effect.
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to discover and adapt to inverted action semantics.
"""

from typing import Any, Dict, List, Optional, Set

from emtom.core.mechanic import (
    ActionResult,
    Effect,
    MechanicCategory,
    SceneAwareMechanic,
)
from emtom.mechanics.registry import register_mechanic


# Actions that affect specific states
STATE_ACTIONS: Dict[str, tuple] = {
    # action_name: (state_name, intended_value)
    "open": ("is_open", True),
    "close": ("is_open", False),
    "turn_on": ("is_on", True),
    "turn_off": ("is_on", False),
    "toggle": ("is_on", None),  # Toggle flips current value
    "activate": ("is_active", True),
    "deactivate": ("is_active", False),
    "press": ("is_pressed", True),
    "fill": ("is_filled", True),
    "empty": ("is_filled", False),
    "clean": ("is_clean", True),
    "dirty": ("is_clean", False),
    "pick_up": ("is_held", True),
    "put_down": ("is_held", False),
    "lock": ("is_locked", True),
    "unlock": ("is_locked", False),
}


@register_mechanic("inverse_state")
class InverseStateMechanic(SceneAwareMechanic):
    """
    Actions that change binary states have inverse effects.

    Unlike InverseOpenMechanic which only works on doors, this mechanic
    works on ANY object with a binary state property.

    At episode start, it discovers objects in the scene and randomly
    selects which objects (and which states) to invert.

    Examples:
    - Trying to "open" a cabinet closes it instead
    - Trying to "turn_on" a lamp turns it off instead
    - Trying to "fill" a cup empties it instead

    This mechanic tests whether agents can:
    1. Detect the unexpected behavior
    2. Update their mental model
    3. Use the inverse action to achieve their goal
    """

    name = "inverse_state"
    category = MechanicCategory.INVERSE
    description = "Actions on selected objects have inverse effects"

    # We'll discover targets at runtime instead of using a fixed affordance
    required_affordance = None

    def __init__(
        self,
        allowed_states: Optional[List[str]] = None,
        max_targets: int = 3,
    ):
        """
        Initialize the inverse state mechanic.

        Args:
            allowed_states: List of state names that can be inverted.
                If None, all binary states are candidates.
            max_targets: Maximum number of objects to affect.
        """
        super().__init__()
        self.allowed_states: Optional[Set[str]] = (
            set(allowed_states) if allowed_states else None
        )
        self.max_targets = max_targets

    def bind_to_scene(self, world_state: Any) -> bool:
        """
        Discover objects with binary states and select targets.

        Returns True if at least one target was found.
        """
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

        # Must be a state-changing action
        if action_name not in STATE_ACTIONS:
            return False

        # Must be acting on one of our bound targets
        if target not in self._bound_targets:
            return False

        # Check if the action affects the state we're inverting for this target
        state_name, _ = STATE_ACTIONS[action_name]
        bound_state = self._bound_states.get(target)

        # If we have a specific state bound for this target, only apply if action affects it
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
        """Transform the action to have inverse effect."""
        state_name, intended_value = STATE_ACTIONS[action_name]
        current_state = world_state.get_property(target, state_name, False)

        # Calculate what would normally happen
        if intended_value is None:
            # Toggle action
            normal_result = not current_state
        else:
            normal_result = intended_value

        # Invert the result
        new_state = not normal_result

        # Generate observation message
        entity = world_state.get_entity(target)
        entity_type = entity.entity_type if entity else "object"
        state_label = state_name.replace("is_", "").replace("_", " ")

        # Create descriptive observation
        if new_state == current_state:
            observation = f"You try to {action_name.replace('_', ' ')} the {target}, but nothing happens."
        else:
            expected_adj = self._get_state_adjective(state_name, normal_result)
            actual_adj = self._get_state_adjective(state_name, new_state)
            observation = (
                f"You try to {action_name.replace('_', ' ')} the {target}, "
                f"expecting it to become {expected_adj}, but it becomes {actual_adj} instead!"
            )

        effect = Effect(
            target=target,
            property_changed=state_name,
            old_value=current_state,
            new_value=new_state,
            visible_to={actor_id},
            description=observation,
        )

        # Generate surprise trigger
        surprise_triggers = {}
        if new_state != normal_result:
            expected_adj = self._get_state_adjective(state_name, normal_result)
            actual_adj = self._get_state_adjective(state_name, new_state)
            surprise_triggers[actor_id] = (
                f"Expected {target} to become {expected_adj}, "
                f"but it became {actual_adj}"
            )

        return ActionResult(
            success=True,
            effects=[effect],
            pending_effects=[],
            observations={actor_id: observation},
            surprise_triggers=surprise_triggers,
        )

    def _get_state_adjective(self, state_name: str, value: bool) -> str:
        """Convert a state name and value to a human-readable adjective."""
        adjectives = {
            "is_open": ("open", "closed"),
            "is_on": ("on", "off"),
            "is_active": ("active", "inactive"),
            "is_pressed": ("pressed", "released"),
            "is_filled": ("filled", "empty"),
            "is_clean": ("clean", "dirty"),
            "is_held": ("held", "placed"),
            "is_locked": ("locked", "unlocked"),
            "is_powered": ("powered", "unpowered"),
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
                return f"{target} should toggle its {state_name.replace('is_', '')} state"
            adj = self._get_state_adjective(state_name, intended_value)
            return f"{target} should become {adj}"
        return f"{action_name} on {target} should have normal effect"

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit target mappings.

        Args:
            bindings: List of binding dicts with keys:
                - trigger_object: Object to apply inverse effect to
                - target_state: State to invert (default: "is_open")

        Returns:
            True if bindings were applied successfully
        """
        self._explicit_bindings = bindings
        self._bound_targets = []
        self._bound_states = {}

        for binding in bindings:
            trigger = binding.get("trigger_object")
            state = binding.get("target_state", "is_open")

            if trigger:
                self._bound_targets.append(trigger)
                self._bound_states[trigger] = state

        self._is_bound = len(self._bound_targets) > 0
        return self._is_bound

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "allowed_states": list(self.allowed_states) if self.allowed_states else None,
            "max_targets": self.max_targets,
        })
        return base_info
