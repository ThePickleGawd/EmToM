"""
Core mechanic abstractions for EMTOM benchmark.

Mechanics define how actions produce effects that may differ from agent expectations,
enabling theory of mind reasoning through "unexpected behaviors".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class MechanicCategory(Enum):
    """Categories of mechanics based on their unexpected behavior type."""

    INVERSE = "inverse"  # Actions have opposite effects
    HIDDEN_MAPPING = "hidden_mapping"  # Actions affect unexpected targets
    CONDITIONAL = "conditional"  # Effects depend on hidden state
    TIME_DELAYED = "time_delayed"  # Effects happen after a delay
    PER_AGENT = "per_agent"  # Different agents observe different effects
    COMPOUND = "compound"  # Multiple mechanics combined


@dataclass
class Effect:
    """
    Represents the result of an action on the world state.

    Effects can be immediate or delayed, and may be visible to different agents.
    """

    target: str  # Entity ID that was affected
    property_changed: str  # Which property changed
    old_value: Any  # Previous value
    new_value: Any  # New value
    visible_to: Set[str] = field(default_factory=set)  # Which agents can see this
    delay_steps: int = 0  # Steps until effect manifests (0 = immediate)
    description: str = ""  # Human-readable description of the effect

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "target": self.target,
            "property_changed": self.property_changed,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "visible_to": list(self.visible_to),
            "delay_steps": self.delay_steps,
            "description": self.description,
        }


@dataclass
class ActionResult:
    """
    Result of applying an action through the mechanics system.

    Contains immediate and delayed effects, per-agent observations,
    and triggers for surprise detection.
    """

    success: bool  # Whether the action succeeded
    effects: List[Effect] = field(default_factory=list)  # Immediate effects
    pending_effects: List[Effect] = field(default_factory=list)  # Delayed effects
    observations: Dict[str, str] = field(
        default_factory=dict
    )  # Per-agent observation text
    surprise_triggers: Dict[str, str] = field(
        default_factory=dict
    )  # Per-agent surprise explanations
    error_message: Optional[str] = None  # Error message if action failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "effects": [e.to_dict() for e in self.effects],
            "pending_effects": [e.to_dict() for e in self.pending_effects],
            "observations": self.observations,
            "surprise_triggers": self.surprise_triggers,
            "error_message": self.error_message,
        }


class Mechanic(ABC):
    """
    Base class for EMTOM mechanics.

    Each mechanic defines how actions produce effects that may differ from
    agent expectations. Subclasses implement specific "unexpected behaviors"
    that induce theory of mind reasoning.

    Mechanics are applied in order; the first matching mechanic handles the action.
    """

    name: str = "base_mechanic"
    category: MechanicCategory = MechanicCategory.INVERSE
    description: str = "Base mechanic - override in subclass"

    @abstractmethod
    def applies_to(
        self, action_name: str, target: str, world_state: "TextWorldState"
    ) -> bool:
        """
        Check if this mechanic should intercept this action.

        Args:
            action_name: Name of the action being performed (e.g., "open", "press")
            target: ID of the target entity
            world_state: Current world state

        Returns:
            True if this mechanic should handle the action
        """
        pass

    @abstractmethod
    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: "TextWorldState",
    ) -> ActionResult:
        """
        Transform the intended effect into the actual effect.

        This is where the "unexpected behavior" happens. The mechanic
        can modify, replace, or add to the intended effect.

        Args:
            action_name: Name of the action being performed
            actor_id: ID of the agent performing the action
            target: ID of the target entity
            intended_effect: What would normally happen
            world_state: Current world state

        Returns:
            ActionResult with actual effects and observations
        """
        pass

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """
        Return a description of what an agent would normally expect from this action.

        Used for surprise detection - comparing expected vs actual outcomes.
        """
        return f"Performing {action_name} on {target}"

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """
        Return any hidden state for debugging/logging.

        This is NOT shown to agents but can be logged for analysis.
        """
        return {}

    def reset(self) -> None:
        """
        Reset any internal state of the mechanic.

        Called at the start of each episode.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category.value})"


class SceneAwareMechanic(Mechanic):
    """
    A mechanic that discovers applicable objects at episode start.

    Instead of hardcoding specific entity types like "door" or "button",
    SceneAwareMechanics use affordance-based selection to work with
    whatever objects exist in the current scene.

    Subclasses should:
    1. Set `required_affordance` or override `bind_to_scene()`
    2. Store discovered targets in `bound_targets`
    3. Use `is_bound` to check if the mechanic is active
    """

    name: str = "scene_aware_mechanic"
    description: str = "Scene-aware mechanic - discovers targets at runtime"

    # The affordance required for this mechanic (e.g., "openable", "toggleable")
    # If None, subclass must override bind_to_scene()
    required_affordance: Optional[str] = None

    def __init__(self) -> None:
        """Initialize scene-aware mechanic."""
        self._is_bound: bool = False
        self._bound_targets: List[str] = []  # Internal entity IDs
        self._bound_names: List[str] = []  # Human-readable names (e.g., fridge_58)
        self._id_to_name: Dict[str, str] = {}  # Mapping from ID to name
        self._name_to_id: Dict[str, str] = {}  # Mapping from name to ID
        self._bound_states: Dict[str, str] = {}  # entity_id -> state_name
        self._selector: Optional["ObjectSelector"] = None

    @property
    def is_bound(self) -> bool:
        """Check if the mechanic has been bound to the current scene."""
        return self._is_bound

    @property
    def bound_targets(self) -> List[str]:
        """Get the entity IDs this mechanic is bound to."""
        return self._bound_targets

    def get_selector(self) -> "ObjectSelector":
        """Get or create the object selector."""
        if self._selector is None:
            from emtom.core.object_selector import ObjectSelector
            self._selector = ObjectSelector()
        return self._selector

    def bind_to_scene(self, world_state: "TextWorldState") -> bool:
        """
        Bind the mechanic to objects in the current scene.

        Called at episode start. The mechanic should discover which objects
        it can affect and store them for later use.

        Args:
            world_state: The current world state with all scene objects

        Returns:
            True if the mechanic found applicable objects and is now active,
            False if no suitable objects exist (mechanic will be inactive)
        """
        if self.required_affordance is None:
            # Subclass must override
            return False

        selector = self.get_selector()
        candidates = selector.select_by_affordance(world_state, self.required_affordance)

        if not candidates:
            self._is_bound = False
            self._bound_targets = []
            return False

        # Default: select all candidates
        self._bound_targets = [e.id for e in candidates]
        self._is_bound = True
        return True

    def bind_to_entities_with_state(
        self,
        world_state: "TextWorldState",
        state_names: Optional[List[str]] = None,
        max_targets: int = 1,
        random_select: bool = True,
    ) -> bool:
        """
        Bind to entities that have specific binary states.

        Helper method for mechanics that operate on any binary state.

        Args:
            world_state: The current world state
            state_names: List of acceptable state names, or None for any binary state
            max_targets: Maximum number of targets to bind
            random_select: If True, randomly select from candidates

        Returns:
            True if successfully bound to at least one target
        """
        import random

        selector = self.get_selector()
        candidates_with_states = selector.select_with_binary_state(world_state)

        if not candidates_with_states:
            self._is_bound = False
            return False

        # Filter by allowed state names if specified
        valid_candidates: List[Tuple["Entity", List[str]]] = []
        for entity, entity_states in candidates_with_states:
            if state_names is None:
                valid_states = entity_states
            else:
                valid_states = [s for s in entity_states if s in state_names]

            if valid_states:
                valid_candidates.append((entity, valid_states))

        if not valid_candidates:
            self._is_bound = False
            return False

        # Select targets
        if random_select and len(valid_candidates) > max_targets:
            selected = random.sample(valid_candidates, max_targets)
        else:
            selected = valid_candidates[:max_targets]

        self._bound_targets = []
        self._bound_names = []
        self._id_to_name = {}
        self._name_to_id = {}
        self._bound_states = {}

        for entity, states in selected:
            entity_id = entity.id
            # Get human-readable name - try 'name' property first, fall back to id
            entity_name = getattr(entity, 'name', None) or entity_id

            self._bound_targets.append(entity_id)
            self._bound_names.append(entity_name)
            self._id_to_name[entity_id] = entity_name
            self._name_to_id[entity_name] = entity_id
            # Pick a random state from available states for this entity
            self._bound_states[entity_id] = random.choice(states) if random_select else states[0]

        self._is_bound = True
        return True

    def applies_to(
        self, action_name: str, target: str, world_state: "TextWorldState"
    ) -> bool:
        """
        Check if this mechanic applies to the given action.

        By default, applies to any action on bound targets.
        Checks both internal IDs and human-readable names.
        """
        if not self._is_bound:
            return False
        # Check if target matches either internal ID or human-readable name
        return target in self._bound_targets or target in self._bound_names

    def reset(self) -> None:
        """Reset mechanic state including bindings."""
        super().reset()
        self._is_bound = False
        self._bound_targets = []
        self._bound_names = []
        self._id_to_name = {}
        self._name_to_id = {}
        self._bound_states = {}

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return hidden state for debugging."""
        return {
            "is_bound": self._is_bound,
            "bound_targets": self._bound_targets,
            "bound_names": self._bound_names,
            "bound_states": self._bound_states,
            "required_affordance": self.required_affordance,
        }


def create_default_effect(
    action_name: str,
    target: str,
    world_state: "TextWorldState",
) -> Effect:
    """
    Create the default/expected effect for a standard action.

    This is what would happen in a "normal" world without unexpected mechanics.
    """
    # Map action names to expected property changes
    action_effects = {
        "open": ("is_open", True),
        "close": ("is_open", False),
        "toggle": ("is_on", None),  # Toggle flips current value
        "press": ("is_pressed", True),
        "activate": ("is_active", True),
        "deactivate": ("is_active", False),
        "pick_up": ("is_held", True),
        "put_down": ("is_held", False),
        "turn_on": ("is_on", True),
        "turn_off": ("is_on", False),
    }

    if action_name in action_effects:
        prop, new_val = action_effects[action_name]
        old_val = world_state.get_property(target, prop, False)
        if new_val is None:  # Toggle
            new_val = not old_val
        return Effect(
            target=target,
            property_changed=prop,
            old_value=old_val,
            new_value=new_val,
            description=f"{action_name.replace('_', ' ')} {target}",
        )

    # Generic effect for unknown actions
    return Effect(
        target=target,
        property_changed="last_action",
        old_value=None,
        new_value=action_name,
        description=f"Performed {action_name} on {target}",
    )
