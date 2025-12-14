"""
Remote Control Mechanic.

Interacting with one object affects a different object (hidden mapping).
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to discover hidden cause-effect relationships.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from emtom.core.mechanic import (
    ActionResult,
    Effect,
    MechanicCategory,
    SceneAwareMechanic,
)
from emtom.core.object_selector import BINARY_STATES
from emtom.mechanics.registry import register_mechanic


@register_mechanic("remote_control")
class RemoteControlMechanic(SceneAwareMechanic):
    """
    Interacting with one object affects a different object.

    Unlike RemoteSwitchMechanic which requires "switch" entities, this
    mechanic works with ANY objects that have state. At episode start,
    it discovers objects in the scene and creates random mappings.

    Examples:
    - Opening a cabinet might turn on a lamp
    - Toggling a faucet might open a drawer
    - Pressing a button might fill a bowl

    The agent must discover these hidden mappings through exploration.

    This mechanic tests whether agents can:
    1. Discover non-local cause-effect relationships
    2. Build a mental model of the hidden mapping
    3. Use this knowledge to achieve goals
    4. Communicate mappings to other agents (theory of mind)
    """

    name = "remote_control"
    category = MechanicCategory.HIDDEN_MAPPING
    description = "Interacting with objects affects different objects"

    # We discover targets at runtime
    required_affordance = None

    def __init__(
        self,
        num_mappings: int = 2,
        seed: Optional[int] = None,
        same_room_only: bool = False,
    ):
        """
        Initialize the remote control mechanic.

        Args:
            num_mappings: Number of control->target mappings to create.
            seed: Random seed for reproducible mappings.
            same_room_only: If True, only create mappings within same room.
        """
        super().__init__()
        self.num_mappings = num_mappings
        self.seed = seed
        self.same_room_only = same_room_only
        self._rng = random.Random(seed)

        # Mapping from controller_id -> (target_id, target_state)
        self._mappings: Dict[str, Tuple[str, str]] = {}
        # Actions that trigger the remote effect
        self._trigger_actions: Set[str] = {
            "open", "close", "toggle", "turn_on", "turn_off",
            "press", "activate", "pick_up", "put_down",
        }
        # Track which agents have discovered which mappings
        self._discovered: Dict[str, Set[str]] = {}

    def bind_to_scene(self, world_state: Any) -> bool:
        """
        Discover objects and create random control mappings.

        Returns True if at least one mapping was created.
        """
        selector = self.get_selector()

        # Find all interactable objects
        all_interactable = selector.select_interactable(world_state)

        if len(all_interactable) < 2:
            self._is_bound = False
            return False

        # Find objects with binary states that can be controlled
        objects_with_states: List[Tuple[Entity, List[str]]] = []
        for entity in all_interactable:
            states = []
            for prop in entity.properties:
                if prop in BINARY_STATES:
                    states.append(prop)
            # Also infer states from entity type
            if entity.entity_type in {"light", "lamp", "tv", "fan"}:
                if "is_on" not in states:
                    states.append("is_on")
            if entity.entity_type in {"door", "cabinet", "drawer", "fridge"}:
                if "is_open" not in states:
                    states.append("is_open")
            if states:
                objects_with_states.append((entity, states))

        if len(objects_with_states) < 2:
            self._is_bound = False
            return False

        # Create random mappings
        self._mappings.clear()
        self._rng.shuffle(objects_with_states)

        # Need at least num_mappings * 2 objects (controllers and targets)
        num_pairs = min(self.num_mappings, len(objects_with_states) // 2)

        controllers = objects_with_states[:num_pairs]
        targets = objects_with_states[num_pairs : num_pairs * 2]

        for (controller, _), (target, target_states) in zip(controllers, targets):
            # Optionally filter by room
            if self.same_room_only and controller.location != target.location:
                continue

            # Pick a random state to control on the target
            target_state = self._rng.choice(target_states)
            self._mappings[controller.id] = (target.id, target_state)

        if not self._mappings:
            self._is_bound = False
            return False

        # Store bound targets (the controllers)
        self._bound_targets = list(self._mappings.keys())
        self._is_bound = True
        return True

    def applies_to(
        self, action_name: str, target: str, world_state: Any
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if not self._is_bound:
            return False

        # Must be an action that could trigger remote effect
        if action_name not in self._trigger_actions:
            return False

        # Must be acting on one of our controller objects
        return target in self._mappings

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: Any,
    ) -> ActionResult:
        """Transform the action to also affect a remote object."""
        mapping = self._mappings.get(target)
        if not mapping:
            return ActionResult(
                success=False,
                error_message=f"No mapping found for {target}",
                observations={actor_id: f"You interact with {target}. Nothing special happens."},
            )

        controlled_id, controlled_state = mapping
        controlled_entity = world_state.get_entity(controlled_id)

        if not controlled_entity:
            return ActionResult(
                success=False,
                error_message=f"Controlled object {controlled_id} not found",
                observations={actor_id: f"You interact with {target}. Something feels off."},
            )

        # Toggle the controlled state
        old_state = world_state.get_property(controlled_id, controlled_state, False)
        new_state = not old_state

        # Get locations for visibility calculation
        controller_entity = world_state.get_entity(target)
        controller_location = controller_entity.location if controller_entity else None
        controlled_location = controlled_entity.location
        actor_location = world_state.get_agent_location(actor_id)

        # Effect on the distant object
        effect = Effect(
            target=controlled_id,
            property_changed=controlled_state,
            old_value=old_state,
            new_value=new_state,
            visible_to=self._get_observers_in_location(controlled_location, world_state),
            description=f"{controlled_id} {controlled_state} changed to {new_state}",
        )

        # Build observations
        state_adj = self._get_state_adjective(controlled_state, new_state)
        observations: Dict[str, str] = {}
        surprise_triggers: Dict[str, str] = {}

        # Actor observation - they interact with the controller
        base_obs = f"You {action_name.replace('_', ' ')} {target}."

        # Check if actor can see the controlled object
        if actor_location == controlled_location:
            observations[actor_id] = f"{base_obs} The {controlled_id} becomes {state_adj}!"
            surprise_triggers[actor_id] = (
                f"Interacting with {target} affected {controlled_id}"
            )
            self._record_discovery(actor_id, target)
        elif actor_location == controller_location and controller_location != controlled_location:
            # Actor is at controller but controlled is elsewhere - no immediate visible effect
            observations[actor_id] = f"{base_obs} You hear something in the distance."
            surprise_triggers[actor_id] = (
                f"Interacting with {target} caused a sound elsewhere"
            )
        else:
            observations[actor_id] = base_obs

        # Other agents in the controlled object's location see the effect
        for agent_id in self._get_agents_in_location(controlled_location, world_state):
            if agent_id != actor_id:
                observations[agent_id] = f"The {controlled_id} suddenly becomes {state_adj}!"
                surprise_triggers[agent_id] = (
                    f"{controlled_id} changed without anyone nearby"
                )

        # Also apply the intended effect on the controller (if it makes sense)
        effects = [effect]
        if intended_effect.target == target:
            # Also apply the normal effect on the controller
            effects.append(intended_effect)

        return ActionResult(
            success=True,
            effects=effects,
            pending_effects=[],
            observations=observations,
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
        }
        if state_name in adjectives:
            true_adj, false_adj = adjectives[state_name]
            return true_adj if value else false_adj
        return f"{state_name}={value}"

    def _get_observers_in_location(
        self, location: Optional[str], world_state: Any
    ) -> Set[str]:
        """Get IDs of agents who can observe changes in a location."""
        if location is None:
            return set()
        agents = world_state.get_entities_by_type("agent")
        return {a.id for a in agents if a.location == location}

    def _get_agents_in_location(
        self, location: Optional[str], world_state: Any
    ) -> List[str]:
        """Get list of agent IDs in a location."""
        if location is None:
            return []
        agents = world_state.get_entities_by_type("agent")
        return [a.id for a in agents if a.location == location]

    def _record_discovery(self, agent_id: str, controller_id: str) -> None:
        """Record that an agent has discovered a control mapping."""
        if agent_id not in self._discovered:
            self._discovered[agent_id] = set()
        self._discovered[agent_id].add(controller_id)

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        return f"Performing {action_name} on {target} should only affect {target}"

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "mappings": {k: {"target": v[0], "state": v[1]} for k, v in self._mappings.items()},
            "discovered_by_agent": {k: list(v) for k, v in self._discovered.items()},
            "num_mappings": self.num_mappings,
            "same_room_only": self.same_room_only,
        })
        return base_info

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit target mappings.

        Args:
            bindings: List of binding dicts with keys:
                - trigger_object: Object that triggers the remote effect
                - target_object: Object that is affected remotely
                - target_state: State to toggle on target (default: "is_open")

        Returns:
            True if bindings were applied successfully
        """
        self._mappings.clear()
        self._explicit_bindings = bindings

        for binding in bindings:
            trigger = binding.get("trigger_object")
            target = binding.get("target_object")
            state = binding.get("target_state", "is_open")

            if trigger and target:
                self._mappings[trigger] = (target, state)

        self._bound_targets = list(self._mappings.keys())
        self._is_bound = len(self._mappings) > 0
        return self._is_bound

    def reset(self) -> None:
        """Reset per-episode state."""
        super().reset()
        self._discovered.clear()
        self._mappings.clear()
        if self.seed is not None:
            self._rng = random.Random(self.seed)
