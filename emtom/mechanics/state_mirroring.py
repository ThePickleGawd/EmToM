"""
State Mirroring Mechanic.

Two objects always maintain the same state - changing one changes the other.
Works with whatever objects exist in the scene (object-agnostic).

Tests the agent's ability to discover linked/synchronized objects.
Real-world analogy: smart home sync, networked devices, linked switches.
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


@register_mechanic("state_mirroring")
class StateMirroringMechanic(SceneAwareMechanic):
    """
    Two objects always have synchronized states.

    When an agent changes the state of one object in a linked pair,
    the other object automatically changes to match.

    Examples:
    - Opening drawer A also opens drawer B
    - Turning on lamp 1 also turns on lamp 2
    - Both cabinets always open/close together

    This mechanic tests whether agents can:
    1. Detect the unexpected synchronized behavior
    2. Understand the bidirectional link
    3. Use one object to control the other
    4. Predict effects on linked objects
    """

    name = "state_mirroring"
    category = MechanicCategory.HIDDEN_MAPPING
    description = "Paired objects always have the same state"

    required_affordance = None

    def __init__(
        self,
        num_pairs: int = 1,
        seed: Optional[int] = None,
        same_type_only: bool = True,
    ):
        """
        Initialize the state mirroring mechanic.

        Args:
            num_pairs: Number of object pairs to link.
            seed: Random seed for reproducible pairings.
            same_type_only: If True, only pair objects of the same type.
        """
        super().__init__()
        self.num_pairs = num_pairs
        self.seed = seed
        self.same_type_only = same_type_only
        self._rng = random.Random(seed)

        # Mapping from object_id -> (partner_id, state_name)
        # Bidirectional: both objects in a pair have entries
        self._mirrors: Dict[str, Tuple[str, str]] = {}
        # Track which agents have observed mirroring
        self._observed: Dict[str, Set[str]] = {}

    def bind_to_scene(self, world_state: Any) -> bool:
        """Discover objects and create random mirror pairs."""
        selector = self.get_selector()
        all_interactable = selector.select_interactable(world_state)

        if len(all_interactable) < 2:
            self._is_bound = False
            return False

        # Group objects by type and state
        objects_by_type: Dict[str, List[Tuple[Any, List[str]]]] = {}
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
                etype = entity.entity_type
                if etype not in objects_by_type:
                    objects_by_type[etype] = []
                objects_by_type[etype].append((entity, states))

        # Create pairs
        self._mirrors.clear()
        pairs_created = 0

        if self.same_type_only:
            # Pair objects of the same type
            for etype, objects in objects_by_type.items():
                if len(objects) >= 2:
                    self._rng.shuffle(objects)
                    for i in range(0, len(objects) - 1, 2):
                        if pairs_created >= self.num_pairs:
                            break
                        obj_a, states_a = objects[i]
                        obj_b, states_b = objects[i + 1]

                        # Find common state
                        common_states = set(states_a) & set(states_b)
                        if common_states:
                            state = self._rng.choice(list(common_states))
                            self._mirrors[obj_a.id] = (obj_b.id, state)
                            self._mirrors[obj_b.id] = (obj_a.id, state)
                            pairs_created += 1

                    if pairs_created >= self.num_pairs:
                        break
        else:
            # Pair any objects with common states
            all_objects = [
                (e, s) for objs in objects_by_type.values() for e, s in objs
            ]
            self._rng.shuffle(all_objects)

            for i in range(0, len(all_objects) - 1, 2):
                if pairs_created >= self.num_pairs:
                    break
                obj_a, states_a = all_objects[i]
                obj_b, states_b = all_objects[i + 1]

                common_states = set(states_a) & set(states_b)
                if common_states:
                    state = self._rng.choice(list(common_states))
                    self._mirrors[obj_a.id] = (obj_b.id, state)
                    self._mirrors[obj_b.id] = (obj_a.id, state)
                    pairs_created += 1

        if not self._mirrors:
            self._is_bound = False
            return False

        self._bound_targets = list(self._mirrors.keys())
        self._is_bound = True
        return True

    def applies_to(
        self, action_name: str, target: str, world_state: Any
    ) -> bool:
        """Check if this mechanic should handle the action."""
        if not self._is_bound:
            return False

        if action_name not in STATE_ACTIONS:
            return False

        if target not in self._mirrors:
            return False

        # Check if action affects the mirrored state
        state_name, _ = STATE_ACTIONS[action_name]
        _, mirror_state = self._mirrors[target]

        return state_name == mirror_state

    def transform_effect(
        self,
        action_name: str,
        actor_id: str,
        target: str,
        intended_effect: Effect,
        world_state: Any,
    ) -> ActionResult:
        """Transform the action to also affect the mirror partner."""
        partner_id, state_name = self._mirrors[target]
        state_name_action, intended_value = STATE_ACTIONS[action_name]

        # Get current states
        current_state = world_state.get_property(target, state_name, False)
        partner_entity = world_state.get_entity(partner_id)

        if not partner_entity:
            return ActionResult(
                success=True,
                effects=[intended_effect],
                pending_effects=[],
                observations={actor_id: f"You {action_name.replace('_', ' ')} the {target}."},
                surprise_triggers={},
            )

        partner_state = world_state.get_property(partner_id, state_name, False)

        # Calculate new state
        if intended_value is None:
            new_state = not current_state
        else:
            new_state = intended_value

        # Effect on the target
        target_effect = Effect(
            target=target,
            property_changed=state_name,
            old_value=current_state,
            new_value=new_state,
            visible_to={actor_id},
            description=f"{target} becomes {self._get_state_adjective(state_name, new_state)}",
        )

        # Effect on the partner (mirrored)
        partner_effect = Effect(
            target=partner_id,
            property_changed=state_name,
            old_value=partner_state,
            new_value=new_state,
            visible_to=set(),  # Visibility depends on location
            description=f"{partner_id} mirrors {target}",
        )

        # Get locations for observations
        target_entity = world_state.get_entity(target)
        target_location = target_entity.location if target_entity else None
        partner_location = partner_entity.location
        actor_location = world_state.get_agent_location(actor_id)

        # Build observations
        state_adj = self._get_state_adjective(state_name, new_state)
        observations: Dict[str, str] = {}
        surprise_triggers: Dict[str, str] = {}

        # Actor observation
        base_obs = f"You {action_name.replace('_', ' ')} the {target}."

        if actor_location == partner_location or target_location == partner_location:
            # Actor can see both objects
            observations[actor_id] = (
                f"{base_obs} The {partner_id} also becomes {state_adj}!"
            )
            surprise_triggers[actor_id] = (
                f"{target} and {partner_id} changed together"
            )
            self._record_observation(actor_id, target)
        else:
            observations[actor_id] = base_obs

        # Other agents who can see the partner
        for agent_id in self._get_agents_in_location(partner_location, world_state):
            if agent_id != actor_id:
                observations[agent_id] = (
                    f"The {partner_id} suddenly becomes {state_adj}!"
                )
                surprise_triggers[agent_id] = (
                    f"{partner_id} changed without being touched"
                )

        return ActionResult(
            success=True,
            effects=[target_effect, partner_effect],
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
            "is_locked": ("locked", "unlocked"),
        }
        if state_name in adjectives:
            true_adj, false_adj = adjectives[state_name]
            return true_adj if value else false_adj
        return f"{state_name}={value}"

    def _get_agents_in_location(
        self, location: Optional[str], world_state: Any
    ) -> List[str]:
        """Get list of agent IDs in a location."""
        if location is None:
            return []
        agents = world_state.get_entities_by_type("agent")
        return [a.id for a in agents if a.location == location]

    def _record_observation(self, agent_id: str, object_id: str) -> None:
        """Record that an agent has observed the mirroring."""
        if agent_id not in self._observed:
            self._observed[agent_id] = set()
        self._observed[agent_id].add(object_id)

    def get_partner(self, object_id: str) -> Optional[str]:
        """Get the mirror partner of an object."""
        if object_id in self._mirrors:
            return self._mirrors[object_id][0]
        return None

    def get_expected_effect_description(self, action_name: str, target: str) -> str:
        """What an agent would normally expect."""
        return f"{action_name} on {target} should only affect {target}"

    def bind_explicit(self, bindings: List[Dict[str, Any]]) -> bool:
        """
        Bind the mechanic with explicit mirror pairs.

        Args:
            bindings: List of binding dicts with keys:
                - trigger_object: First object in the pair
                - target_object: Second object in the pair (mirror partner)
                - target_state: State to mirror (default: "is_open")

        Returns:
            True if bindings were applied successfully
        """
        self._mirrors.clear()
        self._explicit_bindings = bindings

        for binding in bindings:
            obj_a = binding.get("trigger_object")
            obj_b = binding.get("target_object")
            state = binding.get("target_state", "is_open")

            if obj_a and obj_b:
                # Bidirectional mirroring
                self._mirrors[obj_a] = (obj_b, state)
                self._mirrors[obj_b] = (obj_a, state)

        self._bound_targets = list(self._mirrors.keys())
        self._is_bound = len(self._mirrors) > 0
        return self._is_bound

    def reset(self) -> None:
        """Reset per-episode state."""
        super().reset()
        self._observed.clear()
        self._mirrors.clear()
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def get_hidden_state_for_debug(self) -> Dict[str, Any]:
        """Return debug info about this mechanic."""
        # Deduplicate pairs for cleaner output
        pairs = []
        seen = set()
        for obj_a, (obj_b, state) in self._mirrors.items():
            pair_key = tuple(sorted([obj_a, obj_b]))
            if pair_key not in seen:
                seen.add(pair_key)
                pairs.append({"objects": [obj_a, obj_b], "state": state})

        base_info = super().get_hidden_state_for_debug()
        base_info.update({
            "mirror_pairs": pairs,
            "observed_by_agent": {k: list(v) for k, v in self._observed.items()},
            "num_pairs": self.num_pairs,
            "same_type_only": self.same_type_only,
        })
        return base_info
