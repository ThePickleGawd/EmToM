"""
Object selection utilities for scene-aware mechanics.

Provides affordance-based object discovery that works with whatever objects
exist in a scene, rather than assuming specific object types.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# Standard affordances and their associated states
AFFORDANCE_STATES: Dict[str, List[str]] = {
    "openable": ["is_open"],
    "toggleable": ["is_on"],
    "fillable": ["is_filled"],
    "cleanable": ["is_clean"],
    "pickable": ["is_held"],
    "activatable": ["is_active"],
    "pressable": ["is_pressed"],
}

# Entity types commonly associated with each affordance
AFFORDANCE_TYPES: Dict[str, Set[str]] = {
    "openable": {"door", "cabinet", "drawer", "fridge", "container", "box"},
    "toggleable": {"switch", "light", "lamp", "tv", "microwave", "fan"},
    "fillable": {"cup", "bowl", "pot", "sink", "container", "glass"},
    "cleanable": {"dish", "pan", "counter", "table", "surface"},
    "pickable": {"object", "tool", "item", "cup", "bowl", "book"},
    "activatable": {"button", "switch", "device", "machine"},
    "pressable": {"button", "switch", "lever"},
}

# States that are considered binary (can be toggled/inverted)
BINARY_STATES: Set[str] = {
    "is_open", "is_on", "is_filled", "is_clean", "is_held",
    "is_active", "is_pressed", "is_locked", "is_powered",
}


@dataclass
class ObjectSelector:
    """
    Selects objects from a scene based on their affordances (what they can do).

    This enables mechanics to work with whatever objects exist in a scene,
    rather than assuming specific object types like "door" or "button".
    """

    # Optional filter for entity types to consider
    allowed_types: Optional[Set[str]] = None
    # Optional filter for entity types to exclude
    excluded_types: Set[str] = field(default_factory=lambda: {"room", "agent"})

    def select_by_state(
        self,
        world_state: "TextWorldState",
        state_name: str,
        require_value: Optional[Any] = None,
    ) -> List["Entity"]:
        """
        Find all entities that have a specific state property.

        Args:
            world_state: The world state to search
            state_name: Name of the state property (e.g., "is_open")
            require_value: If set, only return entities where state == this value

        Returns:
            List of entities with the specified state
        """
        results = []
        for entity in world_state.entities.values():
            if self._should_exclude(entity):
                continue
            if state_name in entity.properties:
                if require_value is None or entity.properties[state_name] == require_value:
                    results.append(entity)
        return results

    def select_by_affordance(
        self,
        world_state: "TextWorldState",
        affordance: str,
    ) -> List["Entity"]:
        """
        Find all entities that have a specific affordance.

        Args:
            world_state: The world state to search
            affordance: Affordance name (e.g., "openable", "toggleable")

        Returns:
            List of entities with the specified affordance
        """
        # Get states associated with this affordance
        states = AFFORDANCE_STATES.get(affordance, [])
        expected_types = AFFORDANCE_TYPES.get(affordance, set())

        results = []
        for entity in world_state.entities.values():
            if self._should_exclude(entity):
                continue

            # Check by type
            if entity.entity_type in expected_types:
                results.append(entity)
                continue

            # Check by having the relevant state
            for state in states:
                if state in entity.properties:
                    results.append(entity)
                    break

        return results

    def select_with_binary_state(
        self,
        world_state: "TextWorldState",
    ) -> List[Tuple["Entity", List[str]]]:
        """
        Find all entities that have any binary state that can be toggled.

        Returns:
            List of (entity, [binary_state_names]) tuples
        """
        results = []
        for entity in world_state.entities.values():
            if self._should_exclude(entity):
                continue

            binary_states = []
            for state_name in entity.properties:
                if state_name in BINARY_STATES:
                    binary_states.append(state_name)

            # Also infer from entity type
            for affordance, types in AFFORDANCE_TYPES.items():
                if entity.entity_type in types:
                    for state in AFFORDANCE_STATES.get(affordance, []):
                        if state in BINARY_STATES and state not in binary_states:
                            binary_states.append(state)

            if binary_states:
                results.append((entity, binary_states))

        return results

    def select_interactable(
        self,
        world_state: "TextWorldState",
    ) -> List["Entity"]:
        """
        Find all entities that can be interacted with in some way.

        Returns:
            List of interactable entities
        """
        results = []
        excluded_types = {"room", "agent"}

        for entity in world_state.entities.values():
            if entity.entity_type in excluded_types:
                continue
            if self._should_exclude(entity):
                continue

            # Has any properties (meaning it has state)
            if entity.properties:
                results.append(entity)
            # Or is a known interactable type
            elif entity.entity_type in {"object", "furniture", "switch", "button", "door"}:
                results.append(entity)

        return results

    def select_random(
        self,
        world_state: "TextWorldState",
        count: int = 1,
        filter_fn: Optional[Callable[["Entity"], bool]] = None,
    ) -> List["Entity"]:
        """
        Select random entities from the scene.

        Args:
            world_state: The world state to search
            count: Number of entities to select
            filter_fn: Optional filter function

        Returns:
            List of randomly selected entities
        """
        candidates = []
        for entity in world_state.entities.values():
            if self._should_exclude(entity):
                continue
            if filter_fn is None or filter_fn(entity):
                candidates.append(entity)

        if not candidates:
            return []

        return random.sample(candidates, min(count, len(candidates)))

    def get_available_affordances(
        self,
        world_state: "TextWorldState",
    ) -> Dict[str, List[str]]:
        """
        Get all affordances available in the current scene.

        Returns:
            Dict mapping affordance name -> list of entity IDs that have it
        """
        result: Dict[str, List[str]] = {}

        for affordance in AFFORDANCE_STATES:
            entities = self.select_by_affordance(world_state, affordance)
            if entities:
                result[affordance] = [e.id for e in entities]

        return result

    def get_entities_by_property(
        self,
        world_state: "TextWorldState",
        property_name: str,
        property_value: Optional[Any] = None,
    ) -> List["Entity"]:
        """
        Find entities by a specific property value.

        Args:
            world_state: The world state to search
            property_name: Name of the property to check
            property_value: If set, only return entities where property == this value
                          If None, return all entities that have the property

        Returns:
            List of matching entities
        """
        results = []
        for entity in world_state.entities.values():
            if self._should_exclude(entity):
                continue
            if property_name in entity.properties:
                if property_value is None or entity.properties[property_name] == property_value:
                    results.append(entity)
        return results

    def _should_exclude(self, entity: "Entity") -> bool:
        """Check if entity should be excluded from selection."""
        if entity.entity_type in self.excluded_types:
            return True
        if self.allowed_types is not None and entity.entity_type not in self.allowed_types:
            return True
        return False


def get_entity_affordances(entity: "Entity") -> List[str]:
    """
    Get all affordances that an entity has based on its type and properties.

    Args:
        entity: The entity to analyze

    Returns:
        List of affordance names
    """
    affordances = []

    for affordance, types in AFFORDANCE_TYPES.items():
        if entity.entity_type in types:
            affordances.append(affordance)
            continue

        # Check by properties
        for state in AFFORDANCE_STATES.get(affordance, []):
            if state in entity.properties:
                affordances.append(affordance)
                break

    return list(set(affordances))


def get_entity_binary_states(entity: "Entity") -> List[str]:
    """
    Get all binary states that an entity has or could have.

    Args:
        entity: The entity to analyze

    Returns:
        List of binary state names
    """
    states = []

    # States already in properties
    for prop in entity.properties:
        if prop in BINARY_STATES:
            states.append(prop)

    # Infer from type
    for affordance, types in AFFORDANCE_TYPES.items():
        if entity.entity_type in types:
            for state in AFFORDANCE_STATES.get(affordance, []):
                if state in BINARY_STATES and state not in states:
                    states.append(state)

    return states
