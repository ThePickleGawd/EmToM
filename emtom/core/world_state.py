"""
Legacy world state classes for text-based simulation.

NOTE: This module is deprecated. The EMTOM benchmark now uses Habitat
for environment simulation. These classes exist only for backwards
compatibility with legacy TaskRunner code.

For new code, use HabitatTaskRunner instead of TaskRunner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Entity:
    """
    Represents an entity in the text-based world simulation.

    DEPRECATED: Use Habitat's world graph nodes instead.
    """

    id: str
    entity_type: str  # "object", "furniture", "room", "agent"
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    affordances: List[str] = field(default_factory=list)

    def get_property(self, prop: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(prop, default)

    def set_property(self, prop: str, value: Any) -> None:
        """Set a property value."""
        self.properties[prop] = value

    def has_affordance(self, affordance: str) -> bool:
        """Check if entity has an affordance."""
        return affordance in self.affordances


class TextWorldState:
    """
    Text-based world state for EMTOM simulation.

    DEPRECATED: The EMTOM benchmark now uses Habitat for environment
    simulation. This class exists only for backwards compatibility.

    For new code, use HabitatTaskRunner and the Habitat environment.
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self._agent_locations: Dict[str, str] = {}

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the world."""
        self.entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_property(self, entity_id: str, prop: str, default: Any = None) -> Any:
        """Get a property of an entity."""
        entity = self.get_entity(entity_id)
        if entity:
            return entity.get_property(prop, default)
        return default

    def set_property(self, entity_id: str, prop: str, value: Any) -> None:
        """Set a property of an entity."""
        entity = self.get_entity(entity_id)
        if entity:
            entity.set_property(prop, value)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """Get the location of an agent."""
        return self._agent_locations.get(agent_id)

    def set_agent_location(self, agent_id: str, location: str) -> None:
        """Set the location of an agent."""
        self._agent_locations[agent_id] = location

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current world state."""
        return {
            "entities": {
                eid: {
                    "id": e.id,
                    "type": e.entity_type,
                    "name": e.name,
                    "properties": dict(e.properties),
                    "location": e.location,
                }
                for eid, e in self.entities.items()
            },
            "agent_locations": dict(self._agent_locations),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "TextWorldState":
        """Restore world state from a snapshot."""
        world = cls()
        for eid, edata in snapshot.get("entities", {}).items():
            entity = Entity(
                id=edata["id"],
                entity_type=edata["type"],
                name=edata["name"],
                properties=edata.get("properties", {}),
                location=edata.get("location"),
            )
            world.add_entity(entity)
        world._agent_locations = dict(snapshot.get("agent_locations", {}))
        return world
