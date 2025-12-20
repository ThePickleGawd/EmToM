"""
EMTOM Game State.

Central state management for all EMTOM-specific state.
This is an overlay on top of Habitat's simulator state.

Design principles:
- Single source of truth for all EMTOM state
- Fully serializable (can checkpoint/restore)
- Mechanics are stateless transforms on this state
- Syncs relevant properties from Habitat each step
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum
import json
import copy

if TYPE_CHECKING:
    from emtom.state.items import ItemDefinition


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SpawnedItem:
    """An item that was spawned/revealed during gameplay."""
    item_id: str
    item_type: str
    spawned_by_action: str  # e.g., "Shake"
    spawned_from: str  # e.g., "vase_1"
    spawned_at_step: int
    location: Optional[str] = None  # room or position
    picked_up_by: Optional[str] = None  # agent who picked it up


@dataclass
class PendingEffect:
    """A delayed effect waiting to trigger."""
    effect_id: str
    target: str
    property_name: str
    new_value: Any
    steps_remaining: int
    triggered_by: str  # agent who caused this
    triggered_at_step: int
    description: str = ""


@dataclass
class ActionRecord:
    """Record of an action taken."""
    step: int
    agent_id: str
    action_name: str
    target: Optional[str]
    success: bool
    observation: str
    effects: List[str] = field(default_factory=list)
    mechanic_applied: Optional[str] = None


@dataclass
class Goal:
    """A task goal to be achieved."""
    goal_id: str
    description: str
    goal_type: str  # e.g., "find_item", "change_state", "reach_location"
    target: Optional[str] = None
    target_state: Optional[Dict[str, Any]] = None
    status: GoalStatus = GoalStatus.PENDING
    completed_at_step: Optional[int] = None
    completed_by: Optional[str] = None


@dataclass
class AgentBelief:
    """What an agent believes about the world (for ToM)."""
    agent_id: str
    # Object states the agent believes to be true
    believed_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Objects the agent knows about
    known_objects: Set[str] = field(default_factory=set)
    # Objects the agent has seen hidden
    known_hidden: Set[str] = field(default_factory=set)
    # Mechanics the agent has discovered
    discovered_mechanics: Set[str] = field(default_factory=set)
    # Last known locations of objects
    last_known_locations: Dict[str, str] = field(default_factory=dict)


@dataclass
class EMTOMGameState:
    """
    Central game state for EMTOM.

    All EMTOM-specific state lives here. Mechanics are stateless
    transforms that read/write this state.
    """

    # === Synced from Habitat (updated each step) ===
    agent_positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    agent_rooms: Dict[str, str] = field(default_factory=dict)
    # Ground truth object states from Habitat (is_open, is_on, etc.)
    object_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # All entities in the scene
    entities: List[Dict[str, Any]] = field(default_factory=list)

    # === Our overlay (custom properties) ===
    # Custom properties per object (hidden_items, is_locked, inverse, linked_to, etc.)
    object_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Items spawned by actions (Shake reveals key, etc.)
    spawned_items: List[SpawnedItem] = field(default_factory=list)
    # Objects hidden by Hide action
    hidden_objects: Set[str] = field(default_factory=set)
    # Messages written by WriteMessage action
    written_messages: Dict[str, str] = field(default_factory=dict)
    # World objects spawned on furniture (e.g., key on table)
    world_objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # === Mechanic state (owned here, mechanics are stateless) ===
    # delayed_effect, decaying_state: effects waiting to trigger
    pending_effects: List[PendingEffect] = field(default_factory=list)
    # conditional_unlock: targets that have been unlocked
    unlocked_targets: Set[str] = field(default_factory=set)
    # sequence_lock: progress through sequences (target -> step count)
    sequence_progress: Dict[str, int] = field(default_factory=dict)
    # sequence_lock: whether target is unlocked
    sequence_unlocked: Set[str] = field(default_factory=set)
    # counting_state: interaction counts per object
    interaction_counts: Dict[str, int] = field(default_factory=dict)
    # state_mirroring: pairs of linked objects [(obj_a, obj_b, state), ...]
    mirror_pairs: List[Tuple[str, str, str]] = field(default_factory=list)
    # inverse_state: objects with inverted behavior
    inverse_objects: Set[str] = field(default_factory=set)
    # remote_control: mappings from trigger -> (target, state)
    remote_mappings: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # === Agent beliefs (ToM) ===
    agent_beliefs: Dict[str, AgentBelief] = field(default_factory=dict)
    # Per-agent observation history
    agent_observations: Dict[str, List[str]] = field(default_factory=dict)
    # Per-agent inventory (items collected, not in world graph)
    agent_inventory: Dict[str, List[str]] = field(default_factory=dict)

    # === Timeline ===
    current_step: int = 0
    action_history: List[ActionRecord] = field(default_factory=list)

    # === Goals ===
    goals: List[Goal] = field(default_factory=list)
    completed_goals: Set[str] = field(default_factory=set)

    # === Mechanic bindings (from task definition) ===
    mechanic_bindings: List[Dict[str, Any]] = field(default_factory=list)
    active_mechanics: List[str] = field(default_factory=list)

    # === Item definitions (from task definition) ===
    # Maps item_id -> ItemDefinition dict (stored as dict for serialization)
    item_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_object_property(self, obj_id: str, prop: str, default: Any = None) -> Any:
        """Get a custom property for an object."""
        return self.object_properties.get(obj_id, {}).get(prop, default)

    def set_object_property(self, obj_id: str, prop: str, value: Any) -> "EMTOMGameState":
        """Set a custom property for an object. Returns new state."""
        new_state = copy.copy(self)
        new_props = copy.copy(self.object_properties)
        if obj_id not in new_props:
            new_props[obj_id] = {}
        else:
            new_props[obj_id] = copy.copy(new_props[obj_id])
        new_props[obj_id][prop] = value
        new_state.object_properties = new_props
        return new_state

    def get_habitat_state(self, obj_id: str, prop: str, default: Any = None) -> Any:
        """Get a property from Habitat's object state."""
        return self.object_states.get(obj_id, {}).get(prop, default)

    def add_pending_effect(self, effect: PendingEffect) -> "EMTOMGameState":
        """Add a pending effect. Returns new state."""
        new_state = copy.copy(self)
        new_state.pending_effects = self.pending_effects + [effect]
        return new_state

    def add_spawned_item(self, item: SpawnedItem) -> "EMTOMGameState":
        """Add a spawned item. Returns new state."""
        new_state = copy.copy(self)
        new_state.spawned_items = self.spawned_items + [item]
        return new_state

    def record_action(self, record: ActionRecord) -> "EMTOMGameState":
        """Record an action. Returns new state."""
        new_state = copy.copy(self)
        new_state.action_history = self.action_history + [record]
        return new_state

    def add_observation(self, agent_id: str, observation: str) -> "EMTOMGameState":
        """Add an observation for an agent. Returns new state."""
        new_state = copy.copy(self)
        new_obs = copy.copy(self.agent_observations)
        if agent_id not in new_obs:
            new_obs[agent_id] = []
        else:
            new_obs[agent_id] = new_obs[agent_id] + [observation]
        new_state.agent_observations = new_obs
        return new_state

    def increment_step(self) -> "EMTOMGameState":
        """Increment the step counter. Returns new state."""
        new_state = copy.copy(self)
        new_state.current_step = self.current_step + 1
        return new_state

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for checkpointing."""
        return {
            "agent_positions": self.agent_positions,
            "agent_rooms": self.agent_rooms,
            "object_states": self.object_states,
            "object_properties": self.object_properties,
            "spawned_items": [
                {
                    "item_id": s.item_id,
                    "item_type": s.item_type,
                    "spawned_by_action": s.spawned_by_action,
                    "spawned_from": s.spawned_from,
                    "spawned_at_step": s.spawned_at_step,
                    "location": s.location,
                    "picked_up_by": s.picked_up_by,
                }
                for s in self.spawned_items
            ],
            "hidden_objects": list(self.hidden_objects),
            "world_objects": self.world_objects,
            "pending_effects": [
                {
                    "effect_id": e.effect_id,
                    "target": e.target,
                    "property_name": e.property_name,
                    "new_value": e.new_value,
                    "steps_remaining": e.steps_remaining,
                    "triggered_by": e.triggered_by,
                    "triggered_at_step": e.triggered_at_step,
                    "description": e.description,
                }
                for e in self.pending_effects
            ],
            "unlocked_targets": list(self.unlocked_targets),
            "sequence_progress": self.sequence_progress,
            "sequence_unlocked": list(self.sequence_unlocked),
            "interaction_counts": self.interaction_counts,
            "mirror_pairs": self.mirror_pairs,
            "inverse_objects": list(self.inverse_objects),
            "remote_mappings": self.remote_mappings,
            "current_step": self.current_step,
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "description": g.description,
                    "goal_type": g.goal_type,
                    "target": g.target,
                    "target_state": g.target_state,
                    "status": g.status.value,
                }
                for g in self.goals
            ],
            "completed_goals": list(self.completed_goals),
            "active_mechanics": self.active_mechanics,
            "item_definitions": self.item_definitions,
            "agent_inventory": self.agent_inventory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EMTOMGameState":
        """Deserialize state from dict."""
        state = cls()
        state.agent_positions = data.get("agent_positions", {})
        state.agent_rooms = data.get("agent_rooms", {})
        state.object_states = data.get("object_states", {})
        state.object_properties = data.get("object_properties", {})
        state.spawned_items = [
            SpawnedItem(**s) for s in data.get("spawned_items", [])
        ]
        state.hidden_objects = set(data.get("hidden_objects", []))
        state.world_objects = data.get("world_objects", {})
        state.pending_effects = [
            PendingEffect(**e) for e in data.get("pending_effects", [])
        ]
        state.unlocked_targets = set(data.get("unlocked_targets", []))
        state.sequence_progress = data.get("sequence_progress", {})
        state.sequence_unlocked = set(data.get("sequence_unlocked", []))
        state.interaction_counts = data.get("interaction_counts", {})
        state.mirror_pairs = data.get("mirror_pairs", [])
        state.inverse_objects = set(data.get("inverse_objects", []))
        state.remote_mappings = data.get("remote_mappings", {})
        state.current_step = data.get("current_step", 0)
        state.goals = [
            Goal(
                goal_id=g["goal_id"],
                description=g["description"],
                goal_type=g["goal_type"],
                target=g.get("target"),
                target_state=g.get("target_state"),
                status=GoalStatus(g.get("status", "pending")),
            )
            for g in data.get("goals", [])
        ]
        state.completed_goals = set(data.get("completed_goals", []))
        state.active_mechanics = data.get("active_mechanics", [])
        state.item_definitions = data.get("item_definitions", {})
        state.agent_inventory = data.get("agent_inventory", {})
        return state

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EMTOMGameState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
