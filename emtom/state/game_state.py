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
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import json
import copy


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
    spawned_by_action: str  # e.g., "Search"
    spawned_from: str  # e.g., "vase_1"
    spawned_at_step: int
    location: Optional[str] = None  # room or position
    picked_up_by: Optional[str] = None  # agent who picked it up


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
    # Items spawned by actions (Search reveals key, etc.)
    spawned_items: List[SpawnedItem] = field(default_factory=list)
    # Objects hidden by Hide action
    hidden_objects: Set[str] = field(default_factory=set)
    # Messages written by WriteMessage action
    written_messages: Dict[str, str] = field(default_factory=dict)
    # World objects spawned on furniture (e.g., key on table)
    world_objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # === Mechanic state (owned here, mechanics are stateless) ===
    # conditional_unlock: targets that have been unlocked
    unlocked_targets: Set[str] = field(default_factory=set)
    # state_mirroring: pairs of linked objects [(obj_a, obj_b, state), ...]
    mirror_pairs: List[Tuple[str, str, str]] = field(default_factory=list)
    # inverse_state: objects with inverted behavior
    inverse_objects: Set[str] = field(default_factory=set)
    # remote_control: mappings from trigger -> (target, state)
    remote_mappings: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # === Theory of Mind mechanics state ===
    # location_change: tracks original locations and moves {obj_id: {"original": loc, "moved_to": loc, "moved_at_step": N, "absent_agents": [...]}}
    location_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # container_swap: tracks swapped contents {container_id: {"original_contents": [...], "swapped_with": container_id, "swapped_at_step": N}}
    container_swaps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # state_change_unseen: tracks state changes agents haven't observed {obj_id: {"property": str, "old_value": Any, "new_value": Any, "changed_at_step": N, "unaware_agents": [...]}}
    unseen_state_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # delayed_information: info revealed after N turns {info_id: {"content": str, "reveal_at_step": N, "target_agents": [...], "revealed": bool}}
    delayed_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Track which room each agent was in at each step (for absence tracking)
    agent_room_history: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)

    # === Communication mechanics state ===
    # limited_bandwidth: tracks message counts per agent {agent_id: count}
    message_counts: Dict[str, int] = field(default_factory=dict)
    # max messages allowed per agent (set via mechanic binding)
    max_messages: Dict[str, int] = field(default_factory=dict)
    # delayed_messages: messages waiting to be delivered [{msg_id, from, to, content, deliver_at_step, sent_at_step}, ...]
    pending_messages: List[Dict[str, Any]] = field(default_factory=list)
    # noisy_channel: tracks noise settings {"corruption_rate": 0.0-1.0, "drop_rate": 0.0-1.0}
    channel_noise: Dict[str, float] = field(default_factory=dict)
    # Message history for all agents
    message_history: List[Dict[str, Any]] = field(default_factory=list)

    # === Coordination mechanics state ===
    # hidden_agenda: per-agent secret goals {agent_id: {"goal": str, "target": str, "achieved": bool, "conflicts_with": [agent_ids]}}
    hidden_agendas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # simultaneous_action: tracks coordinated action requirements {action_id: {"required_agents": [...], "action": str, "target": str, "pending_agents": [...], "window_start": step, "window_size": N}}
    simultaneous_requirements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Track actions taken this step for simultaneous checking {agent_id: {"action": str, "target": str}}
    current_step_actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-agent observation history
    agent_observations: Dict[str, List[str]] = field(default_factory=dict)
    # Per-agent inventory (items collected, not in world graph)
    # Uses Set for O(1) lookup
    agent_inventory: Dict[str, Set[str]] = field(default_factory=dict)

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
    # Maps item_id -> item data dict (stored as dict for serialization)
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
            "unlocked_targets": list(self.unlocked_targets),
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
            # Convert sets to lists for JSON serialization
            "agent_inventory": {
                agent_id: list(items)
                for agent_id, items in self.agent_inventory.items()
            },
            # ToM mechanics state
            "location_changes": self.location_changes,
            "container_swaps": self.container_swaps,
            "unseen_state_changes": self.unseen_state_changes,
            "delayed_info": self.delayed_info,
            "agent_room_history": self.agent_room_history,
            # Communication mechanics state
            "message_counts": self.message_counts,
            "max_messages": self.max_messages,
            "pending_messages": self.pending_messages,
            "channel_noise": self.channel_noise,
            "message_history": self.message_history,
            # Coordination mechanics state
            "hidden_agendas": self.hidden_agendas,
            "simultaneous_requirements": self.simultaneous_requirements,
            "current_step_actions": self.current_step_actions,
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
        state.unlocked_targets = set(data.get("unlocked_targets", []))
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
        # Convert lists back to sets
        state.agent_inventory = {
            agent_id: set(items)
            for agent_id, items in data.get("agent_inventory", {}).items()
        }
        # ToM mechanics state
        state.location_changes = data.get("location_changes", {})
        state.container_swaps = data.get("container_swaps", {})
        state.unseen_state_changes = data.get("unseen_state_changes", {})
        state.delayed_info = data.get("delayed_info", {})
        state.agent_room_history = data.get("agent_room_history", {})
        # Communication mechanics state
        state.message_counts = data.get("message_counts", {})
        state.max_messages = data.get("max_messages", {})
        state.pending_messages = data.get("pending_messages", [])
        state.channel_noise = data.get("channel_noise", {})
        state.message_history = data.get("message_history", [])
        # Coordination mechanics state
        state.hidden_agendas = data.get("hidden_agendas", {})
        state.simultaneous_requirements = data.get("simultaneous_requirements", {})
        state.current_step_actions = data.get("current_step_actions", {})
        return state

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EMTOMGameState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
