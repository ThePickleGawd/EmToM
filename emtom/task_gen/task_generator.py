"""
LLM-based task generator for EMTOM benchmark.

Generates collaborative challenges by feeding trajectory surprises to an LLM
and having it create tasks that leverage the discovered mechanics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set

class TaskCategory(Enum):
    """Category of task - determines evaluation criteria."""

    COOPERATIVE = "cooperative"  # All agents share same goal, must work together
    COMPETITIVE = "competitive"  # Two teams with opposing win conditions
    MIXED = "mixed"  # Shared main goal, but agents have secret conflicting subgoals


def extract_room_ids_from_problem_pddl(problem_pddl: Optional[str]) -> Set[str]:
    """Best-effort extraction of declared room IDs from canonical problem_pddl."""
    if not isinstance(problem_pddl, str) or not problem_pddl.strip():
        return set()

    try:
        from emtom.pddl.problem_pddl import parse_problem_pddl

        parsed = parse_problem_pddl(problem_pddl)
    except Exception:
        return set()

    return {
        name
        for name, obj_type in parsed.objects.items()
        if obj_type == "room"
    }


def normalize_mechanic_binding_dict(
    binding: Dict[str, Any],
    room_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Normalize shorthand mechanic bindings emitted by external agents."""
    if not isinstance(binding, dict):
        return {}

    normalized = dict(binding)
    mechanic_type = normalized.get("mechanic_type")
    known_rooms = {room for room in (room_ids or set()) if isinstance(room, str) and room}

    if mechanic_type == "room_restriction":
        if not isinstance(normalized.get("for_agents"), list):
            agent_id = normalized.get("agent_id")
            if isinstance(agent_id, str) and agent_id:
                normalized["for_agents"] = [agent_id]

        restricted_rooms = normalized.get("restricted_rooms")
        if not (isinstance(restricted_rooms, list) and restricted_rooms):
            allowed_rooms = normalized.get("allowed_rooms")
            if isinstance(allowed_rooms, list):
                allowed_set = {room for room in allowed_rooms if isinstance(room, str) and room}
                normalized["restricted_rooms"] = sorted(known_rooms - allowed_set) if known_rooms else []

    elif mechanic_type == "limited_bandwidth":
        if not isinstance(normalized.get("message_limits"), dict):
            agent_id = normalized.get("agent_id")
            max_messages = normalized.get("max_messages")
            if isinstance(agent_id, str) and isinstance(max_messages, (int, float)):
                normalized["message_limits"] = {agent_id: int(max_messages)}

    elif mechanic_type == "restricted_communication":
        allowed_targets = normalized.get("allowed_targets")
        agent_id = normalized.get("agent_id")
        if isinstance(agent_id, str) and isinstance(allowed_targets, list):
            normalized["allowed_targets"] = {
                agent_id: [target for target in allowed_targets if isinstance(target, str) and target]
            }

    return normalized


def normalize_mechanic_bindings(
    mechanic_bindings: Any,
    problem_pddl: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return mechanic bindings in canonical schema."""
    if not isinstance(mechanic_bindings, list):
        return []

    room_ids = extract_room_ids_from_problem_pddl(problem_pddl)
    normalized: List[Dict[str, Any]] = []
    for binding in mechanic_bindings:
        if isinstance(binding, dict):
            normalized.append(normalize_mechanic_binding_dict(binding, room_ids=room_ids))
    return normalized




@dataclass
class MechanicBinding:
    """Specifies how a mechanic is bound to scene objects."""
    mechanic_type: str  # "inverse_state", "remote_control", "state_mirroring", "conditional_unlock", "room_restriction", etc.
    trigger_object: Optional[str] = None  # Object that triggers (optional - some mechanics use other keys)
    target_object: Optional[str] = None  # For remote_control/state_mirroring: the affected object
    target_state: Optional[str] = None  # State being affected (e.g., "is_open")
    prerequisite_object: Optional[str] = None  # For conditional_unlock: object interaction that unlocks trigger_object
    requires_item: Optional[str] = None  # For conditional_unlock: item that unlocks trigger_object
    count: Optional[int] = None  # Reserved for future use
    # room_restriction mechanic fields
    restricted_rooms: Optional[List[str]] = None  # Rooms agents cannot enter
    for_agents: Optional[List[str]] = None  # Which agents are restricted
    # limited_bandwidth mechanic fields
    message_limits: Optional[Dict[str, int]] = None  # agent_id -> max messages
    # restricted_communication mechanic fields
    allowed_targets: Optional[Dict[str, List[str]]] = None  # agent_id -> allowed recipient agent_ids
    # unreliable_communication mechanic fields
    failure_probability: Optional[Any] = None  # float or dict of agent_id -> float

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "mechanic_type": self.mechanic_type,
        }
        # Only include non-None fields
        if self.trigger_object is not None:
            result["trigger_object"] = self.trigger_object
        if self.target_object is not None:
            result["target_object"] = self.target_object
        if self.target_state is not None:
            result["target_state"] = self.target_state
        if self.prerequisite_object is not None:
            result["prerequisite_object"] = self.prerequisite_object
        if self.requires_item is not None:
            result["requires_item"] = self.requires_item
        if self.count is not None:
            result["count"] = self.count
        if self.restricted_rooms is not None:
            result["restricted_rooms"] = self.restricted_rooms
        if self.for_agents is not None:
            result["for_agents"] = self.for_agents
        if self.message_limits is not None:
            result["message_limits"] = self.message_limits
        if self.allowed_targets is not None:
            result["allowed_targets"] = self.allowed_targets
        if self.failure_probability is not None:
            result["failure_probability"] = self.failure_probability
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicBinding":
        normalized = normalize_mechanic_binding_dict(data)
        return cls(
            mechanic_type=normalized["mechanic_type"],
            trigger_object=normalized.get("trigger_object"),
            target_object=normalized.get("target_object"),
            target_state=normalized.get("target_state"),
            prerequisite_object=normalized.get("prerequisite_object"),
            requires_item=normalized.get("requires_item"),
            count=normalized.get("count"),
            restricted_rooms=normalized.get("restricted_rooms"),
            for_agents=normalized.get("for_agents"),
            message_limits=normalized.get("message_limits"),
            allowed_targets=normalized.get("allowed_targets"),
            failure_probability=normalized.get("failure_probability"),
        )


@dataclass
class GeneratedTask:
    """A collaborative challenge task with clean public/secret separation."""

    task_id: str
    title: str

    # CATEGORY (determines evaluation criteria)
    category: str  # "cooperative", "competitive", or "mixed"

    # SCENE & ENVIRONMENT
    scene_id: str  # Habitat scene ID (e.g., "102817140")
    episode_id: str  # PARTNR dataset episode ID (e.g., "1944")
    active_mechanics: List[str]
    mechanic_bindings: List[MechanicBinding]

    # TASK DESCRIPTION
    task: Optional[str]  # The task description shown to agents

    # PER-AGENT CONFIG
    agent_secrets: Dict[str, List[str]]
    agent_actions: Dict[str, List[str]]

    # METADATA
    num_agents: int

    # Single-format PDDL problem payload (authoritative goal spec)
    pddl_domain: str = "emtom"  # Domain name pinned by task (must match problem_pddl :domain)
    problem_pddl: Optional[str] = None  # Full inline PDDL problem string

    # Scene objects
    items: List[Dict[str, Any]] = field(default_factory=list)
    locked_containers: Dict[str, str] = field(default_factory=dict)
    initial_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # THEORY OF MIND
    tom_level: int = 0
    tom_reasoning: Optional[str] = None
    functional_goal_pddl: Optional[str] = None
    literal_tom_probes: List[Dict[str, Any]] = field(default_factory=list)
    runtime_semantics_version: Optional[str] = None

    # MESSAGE TARGETING (optional, restricts who each agent can message)
    message_targets: Optional[Dict[str, List[str]]] = None

    # COMPETITIVE-SPECIFIC (optional, for category="competitive")
    teams: Optional[Dict[str, List[str]]] = None
    team_secrets: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedTask":
        """Create task from dictionary."""
        problem_pddl = data.get("problem_pddl") if isinstance(data.get("problem_pddl"), str) else None
        if not (problem_pddl and problem_pddl.strip()):
            raise ValueError(
                "Task must define non-empty 'problem_pddl'. "
                "Legacy goal fields are no longer supported."
            )
        from emtom.pddl.problem_pddl import normalize_problem_pddl

        problem_pddl = normalize_problem_pddl(problem_pddl)

        # Parse mechanic bindings
        bindings = []
        raw_bindings = normalize_mechanic_bindings(data.get("mechanic_bindings", []), problem_pddl=problem_pddl)
        for b in raw_bindings:
            bindings.append(MechanicBinding.from_dict(b))

        # Parse items
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []

        # Parse locked_containers
        locked_containers = data.get("locked_containers", {})
        if not isinstance(locked_containers, dict):
            locked_containers = {}

        # Parse initial_states
        initial_states = data.get("initial_states", {})
        if not isinstance(initial_states, dict):
            initial_states = {}
        initial_states = {
            k: v for k, v in initial_states.items()
            if isinstance(v, dict) and not k.startswith("EXAMPLE_")
        }

        # Parse category
        category = data.get("category", "cooperative")
        if category not in ("cooperative", "competitive", "mixed"):
            category = "cooperative"

        # Parse message_targets
        message_targets = data.get("message_targets") if isinstance(data.get("message_targets"), dict) else None

        # Parse competitive-specific fields
        teams = data.get("teams") if isinstance(data.get("teams"), dict) else None
        team_secrets = data.get("team_secrets") if isinstance(data.get("team_secrets"), dict) else None

        # Parse canonical PDDL payload (required end-to-end).
        pddl_domain = data.get("pddl_domain", "emtom")

        # Parse agent config
        agent_secrets = data.get("agent_secrets", {})
        if not isinstance(agent_secrets, dict):
            agent_secrets = {}
        agent_actions = data.get("agent_actions", {})
        if not isinstance(agent_actions, dict):
            agent_actions = {}
        active_mechanics = data.get("active_mechanics", [])
        if not isinstance(active_mechanics, list):
            active_mechanics = []
        literal_tom_probes = data.get("literal_tom_probes", [])
        if not isinstance(literal_tom_probes, list):
            literal_tom_probes = []

        return cls(
            task_id=data.get("task_id", "unknown"),
            title=data.get("title", "Untitled"),
            category=category,
            scene_id=data.get("scene_id", "unknown"),
            episode_id=data.get("episode_id", "unknown"),
            active_mechanics=active_mechanics,
            mechanic_bindings=bindings,
            task=data.get("task"),
            agent_secrets=agent_secrets,
            agent_actions=agent_actions,
            num_agents=data.get("num_agents", 2),
            pddl_domain=pddl_domain,
            problem_pddl=problem_pddl,
            items=items,
            locked_containers=locked_containers,
            initial_states=initial_states,
            tom_level=data.get("tom_level", 0),
            tom_reasoning=data.get("tom_reasoning"),
            functional_goal_pddl=data.get("functional_goal_pddl"),
            literal_tom_probes=literal_tom_probes,
            runtime_semantics_version=data.get("runtime_semantics_version"),
            message_targets=message_targets,
            teams=teams,
            team_secrets=team_secrets,
        )

    # PDDL-related methods

    @property
    def uses_pddl(self) -> bool:
        """Check if this task has a PDDL goal."""
        return self.problem_pddl is not None

    def get_pddl_goal_checker(self, functional_only: bool = True):
        """Create a PDDLGoalChecker for this task."""
        from emtom.pddl.goal_checker import PDDLGoalChecker

        return PDDLGoalChecker.from_task_data(
            self.to_dict(),
            functional_only=functional_only,
        )

    def compute_tom_level(self, scene_data=None) -> int:
        """Compute ToM level from PDDL. Returns stored tom_level if no PDDL."""
        if not self.problem_pddl:
            return self.tom_level
        from emtom.pddl.tom_verifier import compute_tom_depth
        depth = compute_tom_depth(self, scene_data)
        return depth

    def get_pddl_propositions(self) -> List[Dict[str, Any]]:
        """Get goal conjuncts as evaluation.py proposition format."""
        checker = self.get_pddl_goal_checker()
        if checker:
            return checker.to_propositions()
        return []

    def get_literal_tom_probes(self) -> List[Dict[str, Any]]:
        """Get persisted or derived literal-ToM probes for this task."""
        if self.literal_tom_probes:
            return list(self.literal_tom_probes)
        from emtom.pddl.runtime_projection import build_runtime_metadata

        metadata = build_runtime_metadata(self.to_dict())
        return list(metadata.get("literal_tom_probes", []))

    def get_required_pddl_propositions(self) -> List[Dict[str, Any]]:
        """Get only required (non-owned) propositions."""
        return [p for p in self.get_pddl_propositions() if p.get("required") is True]

    def get_team_pddl_propositions(self, team_id: str) -> List[Dict[str, Any]]:
        """Get propositions owned by a team."""
        return [p for p in self.get_pddl_propositions() if p.get("required") == team_id]

    def get_agent_pddl_propositions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get propositions owned by an agent."""
        return [p for p in self.get_pddl_propositions() if p.get("required") == agent_id]
