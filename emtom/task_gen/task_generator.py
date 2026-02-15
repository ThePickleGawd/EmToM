"""
LLM-based task generator for EMTOM benchmark.

Generates collaborative challenges by feeding trajectory surprises to an LLM
and having it create tasks that leverage the discovered mechanics.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from emtom.task_gen.trajectory_analyzer import TrajectoryAnalysis


def _ensure_list(value: Any, default: Optional[list] = None) -> list:
    """Coerce value to a list. Returns default (or []) if value is a string or missing."""
    if isinstance(value, list):
        return value
    return default if default is not None else []


def _ensure_dict(value: Any, default: Optional[dict] = None) -> dict:
    """Coerce value to a dict. Returns default (or {}) if value is a string or missing."""
    if isinstance(value, dict):
        return value
    return default if default is not None else {}


def load_scenario_inspirations(
    directory: str = "data/emtom/scenarios/scraped",
    max_scenarios: int = 10,
) -> List[str]:
    """Load scraped scenario briefings to use as creative inspiration."""
    path = Path(directory)
    if not path.exists():
        return []

    txt_files = list(path.glob("*.txt"))
    if not txt_files:
        return []

    if len(txt_files) > max_scenarios:
        txt_files = random.sample(txt_files, max_scenarios)

    scenarios = []
    for txt_file in txt_files:
        try:
            with open(txt_file, encoding="utf-8") as f:
                scenarios.append(f.read().strip())
        except Exception:
            continue

    return scenarios


class TaskCategory(Enum):
    """Category of task - determines evaluation criteria."""

    COOPERATIVE = "cooperative"  # All agents share same goal, must work together
    COMPETITIVE = "competitive"  # Two teams with opposing win conditions
    MIXED = "mixed"  # Shared main goal, but agents have secret conflicting subgoals




@dataclass
class Subtask:
    """
    A subtask within a larger challenge (node in task DAG).

    Subtasks form a DAG where:
    - Each node has a success_condition (PARTNR predicate)
    - Edges are defined by depends_on

    The `required` field is polymorphic based on task category:
    - True: Must be completed for task success (cooperative) or shared prerequisite
    - False: Optional/tracking only
    - "team_X": Completing this means team_X wins (competitive)
    - "agent_X": This is agent_X's personal subgoal (mixed)
    """

    subtask_id: str
    description: str
    success_condition: Dict[str, Any]
    assigned_agent: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    required: Union[bool, str] = True  # bool for cooperative, str for team/agent ownership

    @property
    def id(self) -> str:
        """Alias for subtask_id for cleaner access."""
        return self.subtask_id

    @property
    def is_required_for_task(self) -> bool:
        """Check if this subtask is required for overall task success."""
        return self.required is True

    @property
    def is_optional(self) -> bool:
        """Check if this subtask is optional."""
        return self.required is False

    @property
    def owner(self) -> Optional[str]:
        """Get the team/agent owner if this is a team/agent-specific subtask."""
        if isinstance(self.required, str):
            return self.required
        return None

    def has_valid_condition(self) -> bool:
        """Check if success_condition is properly defined."""
        if not self.success_condition or not isinstance(self.success_condition, dict):
            return False
        return "entity" in self.success_condition and "property" in self.success_condition

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        # Support both 'id' and 'subtask_id' fields
        subtask_id = data.get("subtask_id") or data.get("id", "unknown")
        return cls(
            subtask_id=subtask_id,
            description=data.get("description", ""),
            success_condition=data.get("success_condition", {}),
            assigned_agent=data.get("assigned_agent"),
            depends_on=data.get("depends_on", []),
            hints=data.get("hints", []),
            required=data.get("required", True),  # Default to True for backwards compat
        )


@dataclass
class SuccessCondition:
    """Defines what success looks like for a task."""

    description: str
    required_states: List[Dict[str, Any]]
    time_limit: Optional[int] = None
    all_agents_must_survive: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessCondition":
        return cls(
            description=data["description"],
            required_states=data["required_states"],
            time_limit=data.get("time_limit"),
            all_agents_must_survive=data.get("all_agents_must_survive", True),
        )


@dataclass
class MechanicBinding:
    """Specifies how a mechanic is bound to scene objects."""
    mechanic_type: str  # "inverse_state", "remote_control", "state_mirroring", "conditional_unlock", "room_restriction", etc.
    trigger_object: Optional[str] = None  # Object that triggers (optional - some mechanics use other keys)
    target_object: Optional[str] = None  # For remote_control/state_mirroring: the affected object
    target_state: Optional[str] = None  # State being affected (e.g., "is_open")
    count: Optional[int] = None  # Reserved for future use
    # room_restriction mechanic fields
    restricted_rooms: Optional[List[str]] = None  # Rooms agents cannot enter
    for_agents: Optional[List[str]] = None  # Which agents are restricted

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
        if self.count is not None:
            result["count"] = self.count
        if self.restricted_rooms is not None:
            result["restricted_rooms"] = self.restricted_rooms
        if self.for_agents is not None:
            result["for_agents"] = self.for_agents
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicBinding":
        return cls(
            mechanic_type=data["mechanic_type"],
            trigger_object=data.get("trigger_object"),
            target_object=data.get("target_object"),
            target_state=data.get("target_state"),
            count=data.get("count"),
            restricted_rooms=data.get("restricted_rooms"),
            for_agents=data.get("for_agents"),
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

    # INTERNAL (not shown to agents)
    # success_condition is optional - derived from terminal subtasks if not provided
    success_condition: Optional[SuccessCondition]

    # METADATA
    num_agents: int

    # Optional
    subtasks: List[Subtask] = field(default_factory=list)
    items: List[Dict[str, Any]] = field(default_factory=list)
    locked_containers: Dict[str, str] = field(default_factory=dict)
    initial_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Object -> {property: value}

    # THEORY OF MIND
    tom_level: int = 1  # 1=beliefs about others, 2=beliefs about beliefs, 3=3rd-order nesting
    tom_reasoning: Optional[str] = None  # Why this task requires this ToM level

    # MESSAGE TARGETING (optional, restricts who each agent can message)
    message_targets: Optional[Dict[str, List[str]]] = None  # agent_id -> [allowed recipient agent_ids]

    # COMPETITIVE-SPECIFIC (optional, for category="competitive")
    teams: Optional[Dict[str, List[str]]] = None  # team_id -> [agent_ids], e.g. {"team_0": ["agent_0"], "team_1": ["agent_1"]}
    team_secrets: Optional[Dict[str, List[str]]] = None  # team_id -> [secrets]
    # NOTE: team_goals are now unified into subtasks with required="team_X"
    # NOTE: agent_subgoals are now unified into subtasks with required="agent_X"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedTask":
        """Create task from dictionary."""
        # Parse mechanic bindings if present
        bindings = []
        raw_bindings = _ensure_list(data.get("mechanic_bindings", []))
        for b in raw_bindings:
            if isinstance(b, dict):
                bindings.append(MechanicBinding.from_dict(b))
            # Skip non-dict entries (strings, etc.)

        # Parse subtasks - handle both string and dict formats
        subtasks = []
        raw_subtasks = _ensure_list(data.get("subtasks", []))
        for i, s in enumerate(raw_subtasks):
            if isinstance(s, str):
                # Convert simple string to Subtask
                subtasks.append(Subtask(
                    subtask_id=f"subtask_{i}",
                    description=s,
                    success_condition={},
                ))
            elif isinstance(s, dict):
                subtasks.append(Subtask.from_dict(s))

        # Parse success_condition if present and valid
        success_condition = None
        raw_success = data.get("success_condition")
        if isinstance(raw_success, dict) and "description" in raw_success:
            success_condition = SuccessCondition.from_dict(raw_success)

        # Parse items - keep as list of dicts
        items = _ensure_list(data.get("items", []))

        # Parse locked_containers
        locked_containers = _ensure_dict(data.get("locked_containers", {}))

        # Parse initial_states (object -> {property: value})
        initial_states = _ensure_dict(data.get("initial_states", {}))
        # Filter out placeholder examples
        initial_states = {
            k: v for k, v in initial_states.items()
            if isinstance(v, dict) and not k.startswith("EXAMPLE_")
        }

        # Parse category (default to cooperative for backwards compatibility)
        category = data.get("category", "cooperative")
        if category not in ("cooperative", "competitive", "mixed"):
            category = "cooperative"

        # Parse message_targets (None if not a dict)
        message_targets = data.get("message_targets") if isinstance(data.get("message_targets"), dict) else None

        # Parse competitive-specific fields (None if not a dict)
        teams = data.get("teams") if isinstance(data.get("teams"), dict) else None
        team_secrets = data.get("team_secrets") if isinstance(data.get("team_secrets"), dict) else None
        # NOTE: team_goals and agent_subgoals are now unified into subtasks

        return cls(
            task_id=data.get("task_id", "unknown"),
            title=data.get("title", "Untitled"),
            category=category,
            scene_id=data.get("scene_id", "unknown"),
            episode_id=data.get("episode_id", "unknown"),
            active_mechanics=_ensure_list(data.get("active_mechanics", [])),
            mechanic_bindings=bindings,
            task=data.get("task"),
            agent_secrets=_ensure_dict(data.get("agent_secrets", {})),
            agent_actions=_ensure_dict(data.get("agent_actions", {})),
            success_condition=success_condition,
            num_agents=data.get("num_agents", 2),
            subtasks=subtasks,
            items=items,
            locked_containers=locked_containers,
            initial_states=initial_states,
            tom_level=data.get("tom_level", 1),
            tom_reasoning=data.get("tom_reasoning"),
            message_targets=message_targets,
            teams=teams,
            team_secrets=team_secrets,
        )

    # DAG-related methods

    def get_terminal_subtasks(self) -> List[Subtask]:
        """Get terminal subtasks (nodes with no dependents)."""
        from .dag import find_terminal_nodes
        return find_terminal_nodes(self.subtasks)

    def get_terminal_conditions(self) -> List[Dict[str, Any]]:
        """Get success conditions from terminal subtasks."""
        terminals = self.get_terminal_subtasks()
        return [s.success_condition for s in terminals if s.has_valid_condition()]

    def get_required_subtasks(self) -> List[Subtask]:
        """Get subtasks marked as required for task success (required=True only)."""
        return [s for s in self.subtasks if s.is_required_for_task]

    def get_required_conditions(self) -> List[Dict[str, Any]]:
        """Get success conditions from required subtasks."""
        required = self.get_required_subtasks()
        return [s.success_condition for s in required if s.has_valid_condition()]

    def get_team_subtasks(self, team_id: str) -> List[Subtask]:
        """Get subtasks owned by a specific team (required="team_X")."""
        return [s for s in self.subtasks if s.owner == team_id]

    def get_agent_subtasks(self, agent_id: str) -> List[Subtask]:
        """Get subtasks owned by a specific agent (required="agent_X")."""
        return [s for s in self.subtasks if s.owner == agent_id]

    def get_all_teams(self) -> List[str]:
        """Get all team IDs that have subtasks assigned to them."""
        teams = set()
        for s in self.subtasks:
            if s.owner and s.owner.startswith("team_"):
                teams.add(s.owner)
        return sorted(teams)

    def get_effective_success_condition(self) -> Optional[SuccessCondition]:
        """
        Get the effective success condition.

        If success_condition is explicitly set, return it.
        Otherwise, derive from required subtasks.
        """
        if self.success_condition:
            return self.success_condition

        # Derive from required subtasks
        required_conditions = self.get_required_conditions()
        if not required_conditions:
            return None

        return SuccessCondition(
            description=f"Complete all {len(required_conditions)} required subtask(s)",
            required_states=required_conditions,
        )

    def validate_subtask_dag(self) -> Tuple[bool, List[str]]:
        """Validate the subtask DAG structure."""
        from .dag import validate_dag
        return validate_dag(self.subtasks)

    def create_dag_progress(self):
        """Create a DAGProgress tracker for this task."""
        from .dag import DAGProgress
        return DAGProgress.from_subtasks(self.subtasks)

    def has_valid_dag(self) -> bool:
        """Check if task has a valid subtask DAG."""
        if not self.subtasks:
            return False
        is_valid, _ = self.validate_subtask_dag()
        return is_valid
