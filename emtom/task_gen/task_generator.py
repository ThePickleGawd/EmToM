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
from typing import Any, Dict, List, Optional, Tuple

from emtom.task_gen.trajectory_analyzer import TrajectoryAnalysis


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


class TaskType(Enum):
    """Type of task to generate."""

    THEORY_OF_MIND = "theory_of_mind"  # Tasks requiring theory of mind reasoning
    REGULAR = "regular"  # Simple everyday tasks without ToM requirements


class TaskCategory(Enum):
    """Categories of collaborative tasks."""

    COORDINATION = "coordination"  # Agents must coordinate actions
    KNOWLEDGE_ASYMMETRY = "knowledge_asymmetry"  # One agent knows something others don't
    COMMUNICATION = "communication"  # Agents must share information to succeed
    SEQUENTIAL = "sequential"  # Tasks must be done in order
    RESOURCE_SHARING = "resource_sharing"  # Agents share limited resources/abilities
    SIMPLE_ACTION = "simple_action"  # Single agent simple actions (for regular tasks)


@dataclass
class Subtask:
    """
    A subtask within a larger challenge (node in task DAG).

    Subtasks form a DAG where:
    - Each node has a success_condition (PARTNR predicate)
    - Edges are defined by depends_on
    - Terminal nodes (no dependents) define task success
    """

    subtask_id: str
    description: str
    success_condition: Dict[str, Any]
    assigned_agent: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Alias for subtask_id for cleaner access."""
        return self.subtask_id

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
    mechanic_type: str  # "inverse_state", "remote_control", "state_mirroring", "conditional_unlock"
    trigger_object: str  # Object that triggers the mechanic (e.g., "fridge_58")
    target_object: Optional[str] = None  # For remote_control/state_mirroring: the affected object
    target_state: Optional[str] = None  # State being affected (e.g., "is_open")
    count: Optional[int] = None  # Reserved for future use

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanic_type": self.mechanic_type,
            "trigger_object": self.trigger_object,
            "target_object": self.target_object,
            "target_state": self.target_state,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanicBinding":
        return cls(
            mechanic_type=data["mechanic_type"],
            trigger_object=data["trigger_object"],
            target_object=data.get("target_object"),
            target_state=data.get("target_state"),
            count=data.get("count"),
        )


@dataclass
class GeneratedTask:
    """A collaborative challenge task with clean public/secret separation."""

    task_id: str
    title: str
    category: TaskCategory

    # SCENE & ENVIRONMENT
    scene_id: str  # Habitat scene ID (e.g., "102817140")
    episode_id: str  # Trajectory timestamp (e.g., "20251219_194151")
    dataset_episode_id: str  # Actual Habitat dataset episode ID (e.g., "1") for loading
    active_mechanics: List[str]
    mechanic_bindings: List[MechanicBinding]

    # NARRATIVE
    story: Optional[str]  # Atmospheric narrative setting up the puzzle

    # PUBLIC (shared, no secrets)
    public_goal: str
    public_context: Optional[str]

    # PER-AGENT CONFIG
    agent_secrets: Dict[str, List[str]]
    agent_roles: Dict[str, str]
    agent_actions: Dict[str, List[str]]

    # INTERNAL (not shown to agents)
    # success_condition is optional - derived from terminal subtasks if not provided
    success_condition: Optional[SuccessCondition]

    # METADATA
    num_agents: int
    theory_of_mind_required: bool

    # Optional
    subtasks: List[Subtask] = field(default_factory=list)
    items: List[Dict[str, Any]] = field(default_factory=list)
    locked_containers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["category"] = self.category.value
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedTask":
        """Create task from dictionary."""
        # Parse mechanic bindings if present
        bindings = []
        raw_bindings = data.get("mechanic_bindings", [])
        # Handle case where LLM copied placeholder string instead of actual bindings
        if isinstance(raw_bindings, str):
            raw_bindings = []  # Ignore string placeholders
        for b in raw_bindings:
            if isinstance(b, dict):
                bindings.append(MechanicBinding.from_dict(b))
            # Skip non-dict entries (strings, etc.)

        # Parse subtasks - handle both string and dict formats
        subtasks = []
        raw_subtasks = data.get("subtasks", [])
        if isinstance(raw_subtasks, str):
            raw_subtasks = []  # Ignore string placeholders
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
        items = data.get("items", [])
        if isinstance(items, str):
            items = []

        # Parse locked_containers
        locked_containers = data.get("locked_containers", {})
        if isinstance(locked_containers, str):
            locked_containers = {}

        return cls(
            task_id=data.get("task_id", "unknown"),
            title=data.get("title", "Untitled"),
            category=TaskCategory(data.get("category", "knowledge_asymmetry")),
            scene_id=data.get("scene_id", "unknown"),
            episode_id=data.get("episode_id", "unknown"),
            dataset_episode_id=data.get("dataset_episode_id", ""),  # Required - must be set explicitly
            active_mechanics=data.get("active_mechanics", []) if isinstance(data.get("active_mechanics"), list) else [],
            mechanic_bindings=bindings,
            story=data.get("story"),
            public_goal=data.get("public_goal", ""),
            public_context=data.get("public_context"),
            agent_secrets=data.get("agent_secrets", {}) if isinstance(data.get("agent_secrets"), dict) else {},
            agent_roles=data.get("agent_roles", {}) if isinstance(data.get("agent_roles"), dict) else {},
            agent_actions=data.get("agent_actions", {}) if isinstance(data.get("agent_actions"), dict) else {},
            success_condition=success_condition,
            num_agents=data.get("num_agents", 2),
            theory_of_mind_required=data.get("theory_of_mind_required", False),
            subtasks=subtasks,
            items=items,
            locked_containers=locked_containers,
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

    def get_effective_success_condition(self) -> Optional[SuccessCondition]:
        """
        Get the effective success condition.

        If success_condition is explicitly set, return it.
        Otherwise, derive from terminal subtasks.
        """
        if self.success_condition:
            return self.success_condition

        # Derive from terminal subtasks
        terminal_conditions = self.get_terminal_conditions()
        if not terminal_conditions:
            return None

        return SuccessCondition(
            description=f"Complete all {len(terminal_conditions)} terminal subtask(s)",
            required_states=terminal_conditions,
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
