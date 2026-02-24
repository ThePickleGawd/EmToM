"""
LLM-based task generator for EMTOM benchmark.

Generates collaborative challenges by feeding trajectory surprises to an LLM
and having it create tasks that leverage the discovered mechanics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

class TaskCategory(Enum):
    """Category of task - determines evaluation criteria."""

    COOPERATIVE = "cooperative"  # All agents share same goal, must work together
    COMPETITIVE = "competitive"  # Two teams with opposing win conditions
    MIXED = "mixed"  # Shared main goal, but agents have secret conflicting subgoals




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
        return cls(
            mechanic_type=data["mechanic_type"],
            trigger_object=data.get("trigger_object"),
            target_object=data.get("target_object"),
            target_state=data.get("target_state"),
            count=data.get("count"),
            restricted_rooms=data.get("restricted_rooms"),
            for_agents=data.get("for_agents"),
            message_limits=data.get("message_limits"),
            allowed_targets=data.get("allowed_targets"),
            failure_probability=data.get("failure_probability"),
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
    tom_level: int = 1
    tom_reasoning: Optional[str] = None

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
        """Create task from dictionary.

        Handles migration from legacy formats (goals array, pddl_goal triple)
        by converting to problem_pddl on load.
        """
        # Parse mechanic bindings
        bindings = []
        raw_bindings = data.get("mechanic_bindings", [])
        if isinstance(raw_bindings, list):
            for b in raw_bindings:
                if isinstance(b, dict):
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

        # Parse PDDL — canonical problem_pddl, with one-time migration from legacy
        pddl_domain = data.get("pddl_domain", "emtom")
        problem_pddl = data.get("problem_pddl") if isinstance(data.get("problem_pddl"), str) else None

        if not (problem_pddl and problem_pddl.strip()):
            # Migrate from legacy formats to problem_pddl
            problem_pddl = _migrate_legacy_to_problem_pddl(data)

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
            tom_level=data.get("tom_level", 1),
            tom_reasoning=data.get("tom_reasoning"),
            message_targets=message_targets,
            teams=teams,
            team_secrets=team_secrets,
        )

    # PDDL-related methods

    @property
    def uses_pddl(self) -> bool:
        """Check if this task has a PDDL goal."""
        return self.problem_pddl is not None

    def get_pddl_goal_checker(self):
        """Create a PDDLGoalChecker for this task."""
        from emtom.pddl.goal_checker import PDDLGoalChecker

        return PDDLGoalChecker.from_task_data(self.to_dict())

    def compute_tom_level(self, scene_data=None) -> int:
        """Compute ToM level from PDDL. Returns stored tom_level if no PDDL."""
        if not self.problem_pddl:
            return self.tom_level
        from emtom.pddl.tom_verifier import compute_tom_depth
        depth = compute_tom_depth(self, scene_data)
        return max(depth, 1)

    def get_pddl_propositions(self) -> List[Dict[str, Any]]:
        """Get goal conjuncts as evaluation.py proposition format."""
        checker = self.get_pddl_goal_checker()
        if checker:
            return checker.to_propositions()
        return []

    def get_required_pddl_propositions(self) -> List[Dict[str, Any]]:
        """Get only required (non-owned) propositions."""
        return [p for p in self.get_pddl_propositions() if p.get("required") is True]

    def get_team_pddl_propositions(self, team_id: str) -> List[Dict[str, Any]]:
        """Get propositions owned by a team."""
        return [p for p in self.get_pddl_propositions() if p.get("required") == team_id]

    def get_agent_pddl_propositions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get propositions owned by an agent."""
        return [p for p in self.get_pddl_propositions() if p.get("required") == agent_id]


def _migrate_legacy_to_problem_pddl(data: Dict[str, Any]) -> Optional[str]:
    """One-time migration: convert legacy goal formats to problem_pddl string.

    Handles goals array, pddl_goal triple, and subtask-based tasks.
    Returns None if no goal format is found.
    """
    pddl_goal_str = None

    # Try goals array
    raw_goals = data.get("goals")
    if isinstance(raw_goals, list) and raw_goals:
        pddl_strings = [e.get("pddl", "") for e in raw_goals if isinstance(e, dict)]
        if len(pddl_strings) == 1:
            pddl_goal_str = pddl_strings[0]
        elif pddl_strings:
            pddl_goal_str = "(and " + " ".join(pddl_strings) + ")"

    # Try legacy pddl_goal
    if not pddl_goal_str:
        raw_pddl_goal = data.get("pddl_goal")
        if isinstance(raw_pddl_goal, str) and raw_pddl_goal.strip():
            pddl_goal_str = raw_pddl_goal

    if not pddl_goal_str:
        return None

    # Build a minimal problem_pddl string
    domain_name = data.get("pddl_domain", "emtom")
    task_id = data.get("task_id", "migrated")

    # Migrate legacy pddl_owners to :goal-owners section
    goal_owners_section = ""
    pddl_owners = data.get("pddl_owners")
    if isinstance(pddl_owners, dict) and pddl_owners:
        entries = []
        for literal_str, owner in pddl_owners.items():
            entries.append(f"    ({owner} {literal_str})")
        goal_owners_section = "\n  (:goal-owners\n" + "\n".join(entries) + ")\n"

    return (
        f"(define (problem {task_id})\n"
        f"  (:domain {domain_name})\n"
        f"  (:init)\n"
        f"  (:goal {pddl_goal_str})\n"
        f"{goal_owners_section}"
        f")"
    )
