"""
LLM-based task generator for EMTOM benchmark.

Generates collaborative challenges by feeding trajectory surprises to an LLM
and having it create tasks that leverage the discovered mechanics.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from emtom.task_gen.trajectory_analyzer import TrajectoryAnalysis


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
    """A subtask within a larger challenge."""

    subtask_id: str
    description: str
    success_condition: Dict[str, Any]
    assigned_agent: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        return cls(
            subtask_id=data["subtask_id"],
            description=data["description"],
            success_condition=data["success_condition"],
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
class FailureCondition:
    """Defines what causes task failure."""

    description: str
    failure_states: List[Dict[str, Any]] = field(default_factory=list)
    max_failed_attempts: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureCondition":
        return cls(
            description=data["description"],
            failure_states=data.get("failure_states", []),
            max_failed_attempts=data.get("max_failed_attempts"),
        )


@dataclass
class MechanicBinding:
    """Specifies how a mechanic is bound to scene objects."""
    mechanic_type: str  # "inverse_state", "remote_control", "counting_state"
    trigger_object: str  # Object that triggers the mechanic (e.g., "fridge_58")
    target_object: Optional[str] = None  # For remote_control: the affected object
    target_state: Optional[str] = None  # State being affected (e.g., "is_open")
    count: Optional[int] = None  # For counting_state: number of interactions needed

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
    scene_id: str
    active_mechanics: List[str]
    mechanic_bindings: List[MechanicBinding]

    # PUBLIC (shared, no secrets)
    public_goal: str
    public_context: Optional[str]

    # PER-AGENT CONFIG
    agent_secrets: Dict[str, List[str]]
    agent_roles: Dict[str, str]
    agent_actions: Dict[str, List[str]]

    # INTERNAL (not shown to agents)
    success_condition: SuccessCondition
    failure_conditions: List[FailureCondition]
    initial_world_state: Dict[str, Any]

    # METADATA
    num_agents: int
    difficulty: int
    theory_of_mind_required: bool

    # Optional
    subtasks: List[Subtask] = field(default_factory=list)
    source_trajectory: Optional[str] = None

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
        for b in data.get("mechanic_bindings", []):
            bindings.append(MechanicBinding.from_dict(b))

        # Parse subtasks - handle both string and dict formats
        subtasks = []
        for i, s in enumerate(data.get("subtasks", [])):
            if isinstance(s, str):
                # Convert simple string to Subtask
                subtasks.append(Subtask(
                    subtask_id=f"subtask_{i}",
                    description=s,
                    success_condition={},
                ))
            else:
                subtasks.append(Subtask.from_dict(s))

        return cls(
            task_id=data["task_id"],
            title=data["title"],
            category=TaskCategory(data["category"]),
            scene_id=data.get("scene_id", "unknown"),
            active_mechanics=data.get("active_mechanics", []),
            mechanic_bindings=bindings,
            public_goal=data["public_goal"],
            public_context=data.get("public_context"),
            agent_secrets=data.get("agent_secrets", {}),
            agent_roles=data.get("agent_roles", {}),
            agent_actions=data.get("agent_actions", {}),
            success_condition=SuccessCondition.from_dict(data["success_condition"]),
            failure_conditions=[FailureCondition.from_dict(f) for f in data.get("failure_conditions", [])],
            initial_world_state=data.get("initial_world_state", {}),
            num_agents=data.get("num_agents", 2),
            difficulty=data.get("difficulty", 3),
            theory_of_mind_required=data.get("theory_of_mind_required", False),
            subtasks=subtasks,
            source_trajectory=data.get("source_trajectory"),
        )


TASK_GENERATION_PROMPT = '''You are a task designer for a multi-agent collaboration benchmark in a simulated home environment.

## CRITICAL CONSTRAINT - READ CAREFULLY
You MUST ONLY use objects and furniture that exist in the scene inventory below.
DO NOT invent or hallucinate objects like "device", "key", "battery", "PIN", "box", etc.
The task must be completable using ONLY the real objects and furniture listed.

## Scene Inventory (ONLY use these)
Rooms: {rooms}
Furniture (can navigate to, some can be opened): {furniture}
Objects (can be picked up, hidden, inspected): {objects}
Articulated Furniture (can be opened/closed): {articulated}

## Surprises Discovered During Exploration
{surprises}

## Task Requirements
The TYPE of task generated is based on user input:
- If user selects 1: Generate Theory of Mind tasks
- If user selects 2: Generate Regular tasks

Selected task type: {task_type}

### THEORY OF MIND TASKS (option 1):
- Design a task for {num_agents} agents working together
- Use ONLY objects/furniture from the Scene Inventory above
- The task should leverage the discovered mechanics (surprising behaviors)
- agent_0 knows about the mechanics (gets secrets), agent_1 does not (theory of mind)
- Agents must communicate and coordinate to succeed

### REGULAR TASKS (option 2):
- Design simple, everyday household tasks for {num_agents} agent(s)
- Use ONLY objects/furniture from the Scene Inventory above
- Tasks should be straightforward actions like moving objects, opening/closing furniture

## Output Format
Respond with a JSON object:
{{
    "title": "Short descriptive title (max 10 words)",
    "category": "knowledge_asymmetry",
    "public_goal": "Simple imperative goal shared by all agents (e.g., 'Place phone_stand_2 on table_10')",
    "public_context": "Optional shared context without secrets (e.g., 'Some items may be hidden in furniture')",
    "initial_world_state": {{
        "objects": ["REAL objects from inventory"],
        "agent_positions": {{"agent_0": "REAL_room_name", "agent_1": "REAL_room_name"}}
    }},
    "agent_secrets": {{
        "agent_0": ["Secret knowledge only agent_0 knows (e.g., 'Opening fridge_58 also opens chest_of_drawers_52')"],
        "agent_1": []
    }},
    "agent_roles": {{
        "agent_0": "Expert who discovered the mechanics",
        "agent_1": "Novice who must learn through collaboration"
    }},
    "agent_actions": {{
        "agent_0": ["Navigate", "Open", "Close", "Pick", "Place", "Use", "Inspect", "Communicate"],
        "agent_1": ["Navigate", "Open", "Close", "Pick", "Place", "Communicate"]
    }},
    "success_condition": {{
        "description": "What success looks like using REAL object names",
        "required_states": [{{"entity": "REAL_object_name", "property": "location", "value": "REAL_target"}}]
    }},
    "failure_conditions": [
        {{"description": "What causes failure"}}
    ],
    "difficulty": 3
}}

Generate the task JSON (remember: ONLY use objects from the Scene Inventory):'''


class TaskGenerator:
    """
    LLM-based task generator that creates collaborative challenges
    from trajectory surprises.
    """

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client

    def generate_tasks(
        self,
        trajectory: Dict[str, Any],
        analysis: TrajectoryAnalysis,
        num_agents: int = 2,
        max_tasks: int = 5,
        task_type: int = 1,
    ) -> List[GeneratedTask]:
        """
        Generate collaborative tasks from a trajectory using LLM.

        Args:
            trajectory: Original trajectory dict with steps and surprises
            analysis: Analysis with discovered mechanics
            num_agents: Number of agents for tasks
            max_tasks: Maximum tasks to generate
            task_type: Type of task to generate:
                       1 = Theory of Mind tasks
                       2 = Regular tasks

        Returns:
            List of generated challenge tasks
        """
        if self.llm is None:
            raise ValueError("LLM client required for task generation")

        # Extract surprises from trajectory
        surprises = trajectory.get("surprise_summary", [])
        if not surprises:
            # Also check steps for surprises
            for step in trajectory.get("steps", []):
                surprises.extend(step.get("surprises", []))

        if not surprises:
            print("  WARNING: No surprises found in trajectory")
            return []

        # Get scene inventory (critical for grounding tasks in real objects)
        scene_inventory = trajectory.get("scene_inventory", {})
        if not scene_inventory:
            print("  WARNING: No scene inventory in trajectory - tasks may use fictional objects")
            # Fallback: extract objects from trajectory actions
            objects_seen = set()
            for step in trajectory.get("steps", []):
                for action in step.get("agent_actions", {}).values():
                    if action.get("target"):
                        objects_seen.add(action["target"])
            scene_inventory = {
                "rooms": [],
                "furniture": [],
                "objects": list(objects_seen),
                "articulated_furniture": [],
            }

        # Get scene info
        scene_id = trajectory.get("metadata", {}).get("scene_id", "unknown")
        mechanics = trajectory.get("mechanics_active", [])

        # Get mechanic bindings from trajectory (critical for tasks to work!)
        raw_bindings = trajectory.get("mechanic_bindings", {})
        mechanic_bindings = self._convert_trajectory_bindings(raw_bindings)
        if mechanic_bindings:
            print(f"  Found {len(mechanic_bindings)} mechanic bindings from exploration")
        else:
            print("  WARNING: No mechanic bindings found - tasks may not work!")

        # Format surprises for prompt
        surprise_text = self._format_surprises(surprises)

        # Convert task_type to string for prompt
        task_type_str = "Theory of Mind (option 1)" if task_type == 1 else "Regular (option 2)"

        # Generate tasks
        tasks = []
        num_to_generate = min(max_tasks, len(surprises))
        for i in range(num_to_generate):
            print(f"    Generating task {i+1}/{num_to_generate}...", end=" ", flush=True)
            try:
                task = self._generate_single_task(
                    scene_id=scene_id,
                    mechanics=mechanics,
                    surprises=surprise_text,
                    scene_inventory=scene_inventory,
                    num_agents=num_agents,
                    episode_id=trajectory.get("episode_id", "unknown"),
                    task_type=task_type,
                    task_type_str=task_type_str,
                    mechanic_bindings=mechanic_bindings,
                )
                if task:
                    tasks.append(task)
                    print(f"OK - {task.title[:40]}...")
                else:
                    print("SKIP (no valid task)")
            except Exception as e:
                print(f"FAIL: {e}")

        return tasks

    def _convert_trajectory_bindings(
        self, raw_bindings: Dict[str, Any]
    ) -> List[MechanicBinding]:
        """
        Convert trajectory bindings to MechanicBinding objects.

        Args:
            raw_bindings: Dict from trajectory, e.g.:
                {
                    "inverse_state": {"targets": ["fridge_58"]},
                    "remote_control": [{"trigger": "chest_52", "target": "table_59", "target_state": "is_open"}],
                    "counting_state": {"targets": {"cabinet_57": 0}},
                }

        Returns:
            List of MechanicBinding objects
        """
        bindings = []

        # inverse_state: list of targets
        if "inverse_state" in raw_bindings:
            targets = raw_bindings["inverse_state"].get("targets", [])
            for target in targets:
                bindings.append(MechanicBinding(
                    mechanic_type="inverse_state",
                    trigger_object=target,
                ))

        # remote_control: list of {trigger, target, target_state}
        if "remote_control" in raw_bindings:
            rc_list = raw_bindings["remote_control"]
            if isinstance(rc_list, list):
                for rc in rc_list:
                    bindings.append(MechanicBinding(
                        mechanic_type="remote_control",
                        trigger_object=rc["trigger"],
                        target_object=rc["target"],
                        target_state=rc.get("target_state", "is_open"),
                    ))

        # counting_state: dict of {target: current_count}
        if "counting_state" in raw_bindings:
            targets = raw_bindings["counting_state"].get("targets", {})
            for target, count in targets.items():
                bindings.append(MechanicBinding(
                    mechanic_type="counting_state",
                    trigger_object=target,
                    count=3,  # Default required count
                ))

        return bindings

    def _format_surprises(self, surprises: List[Dict[str, Any]]) -> str:
        """Format surprises for the prompt."""
        lines = []
        for i, s in enumerate(surprises, 1):
            lines.append(f"{i}. Action: {s.get('action', 'unknown')} on {s.get('target', 'unknown')}")
            lines.append(f"   Surprise Level: {s.get('surprise_level', 'N/A')}/5")
            lines.append(f"   What happened: {s.get('explanation', 'No explanation')}")
            lines.append(f"   Hypothesis: {s.get('hypothesis', 'No hypothesis')}")
            lines.append("")
        return "\n".join(lines)

    def _generate_single_task(
        self,
        scene_id: str,
        mechanics: List[str],
        surprises: str,
        scene_inventory: Dict[str, List[str]],
        num_agents: int,
        episode_id: str,
        task_type: int = 1,
        task_type_str: str = "Theory of Mind (option 1)",
        mechanic_bindings: Optional[List[MechanicBinding]] = None,
    ) -> Optional[GeneratedTask]:
        """Generate a single task using LLM.

        Args:
            task_type: 1 for Theory of Mind, 2 for Regular tasks
            task_type_str: Human-readable task type for prompt
            mechanic_bindings: Pre-computed bindings from trajectory exploration
        """
        # Format scene inventory for prompt
        rooms = ", ".join(scene_inventory.get("rooms", [])[:10]) or "unknown"
        furniture = ", ".join(scene_inventory.get("furniture", [])[:15]) or "unknown"
        objects = ", ".join(scene_inventory.get("objects", [])[:10]) or "unknown"
        articulated = ", ".join(scene_inventory.get("articulated_furniture", [])[:10]) or "unknown"

        prompt = TASK_GENERATION_PROMPT.format(
            rooms=rooms,
            furniture=furniture,
            objects=objects,
            articulated=articulated,
            surprises=surprises,
            num_agents=num_agents,
            task_type=task_type_str,
        )

        response = self.llm.generate(prompt)

        # Parse JSON from response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                task_data = json.loads(json_match.group())
                return self._parse_task_response(
                    task_data, episode_id, num_agents, task_type,
                    scene_id=scene_id, mechanics=mechanics,
                    mechanic_bindings=mechanic_bindings,
                )
        except json.JSONDecodeError as e:
            print(f"  Failed to parse LLM response as JSON: {e}")
            print(f"  Response: {response[:500]}...")

        return None

    def _parse_task_response(
        self,
        data: Dict[str, Any],
        episode_id: str,
        num_agents: int,
        task_type: int = 1,
        scene_id: str = "unknown",
        mechanics: List[str] = None,
        mechanic_bindings: Optional[List[MechanicBinding]] = None,
    ) -> GeneratedTask:
        """Parse LLM response into GeneratedTask.

        Args:
            task_type: 1 for Theory of Mind, 2 for Regular tasks
            scene_id: The Habitat scene ID
            mechanics: List of active mechanics for this task
            mechanic_bindings: Pre-computed bindings from trajectory
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Parse subtasks
        subtasks = []
        for st in data.get("subtasks", []):
            subtasks.append(Subtask(
                subtask_id=st.get("subtask_id", f"subtask_{len(subtasks)}"),
                description=st.get("description", ""),
                success_condition=st.get("success_condition", {}),
                assigned_agent=st.get("assigned_agent"),
                depends_on=st.get("depends_on", []),
                hints=st.get("hints", []),
            ))

        # Parse success condition
        sc_data = data.get("success_condition", {})
        success_condition = SuccessCondition(
            description=sc_data.get("description", "Complete the task"),
            required_states=sc_data.get("required_states", []),
            time_limit=sc_data.get("time_limit", 30),
        )

        # Parse failure conditions
        failure_conditions = []
        for fc in data.get("failure_conditions", []):
            failure_conditions.append(FailureCondition(
                description=fc.get("description", "Task failed"),
                failure_states=fc.get("failure_states", []),
                max_failed_attempts=fc.get("max_failed_attempts"),
            ))

        # Default failure condition if none provided
        if not failure_conditions:
            failure_conditions.append(FailureCondition(
                description="Too many failed attempts",
                max_failed_attempts=10,
            ))

        # Set theory_of_mind_required based on task_type
        # task_type 1 = Theory of Mind, task_type 2 = Regular
        is_tom_task = (task_type == 1)

        # Default agent_actions if not provided
        default_agent_actions = {
            "agent_0": ["Navigate", "Open", "Close", "Pick", "Place", "Use", "Inspect", "Communicate"],
            "agent_1": ["Navigate", "Open", "Close", "Pick", "Place", "Communicate"],
        }

        return GeneratedTask(
            task_id=task_id,
            title=data.get("title", "Untitled Task"),
            category=TaskCategory(data.get("category", "coordination")),
            scene_id=scene_id,
            active_mechanics=mechanics or [],
            mechanic_bindings=mechanic_bindings or [],  # From trajectory exploration
            public_goal=data.get("public_goal", ""),
            public_context=data.get("public_context"),
            agent_secrets=data.get("agent_secrets", {}),
            agent_roles=data.get("agent_roles", {}),
            agent_actions=data.get("agent_actions", default_agent_actions),
            success_condition=success_condition,
            failure_conditions=failure_conditions,
            initial_world_state=data.get("initial_world_state", {}),
            num_agents=num_agents,
            difficulty=data.get("difficulty", 3 if is_tom_task else 1),
            theory_of_mind_required=is_tom_task,
            subtasks=subtasks,
            source_trajectory=episode_id,
        )
