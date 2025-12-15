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
    """A collaborative challenge task."""

    task_id: str
    title: str
    category: TaskCategory
    description: str

    initial_world_state: Dict[str, Any]
    required_mechanics: List[str]

    num_agents: int
    agent_roles: Dict[str, str]
    agent_knowledge: Dict[str, List[str]]

    subtasks: List[Subtask]
    success_condition: SuccessCondition
    failure_conditions: List[FailureCondition]

    difficulty: int
    estimated_steps: int
    theory_of_mind_required: bool
    communication_required: bool

    # Optional fields with defaults must come last
    mechanic_bindings: List[MechanicBinding] = field(default_factory=list)
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

        return cls(
            task_id=data["task_id"],
            title=data["title"],
            category=TaskCategory(data["category"]),
            description=data["description"],
            initial_world_state=data["initial_world_state"],
            required_mechanics=data["required_mechanics"],
            mechanic_bindings=bindings,
            num_agents=data["num_agents"],
            agent_roles=data["agent_roles"],
            agent_knowledge=data["agent_knowledge"],
            subtasks=[Subtask.from_dict(s) for s in data["subtasks"]],
            success_condition=SuccessCondition.from_dict(data["success_condition"]),
            failure_conditions=[FailureCondition.from_dict(f) for f in data["failure_conditions"]],
            difficulty=data["difficulty"],
            estimated_steps=data["estimated_steps"],
            theory_of_mind_required=data.get("theory_of_mind_required", False),
            communication_required=data.get("communication_required", False),
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

### This prompt is for THEORY OF MIND TASKS (user selected option 1).
- Design a task for {num_agents} agents working together
- Use ONLY objects/furniture from the Scene Inventory above
- The task should leverage the discovered mechanics (surprising behaviors)
- One agent knows about the mechanics, the other does not (theory of mind)
- Agents must communicate and coordinate to succeed
- Success conditions must reference REAL objects from the inventory

### This prompt is for REGULAR TASKS (user selected option 2).
- Design simple, everyday household tasks for {num_agents} agent(s)
- Use ONLY objects/furniture from the Scene Inventory above
- Tasks should be straightforward actions like:
  * Pick up and move objects (e.g., "pick up the cell phone and place it on the table")
  * Clean or organize items (e.g., "put dirty dishes in the sink")
  * Open/close furniture (e.g., "open the fridge")
  * Navigate and interact (e.g., "go to the bedroom and turn on the lamp")
  * Washing/cleaning (e.g., "wash the clothes", "clean the plate")

## Output Format
Respond with a JSON object:
{{
    "title": "Short descriptive title (max 10 words)",
    "category": "knowledge_asymmetry",
    "description": "2-3 sentence description using ONLY real object names from inventory",
    "initial_world_state": {{
        "objects": ["REAL objects from inventory"],
        "agent_positions": {{"agent_0": "REAL_room_name", "agent_1": "REAL_room_name"}}
    }},
    "required_mechanics": ["mechanic_names from surprises"],
    "agent_roles": {{
        "agent_0": "Expert - knows the discovered mechanics",
        "agent_1": "Novice - does not know the mechanics, must follow instructions"
    }},
    "agent_knowledge": {{
        "agent_0": ["Specific mechanics knowledge - e.g., 'Opening fridge_58 also opens chest_of_drawers_52 (remote_control)'"],
        "agent_1": ["Basic knowledge - object locations only"]
    }},
    "subtasks": [
        {{
            "subtask_id": "step_1",
            "description": "Action using REAL object names only",
            "success_condition": {{"entity": "REAL_object_name", "state": "target_state"}}
        }}
    ],
    "success_condition": {{
        "description": "What success looks like using REAL object names",
        "required_states": [{{"entity": "REAL_object_name", "property": "is_open", "value": true}}]
    }},
    "failure_conditions": [
        {{"description": "What causes failure"}}
    ],
    "difficulty": 3,
    "estimated_steps": 15
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
                )
                if task:
                    tasks.append(task)
                    print(f"OK - {task.title[:40]}...")
                else:
                    print("SKIP (no valid task)")
            except Exception as e:
                print(f"FAIL: {e}")

        return tasks

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
    ) -> Optional[GeneratedTask]:
        """Generate a single task using LLM.

        Args:
            task_type: 1 for Theory of Mind, 2 for Regular tasks
            task_type_str: Human-readable task type for prompt
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
                return self._parse_task_response(task_data, episode_id, num_agents, task_type)
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
    ) -> GeneratedTask:
        """Parse LLM response into GeneratedTask.

        Args:
            task_type: 1 for Theory of Mind, 2 for Regular tasks
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

        return GeneratedTask(
            task_id=task_id,
            title=data.get("title", "Untitled Task"),
            category=TaskCategory(data.get("category", "coordination")),
            description=data.get("description", ""),
            initial_world_state=data.get("initial_world_state", {}),
            required_mechanics=data.get("required_mechanics", []),
            num_agents=num_agents,
            agent_roles=data.get("agent_roles", {}),
            agent_knowledge=data.get("agent_knowledge", {}),
            subtasks=subtasks,
            success_condition=success_condition,
            failure_conditions=failure_conditions,
            difficulty=data.get("difficulty", 3 if is_tom_task else 1),
            estimated_steps=data.get("estimated_steps", 15 if is_tom_task else 5),
            theory_of_mind_required=is_tom_task,
            communication_required=is_tom_task,
            source_trajectory=episode_id,
        )
