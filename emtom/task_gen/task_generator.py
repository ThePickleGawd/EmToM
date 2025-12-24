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
    """
    Load scraped scenario briefings to use as creative inspiration.

    Args:
        directory: Path to directory containing .txt scenario files
        max_scenarios: Maximum number of scenarios to include

    Returns:
        List of scenario texts (randomly sampled if more than max)
    """
    path = Path(directory)
    scenarios = []

    if not path.exists():
        return []

    txt_files = list(path.glob("*.txt"))
    if not txt_files:
        return []

    # Randomly sample if we have more than max
    if len(txt_files) > max_scenarios:
        txt_files = random.sample(txt_files, max_scenarios)

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


TASK_GENERATION_PROMPT = '''You are a creative puzzle designer for an escape-room style multi-agent collaboration game.

## YOUR MISSION
Create an engaging, atmospheric MULTI-STEP puzzle scenario inspired by the escape room themes below.
The puzzle should feel like a mystery to solve with 3-5 steps of progression, not a single action.

## Creative Inspiration (IMPORTANT - base your scenario on these themes!)
{scenario_inspirations}

## Scene Inventory (you MUST only use these objects)
Rooms: {rooms}
Furniture (can navigate to, some can be opened): {furniture}
Objects (can be picked up, hidden, inspected): {objects}
Articulated Furniture (can be opened/closed): {articulated}

## Available Items (use these item_ids in hidden_items and locked_containers)
{available_items}

## Discovered Mechanics (hidden connections between objects)
{surprises}

## CRITICAL CONSTRAINTS
- ONLY use objects/furniture from the Scene Inventory above
- ONLY use items from the Available Items list above (use exact item_id like "small_key", "big_key", "radio")
- Instance items by appending _N suffix (e.g., "small_key_1", "small_key_2")
- Adapt the escape room themes to work with the REAL objects and items listed
- Create MULTI-STEP puzzles (3-5 steps) where each step unlocks the next

## MULTI-STEP PUZZLE DESIGN
Design puzzles where progress is GATED by previous steps:
- Step 1: Search a drawer to find a key
- Step 2: UseItem[key, cabinet] to unlock
- Step 3: Search the cabinet to find another key
- Step 4: UseItem[key, chest] to unlock
- Step 5: Retrieve the goal item from the chest

Use these mechanics to create chains:
- "hidden_items": Items hidden inside furniture (found via Search action)
- "is_locked": Locked containers (use UseItem[key, container] to unlock)
- Locked containers block Search until unlocked
- Keys are consumed when used

**CRITICAL: Items are ABSTRACT (not in world graph)**
- Items go directly to inventory when found
- You CANNOT Pick, Navigate to, or physically interact with items
- Use `UseItem[item_id, args]` to use items from inventory

## TOOL Items with Room Restrictions (for Theory of Mind)
TOOL items (like radio) can have "allowed_rooms" to restrict where they work:
- If allowed_rooms is set, the tool ONLY works in those specific rooms
- This creates Theory of Mind challenges: agents must know WHERE to use tools
- Example: A radio that only works in the living room requires navigation + knowledge

To use room restrictions, add "items" array with allowed_rooms:
```
"items": [
  {{"item_id": "radio_1", "allowed_rooms": ["living_room_1", "office_1"]}}
]
```

## Task Type: {task_type}

### THEORY OF MIND TASKS:
- {num_agents} agents work together on a puzzle
- agent_0 has discovered the hidden mechanics through exploration
- agent_1 does NOT know about the mechanics (creates theory of mind challenge)
- Agents must communicate and reason about each other's knowledge
- The narrative should make the asymmetric knowledge feel natural
- Use room-restricted TOOL items to test spatial reasoning (e.g., radio only works in certain rooms)

### REGULAR TASKS:
- Simple collaborative tasks for {num_agents} agent(s)
- Use scene objects for everyday actions

## Output Format
Create an immersive MULTI-STEP puzzle scenario. The "story" field is REQUIRED.

{{
    "title": "Evocative puzzle title (e.g., 'The Mirrored Cabinet', 'Echoes in the Kitchen')",
    "story": "2-3 sentences of atmospheric narrative setting up the puzzle. Draw from the escape room inspiration above.",
    "public_goal": "Clear objective using REAL object names (e.g., 'Find the hidden artifact and place it on the table')",
    "public_context": "What both agents know about the situation (no secrets here)",
    "agent_secrets": {{
        "agent_0": ["The hidden knowledge agent_0 has discovered (mechanics, connections, where keys are hidden)"],
        "agent_1": []
    }},
    "agent_roles": {{
        "agent_0": "The one who knows the house's secrets",
        "agent_1": "The one who must follow strange instructions on faith"
    }},
    "agent_actions": {{
        "agent_0": ["Navigate", "Open", "Close", "Pick", "Place", "UseItem", "Inspect", "Search", "Communicate"],
        "agent_1": ["Navigate", "Open", "Close", "Pick", "Place", "Communicate"]
    }},
    "subtasks": [
        "Step 1: Search drawer_1 to find small_key_1",
        "Step 2: UseItem[small_key_1, cabinet_2] to unlock",
        "Step 3: Search cabinet_2 to find small_key_2",
        "Step 4: UseItem[small_key_2, chest_3] to unlock",
        "Step 5: Retrieve goal_item from chest_3 and place on table"
    ],
    "success_condition": {{
        "description": "What success looks like",
        "required_states": [
            {{"entity": "REAL_object", "property": "location", "value": "REAL_target"}},
            {{"entity": "agent_0", "property": "has_item", "value": "big_key_1"}}
        ]
    }},
    "category": "knowledge_asymmetry"
}}

Generate an engaging MULTI-STEP puzzle scenario with 3-5 chained steps (remember: use the escape room inspiration, but ONLY reference real objects from the inventory):'''


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
        dataset_episode_id = trajectory.get("metadata", {}).get("episode_id", "1")
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
                    dataset_episode_id=dataset_episode_id,
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

    def _format_scenario_inspirations(self, scenarios: List[str]) -> str:
        """Format scraped scenarios as creative inspiration."""
        if not scenarios:
            return "(No scenario inspirations available)"

        lines = []
        for i, scenario in enumerate(scenarios, 1):
            lines.append(f"--- Inspiration {i} ---")
            lines.append(scenario)
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
        dataset_episode_id: str,
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

        # Load scenario inspirations from scraped content
        scenarios = load_scenario_inspirations(max_scenarios=10)
        scenario_text = self._format_scenario_inspirations(scenarios)

        # Get available items from registry (dynamically loaded)
        from emtom.state.item_registry import ItemRegistry
        available_items = ItemRegistry.get_items_for_task_generation()

        prompt = TASK_GENERATION_PROMPT.format(
            rooms=rooms,
            furniture=furniture,
            objects=objects,
            articulated=articulated,
            available_items=available_items,
            surprises=surprises,
            scenario_inspirations=scenario_text,
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
                    task_data, episode_id, dataset_episode_id, num_agents, task_type,
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
        dataset_episode_id: str,
        num_agents: int,
        task_type: int = 1,
        scene_id: str = "unknown",
        mechanics: List[str] = None,
        mechanic_bindings: Optional[List[MechanicBinding]] = None,
    ) -> GeneratedTask:
        """Parse LLM response into GeneratedTask.

        Args:
            episode_id: Trajectory timestamp (e.g., "20251219_194151")
            dataset_episode_id: Habitat dataset episode ID (e.g., "1") for loading
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

        # Set theory_of_mind_required based on task_type
        # task_type 1 = Theory of Mind, task_type 2 = Regular
        is_tom_task = (task_type == 1)

        # Default agent_actions if not provided
        default_agent_actions = {
            "agent_0": ["Navigate", "Open", "Close", "Pick", "Place", "UseItem", "Inspect", "Communicate"],
            "agent_1": ["Navigate", "Open", "Close", "Pick", "Place", "Communicate"],
        }

        return GeneratedTask(
            task_id=task_id,
            title=data.get("title", "Untitled Task"),
            category=TaskCategory(data.get("category", "coordination")),
            scene_id=scene_id,
            episode_id=episode_id,
            dataset_episode_id=dataset_episode_id,
            active_mechanics=mechanics or [],
            mechanic_bindings=mechanic_bindings or [],  # From trajectory exploration
            story=data.get("story"),
            public_goal=data.get("public_goal", ""),
            public_context=data.get("public_context"),
            agent_secrets=data.get("agent_secrets", {}),
            agent_roles=data.get("agent_roles", {}),
            agent_actions=data.get("agent_actions", default_agent_actions),
            success_condition=success_condition,
            num_agents=num_agents,
            theory_of_mind_required=is_tom_task,
            subtasks=subtasks,
        )
