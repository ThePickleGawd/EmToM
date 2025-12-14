"""
Task runner for EMTOM benchmark.

Executes collaborative tasks with multiple agents, tracks progress,
and evaluates success/failure conditions.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from emtom.core.mechanic import Mechanic, ActionResult, Effect, create_default_effect
from emtom.core.world_state import TextWorldState, Entity
from emtom.mechanics.registry import MechanicRegistry
from emtom.task_gen.task_generator import GeneratedTask, Subtask


class TaskStatus(Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class AgentAction:
    """An action taken by an agent."""
    agent_id: str
    action: str
    target: Optional[str]
    message: Optional[str] = None  # For communication actions
    reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StepResult:
    """Result of a single step in task execution."""
    step: int
    agent_actions: Dict[str, AgentAction]
    action_results: Dict[str, ActionResult]
    world_state_snapshot: Dict[str, Any]
    subtasks_completed: List[str]
    messages_sent: List[Dict[str, str]]


@dataclass
class TaskResult:
    """Complete result of a task execution."""
    task_id: str
    status: TaskStatus
    total_steps: int
    steps_taken: List[StepResult]
    subtasks_completed: List[str]
    subtasks_failed: List[str]
    final_world_state: Dict[str, Any]
    failure_reason: Optional[str] = None
    time_elapsed_seconds: float = 0.0
    agent_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "total_steps": self.total_steps,
            "subtasks_completed": self.subtasks_completed,
            "subtasks_failed": self.subtasks_failed,
            "failure_reason": self.failure_reason,
            "time_elapsed_seconds": self.time_elapsed_seconds,
            "agent_metrics": self.agent_metrics,
        }


@dataclass
class RunConfig:
    """Configuration for task execution."""
    max_steps: int = 100
    verbose: bool = True
    log_path: Optional[str] = None
    allow_communication: bool = True
    step_delay: float = 0.0  # Delay between steps (for debugging)


class AgentInterface(ABC):
    """Abstract interface for agents that can execute tasks."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.knowledge: List[str] = []
        self.messages_received: List[Dict[str, str]] = []

    @abstractmethod
    def get_action(
        self,
        world_description: str,
        available_actions: List[Dict[str, Any]],
        task_description: str,
        current_subtask: Optional[str],
        other_agents: List[str],
    ) -> AgentAction:
        """
        Decide on an action given the current state.

        Args:
            world_description: Text description of what the agent sees
            available_actions: List of available actions
            task_description: The overall task goal
            current_subtask: Current subtask to focus on
            other_agents: List of other agent IDs for communication

        Returns:
            AgentAction with the chosen action
        """
        pass

    def receive_message(self, sender_id: str, message: str) -> None:
        """Receive a message from another agent."""
        self.messages_received.append({
            "from": sender_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    def set_knowledge(self, knowledge: List[str]) -> None:
        """Set the agent's initial knowledge."""
        self.knowledge = knowledge.copy()

    def reset(self) -> None:
        """Reset agent state for a new task."""
        self.messages_received.clear()


class LLMAgent(AgentInterface):
    """Agent that uses an LLM to decide actions."""

    PROMPT_TEMPLATE = """You are {agent_id} in a collaborative task.

YOUR ROLE: {role}

YOUR KNOWLEDGE:
{knowledge}

TASK GOAL: {task_description}

CURRENT SUBTASK: {current_subtask}

WHAT YOU SEE:
{world_description}

MESSAGES RECEIVED:
{messages}

AVAILABLE ACTIONS:
{available_actions}

OTHER AGENTS YOU CAN COMMUNICATE WITH: {other_agents}

Choose an action. You can either:
1. Perform a physical action (open, close, toggle, press, move, etc.)
2. Send a message to another agent using: communicate[agent_id, "your message"]

Respond in JSON format:
{{
    "action": "<action_name>",
    "target": "<target_or_null>",
    "message_to": "<agent_id_or_null>",
    "message_content": "<message_or_null>",
    "reasoning": "<why_you_chose_this_action>"
}}"""

    def __init__(self, agent_id: str, llm_client: Any, role: str = ""):
        super().__init__(agent_id)
        self.llm = llm_client
        self.role = role

    def get_action(
        self,
        world_description: str,
        available_actions: List[Dict[str, Any]],
        task_description: str,
        current_subtask: Optional[str],
        other_agents: List[str],
    ) -> AgentAction:
        # Format knowledge
        knowledge_str = "\n".join(f"- {k}" for k in self.knowledge) if self.knowledge else "(none)"

        # Format messages
        if self.messages_received:
            messages_str = "\n".join(
                f"- From {m['from']}: {m['message']}"
                for m in self.messages_received[-5:]  # Last 5 messages
            )
        else:
            messages_str = "(no messages)"

        # Format actions
        actions_str = self._format_actions(available_actions)

        prompt = self.PROMPT_TEMPLATE.format(
            agent_id=self.agent_id,
            role=self.role,
            knowledge=knowledge_str,
            task_description=task_description,
            current_subtask=current_subtask or "Complete the main task",
            world_description=world_description,
            messages=messages_str,
            available_actions=actions_str,
            other_agents=", ".join(other_agents) if other_agents else "(none)",
        )

        response = self.llm.generate(prompt)
        return self._parse_response(response)

    def _format_actions(self, actions: List[Dict[str, Any]]) -> str:
        lines = []
        for action in actions:
            name = action.get("name", "unknown")
            targets = action.get("targets", [])
            desc = action.get("description", "")
            if targets:
                lines.append(f"- {name}[<target>]: {desc} (targets: {', '.join(targets)})")
            else:
                lines.append(f"- {name}: {desc}")
        lines.append("- communicate[<agent_id>, \"<message>\"]: Send a message to another agent")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> AgentAction:
        import re
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())

                # Check if it's a communication action
                if data.get("message_to") and data.get("message_content"):
                    return AgentAction(
                        agent_id=self.agent_id,
                        action="communicate",
                        target=data.get("message_to"),
                        message=data.get("message_content"),
                        reasoning=data.get("reasoning", ""),
                    )

                return AgentAction(
                    agent_id=self.agent_id,
                    action=data.get("action", "wait"),
                    target=data.get("target"),
                    reasoning=data.get("reasoning", ""),
                )
        except json.JSONDecodeError:
            pass

        return AgentAction(
            agent_id=self.agent_id,
            action="wait",
            target=None,
            reasoning="Could not parse response",
        )


class ScriptedAgent(AgentInterface):
    """Agent that follows a predefined script (for testing)."""

    def __init__(self, agent_id: str, script: List[Dict[str, Any]]):
        super().__init__(agent_id)
        self.script = script
        self._index = 0

    def get_action(
        self,
        world_description: str,
        available_actions: List[Dict[str, Any]],
        task_description: str,
        current_subtask: Optional[str],
        other_agents: List[str],
    ) -> AgentAction:
        if self._index >= len(self.script):
            return AgentAction(
                agent_id=self.agent_id,
                action="wait",
                target=None,
                reasoning="Script complete",
            )

        action_data = self.script[self._index]
        self._index += 1

        return AgentAction(
            agent_id=self.agent_id,
            action=action_data.get("action", "wait"),
            target=action_data.get("target"),
            message=action_data.get("message"),
            reasoning="Scripted action",
        )

    def reset(self) -> None:
        super().reset()
        self._index = 0


class TaskRunner:
    """
    Executes EMTOM benchmark tasks with multiple agents.

    Manages world state, applies mechanics, handles agent actions,
    checks success/failure conditions, and tracks metrics.
    """

    def __init__(
        self,
        config: Optional[RunConfig] = None,
    ):
        self.config = config or RunConfig()
        self.world: Optional[TextWorldState] = None
        self.mechanics: List[Mechanic] = []
        self.agents: Dict[str, AgentInterface] = {}
        self.current_task: Optional[GeneratedTask] = None

        # Tracking
        self._step_count = 0
        self._completed_subtasks: List[str] = []
        self._failed_attempts: int = 0
        self._messages: List[Dict[str, str]] = []

    def setup_task(
        self,
        task: GeneratedTask,
        agents: Dict[str, AgentInterface],
    ) -> None:
        """
        Set up a task for execution.

        Args:
            task: The task to execute
            agents: Dict mapping agent_id to AgentInterface
        """
        self.current_task = task
        self.agents = agents

        # Load required mechanics
        self.mechanics = []
        for mechanic_name in task.required_mechanics:
            try:
                # Try exact name first
                mechanic = MechanicRegistry.instantiate(mechanic_name)
                self.mechanics.append(mechanic)
            except KeyError:
                # Try with common suffixes
                for suffix in ["_open", "_switch", "_trigger", "_effect"]:
                    try:
                        mechanic = MechanicRegistry.instantiate(mechanic_name + suffix)
                        self.mechanics.append(mechanic)
                        break
                    except KeyError:
                        continue

        # Initialize world from task's initial state
        self.world = self._create_world_from_task(task)

        # Set up agents with their roles and knowledge
        for agent_id, agent in agents.items():
            agent.reset()
            if agent_id in task.agent_knowledge:
                agent.set_knowledge(task.agent_knowledge[agent_id])
            if hasattr(agent, 'role') and agent_id in task.agent_roles:
                agent.role = task.agent_roles[agent_id]

        # Reset tracking
        self._step_count = 0
        self._completed_subtasks = []
        self._failed_attempts = 0
        self._messages = []

    def _create_world_from_task(self, task: GeneratedTask) -> TextWorldState:
        """Create world state from task's initial_world_state."""
        world = TextWorldState()
        initial = task.initial_world_state

        # Add rooms
        rooms = set()
        if "rooms" in initial:
            for room_data in initial["rooms"]:
                room_id = room_data["id"]
                rooms.add(room_id)
                world.add_entity(Entity(
                    id=room_id,
                    entity_type="room",
                    properties=room_data,
                ))

        # Extract rooms from agent positions
        if "agent_positions" in initial:
            for pos in initial["agent_positions"].values():
                if pos not in rooms:
                    rooms.add(pos)
                    world.add_entity(Entity(
                        id=pos,
                        entity_type="room",
                        properties={"name": pos.replace("_", " ").title()},
                    ))

        # Add agents
        if "agent_positions" in initial:
            for agent_id, location in initial["agent_positions"].items():
                world.add_entity(Entity(
                    id=agent_id,
                    entity_type="agent",
                    properties={"name": agent_id},
                    location=location,
                ))

        # Add doors
        if "doors" in initial:
            for door_data in initial["doors"]:
                world.add_entity(Entity(
                    id=door_data["id"],
                    entity_type="door",
                    properties={"is_open": door_data.get("is_open", False)},
                    location=door_data.get("location"),
                ))

        # Add switches
        if "switches" in initial:
            for switch_data in initial["switches"]:
                world.add_entity(Entity(
                    id=switch_data["id"],
                    entity_type="switch",
                    properties={
                        "is_on": switch_data.get("is_on", False),
                        "controls": switch_data.get("controls"),
                    },
                    location=switch_data.get("location"),
                ))

        # Add lights
        if "lights" in initial:
            for light_data in initial["lights"]:
                world.add_entity(Entity(
                    id=light_data["id"],
                    entity_type="light",
                    properties={"is_on": light_data.get("is_on", False)},
                    location=light_data.get("location"),
                ))

        # Add buttons
        if "buttons" in initial:
            for button_data in initial["buttons"]:
                world.add_entity(Entity(
                    id=button_data["id"],
                    entity_type="button",
                    properties={
                        "is_active": button_data.get("is_active", False),
                        "press_count": button_data.get("press_count", 0),
                        "required_presses": button_data.get("required_presses", 1),
                    },
                    location=button_data.get("location"),
                ))

        # Add any other objects from initial state
        if "target_object" in initial:
            obj_id = initial["target_object"]
            if not world.get_entity(obj_id):
                # Find location from agent positions
                locations = list(initial.get("agent_positions", {}).values())
                location = locations[1] if len(locations) > 1 else (locations[0] if locations else None)
                world.add_entity(Entity(
                    id=obj_id,
                    entity_type="device",
                    properties={"is_active": False},
                    location=location,
                ))

        return world

    def run(self) -> TaskResult:
        """
        Execute the task and return results.

        Returns:
            TaskResult with success/failure status and metrics
        """
        if not self.current_task or not self.world:
            raise RuntimeError("Must call setup_task before run")

        start_time = time.time()
        task = self.current_task
        steps_taken: List[StepResult] = []

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING TASK: {task.title}")
            print(f"{'='*60}")
            print(f"Agents: {list(self.agents.keys())}")
            print(f"Mechanics: {task.required_mechanics}")
            print(f"Goal: {task.success_condition.description}")
            print()

        # Main execution loop
        while self._step_count < self.config.max_steps:
            # Check time limit
            if task.success_condition.time_limit:
                if self._step_count >= task.success_condition.time_limit:
                    return self._create_result(
                        TaskStatus.TIMEOUT,
                        steps_taken,
                        time.time() - start_time,
                        f"Exceeded time limit of {task.success_condition.time_limit} steps",
                    )

            # Execute one step
            step_result = self._execute_step()
            steps_taken.append(step_result)

            # Check success condition
            if self._check_success():
                if self.config.verbose:
                    print(f"\n[SUCCESS] Task completed in {self._step_count} steps!")
                return self._create_result(
                    TaskStatus.SUCCESS,
                    steps_taken,
                    time.time() - start_time,
                )

            # Check failure conditions
            failure_reason = self._check_failure()
            if failure_reason:
                if self.config.verbose:
                    print(f"\n[FAILURE] {failure_reason}")
                return self._create_result(
                    TaskStatus.FAILURE,
                    steps_taken,
                    time.time() - start_time,
                    failure_reason,
                )

            self._step_count += 1

            if self.config.step_delay > 0:
                time.sleep(self.config.step_delay)

        # Max steps exceeded
        return self._create_result(
            TaskStatus.TIMEOUT,
            steps_taken,
            time.time() - start_time,
            f"Exceeded maximum steps ({self.config.max_steps})",
        )

    def _execute_step(self) -> StepResult:
        """Execute a single step where all agents act."""
        agent_actions: Dict[str, AgentAction] = {}
        action_results: Dict[str, ActionResult] = {}
        messages_this_step: List[Dict[str, str]] = []

        if self.config.verbose:
            print(f"\n--- Step {self._step_count} ---")

        # Get current subtask for agents
        current_subtask = self._get_current_subtask()

        # Each agent takes an action
        for agent_id, agent in self.agents.items():
            # Get world description for this agent
            world_desc = self.world.to_text(agent_id)

            # Get available actions
            available_actions = self._get_available_actions(agent_id)

            # Get other agent IDs for communication
            other_agents = [aid for aid in self.agents.keys() if aid != agent_id]

            # Agent decides on action
            action = agent.get_action(
                world_description=world_desc,
                available_actions=available_actions,
                task_description=self.current_task.description,
                current_subtask=current_subtask,
                other_agents=other_agents,
            )
            agent_actions[agent_id] = action

            if self.config.verbose:
                if action.action == "communicate":
                    print(f"  {agent_id}: communicate to {action.target}: \"{action.message}\"")
                else:
                    target_str = f" on {action.target}" if action.target else ""
                    print(f"  {agent_id}: {action.action}{target_str}")

            # Execute the action
            if action.action == "communicate" and self.config.allow_communication:
                result = self._handle_communication(agent_id, action)
                messages_this_step.append({
                    "from": agent_id,
                    "to": action.target,
                    "message": action.message,
                })
            else:
                result = self._execute_action(agent_id, action)

            action_results[agent_id] = result

            if self.config.verbose and result.observations.get(agent_id):
                obs = result.observations[agent_id]
                print(f"    -> {obs[:100]}{'...' if len(obs) > 100 else ''}")

        # Process delayed effects
        ready_effects = self.world.advance_step()
        for effect in ready_effects:
            self.world.apply_effect(effect)

        # Check subtask completion
        newly_completed = self._check_subtasks()

        return StepResult(
            step=self._step_count,
            agent_actions=agent_actions,
            action_results=action_results,
            world_state_snapshot=self.world.snapshot(),
            subtasks_completed=newly_completed,
            messages_sent=messages_this_step,
        )

    def _execute_action(self, agent_id: str, action: AgentAction) -> ActionResult:
        """Execute an agent's physical action."""
        action_name = action.action
        target = action.target

        # Handle special actions
        if action_name == "wait":
            return ActionResult(
                success=True,
                observations={agent_id: "You wait and observe."},
            )

        if action_name == "move" and target:
            return self._handle_move(agent_id, target)

        if action_name == "look":
            description = self.world.to_text(agent_id)
            return ActionResult(
                success=True,
                observations={agent_id: description},
            )

        # Check target exists
        if target and not self.world.get_entity(target):
            self._failed_attempts += 1
            return ActionResult(
                success=False,
                error_message=f"Target '{target}' not found",
                observations={agent_id: f"You don't see any '{target}' here."},
            )

        # Apply mechanics
        for mechanic in self.mechanics:
            if mechanic.applies_to(action_name, target, self.world):
                intended = create_default_effect(action_name, target, self.world)
                result = mechanic.transform_effect(
                    action_name, agent_id, target, intended, self.world
                )
                for effect in result.effects:
                    self.world.apply_effect(effect)
                for effect in result.pending_effects:
                    self.world.add_pending_effect(effect)

                # Track failed attempts (when action doesn't have expected effect)
                if result.surprise_triggers.get(agent_id):
                    self._failed_attempts += 1

                return result

        # Default action handling
        return self._default_action(agent_id, action_name, target)

    def _default_action(
        self, agent_id: str, action_name: str, target: Optional[str]
    ) -> ActionResult:
        """Handle action with default behavior."""
        if not target:
            return ActionResult(
                success=False,
                observations={agent_id: f"You need to specify a target for {action_name}."},
            )

        effect = create_default_effect(action_name, target, self.world)
        self.world.apply_effect(effect)

        return ActionResult(
            success=True,
            effects=[effect],
            observations={agent_id: effect.description or f"You {action_name} the {target}."},
        )

    def _handle_move(self, agent_id: str, destination: str) -> ActionResult:
        """Handle agent movement."""
        rooms = self.world.get_room_ids()

        if destination not in rooms:
            return ActionResult(
                success=False,
                observations={agent_id: f"You don't know how to get to '{destination}'."},
            )

        old_location = self.world.get_agent_location(agent_id)
        self.world.move_entity(agent_id, destination)

        return ActionResult(
            success=True,
            effects=[Effect(
                target=agent_id,
                property_changed="location",
                old_value=old_location,
                new_value=destination,
                visible_to={agent_id},
            )],
            observations={agent_id: f"You move to {destination}.\n{self.world.to_text(agent_id)}"},
        )

    def _handle_communication(self, sender_id: str, action: AgentAction) -> ActionResult:
        """Handle communication between agents."""
        receiver_id = action.target
        message = action.message

        if receiver_id not in self.agents:
            return ActionResult(
                success=False,
                observations={sender_id: f"Cannot find agent '{receiver_id}'."},
            )

        # Deliver message
        self.agents[receiver_id].receive_message(sender_id, message)
        self._messages.append({
            "from": sender_id,
            "to": receiver_id,
            "message": message,
            "step": self._step_count,
        })

        return ActionResult(
            success=True,
            observations={
                sender_id: f"You sent a message to {receiver_id}.",
                receiver_id: f"Message from {sender_id}: {message}",
            },
        )

    def _get_available_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get available actions for an agent."""
        actions = [
            {"name": "wait", "description": "Do nothing", "targets": []},
            {"name": "look", "description": "Look around", "targets": []},
        ]

        # Movement
        rooms = self.world.get_room_ids()
        current_room = self.world.get_agent_location(agent_id)
        other_rooms = [r for r in rooms if r != current_room]
        if other_rooms:
            actions.append({
                "name": "move",
                "description": "Move to another room",
                "targets": other_rooms,
            })

        # Interact with objects in current room
        if current_room:
            for entity in self.world.get_entities_in_location(current_room):
                if entity.id == agent_id:
                    continue

                if entity.entity_type == "door":
                    is_open = entity.get_property("is_open", False)
                    actions.append({
                        "name": "close" if is_open else "open",
                        "description": f"{'Close' if is_open else 'Open'} the door",
                        "targets": [entity.id],
                    })

                elif entity.entity_type == "switch":
                    actions.append({
                        "name": "toggle",
                        "description": "Flip the switch",
                        "targets": [entity.id],
                    })

                elif entity.entity_type == "button":
                    actions.append({
                        "name": "press",
                        "description": "Press the button",
                        "targets": [entity.id],
                    })

                elif entity.entity_type == "device":
                    actions.append({
                        "name": "activate",
                        "description": "Activate the device",
                        "targets": [entity.id],
                    })

                elif entity.entity_type == "light":
                    is_on = entity.get_property("is_on", False)
                    actions.append({
                        "name": "turn_off" if is_on else "turn_on",
                        "description": f"Turn {'off' if is_on else 'on'} the light",
                        "targets": [entity.id],
                    })

        return actions

    def _get_current_subtask(self) -> Optional[str]:
        """Get the current subtask that should be worked on."""
        task = self.current_task
        for subtask in task.subtasks:
            if subtask.subtask_id not in self._completed_subtasks:
                # Check dependencies
                deps_met = all(
                    dep in self._completed_subtasks
                    for dep in subtask.depends_on
                )
                if deps_met:
                    return subtask.description
        return None

    def _check_subtasks(self) -> List[str]:
        """Check which subtasks are newly completed."""
        newly_completed = []
        for subtask in self.current_task.subtasks:
            if subtask.subtask_id in self._completed_subtasks:
                continue

            if self._is_subtask_complete(subtask):
                self._completed_subtasks.append(subtask.subtask_id)
                newly_completed.append(subtask.subtask_id)
                if self.config.verbose:
                    print(f"  [SUBTASK COMPLETE] {subtask.description}")

        return newly_completed

    def _is_subtask_complete(self, subtask: Subtask) -> bool:
        """Check if a subtask's success condition is met."""
        condition = subtask.success_condition

        if "entity" in condition and "property" in condition:
            entity = self.world.get_entity(condition["entity"])
            if entity:
                actual = entity.get_property(condition["property"])
                return actual == condition["value"]

        # Handle special conditions
        if "communication_established" in condition:
            return len(self._messages) > 0

        if "knowledge_transferred" in condition:
            # Check if messages contain relevant information
            return len(self._messages) >= 2

        if "all_agents_in" in condition:
            target_room = condition["all_agents_in"]
            return all(
                self.world.get_agent_location(aid) == target_room
                for aid in self.agents.keys()
            )

        return False

    def _check_success(self) -> bool:
        """Check if the task's success condition is met."""
        condition = self.current_task.success_condition

        for req in condition.required_states:
            if "entity" in req and "property" in req:
                entity = self.world.get_entity(req["entity"])
                if not entity:
                    return False
                actual = entity.get_property(req["property"])
                if actual != req["value"]:
                    return False

        return True

    def _check_failure(self) -> Optional[str]:
        """Check if any failure condition is met."""
        for fc in self.current_task.failure_conditions:
            # Check max failed attempts
            if fc.max_failed_attempts is not None:
                if self._failed_attempts >= fc.max_failed_attempts:
                    return fc.description

            # Check failure states
            for state in fc.failure_states:
                if "entity" in state and "property" in state:
                    entity = self.world.get_entity(state["entity"])
                    if entity:
                        actual = entity.get_property(state["property"])
                        if actual == state["value"]:
                            return fc.description

        return None

    def _create_result(
        self,
        status: TaskStatus,
        steps: List[StepResult],
        elapsed: float,
        failure_reason: Optional[str] = None,
    ) -> TaskResult:
        """Create the final task result."""
        # Compute per-agent metrics
        agent_metrics = {}
        for agent_id in self.agents.keys():
            actions_taken = sum(
                1 for s in steps
                if agent_id in s.agent_actions
            )
            messages_sent = sum(
                1 for m in self._messages
                if m["from"] == agent_id
            )
            agent_metrics[agent_id] = {
                "actions_taken": actions_taken,
                "messages_sent": messages_sent,
            }

        failed_subtasks = [
            st.subtask_id for st in self.current_task.subtasks
            if st.subtask_id not in self._completed_subtasks
        ]

        return TaskResult(
            task_id=self.current_task.task_id,
            status=status,
            total_steps=self._step_count,
            steps_taken=steps,
            subtasks_completed=self._completed_subtasks.copy(),
            subtasks_failed=failed_subtasks,
            final_world_state=self.world.snapshot(),
            failure_reason=failure_reason,
            time_elapsed_seconds=elapsed,
            agent_metrics=agent_metrics,
        )
