"""
EMTOM Game State Manager.

Handles:
- Syncing state from Habitat simulator
- Applying actions through mechanics (stateless transforms)
- Ticking time-based effects
- Setting up initial state from task definition
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import copy
import uuid

from emtom.state.game_state import (
    EMTOMGameState,
    SpawnedItem,
    PendingEffect,
    ActionRecord,
    Goal,
    GoalStatus,
)
from emtom.mechanics.handlers import (
    apply_mechanics,
    HandlerResult,
    MECHANIC_INFO,
)

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


@dataclass
class ActionExecutionResult:
    """Result of executing an action."""
    state: EMTOMGameState
    observation: str
    success: bool
    effects: List[str]
    surprise_trigger: Optional[str] = None
    spawned_items: List[str] = None

    def __post_init__(self):
        if self.spawned_items is None:
            self.spawned_items = []


class GameStateManager:
    """
    Manages EMTOM game state.

    Main interface for:
    - Initializing state from task definition
    - Syncing state from Habitat each step
    - Applying actions (with mechanic transforms)
    - Ticking time-based effects
    - Checking goal completion
    """

    def __init__(self, env_interface: Optional["EnvironmentInterface"] = None):
        """
        Initialize the game state manager.

        Args:
            env_interface: Habitat environment interface for syncing state.
                          Can be None for testing.
        """
        self.env = env_interface
        self.state = EMTOMGameState()

    def initialize_from_task(self, task_data: Dict[str, Any]) -> EMTOMGameState:
        """
        Initialize game state from a task definition.

        Args:
            task_data: Task definition dict containing:
                - mechanics: List of mechanic bindings
                - hidden_items: Objects with hidden contents
                - goals: Task goals

        Returns:
            Initialized game state
        """
        state = EMTOMGameState()

        # Set active mechanics from bindings
        mechanics = task_data.get("mechanics", [])
        state.active_mechanics = []
        state.mechanic_bindings = mechanics

        for binding in mechanics:
            if isinstance(binding, str):
                mech_type = binding
            else:
                mech_type = binding.get("mechanic_type", binding.get("type"))

            if mech_type and mech_type not in state.active_mechanics:
                state.active_mechanics.append(mech_type)

            # Set up mechanic-specific state from bindings
            if isinstance(binding, dict):
                self._setup_mechanic_state(state, mech_type, binding)

        # Set up hidden items (for Shake action)
        hidden_items = task_data.get("hidden_items", {})
        for obj_id, hidden_info in hidden_items.items():
            contains = hidden_info.get("contains") if isinstance(hidden_info, dict) else hidden_info
            state = state.set_object_property(obj_id, "hidden_inside", contains)

        # Set up goals
        goals_data = task_data.get("goals", [])
        for g in goals_data:
            goal = Goal(
                goal_id=g.get("id", str(uuid.uuid4())),
                description=g.get("description", ""),
                goal_type=g.get("type", "unknown"),
                target=g.get("target"),
                target_state=g.get("target_state"),
                status=GoalStatus.PENDING,
            )
            state.goals.append(goal)

        self.state = state
        return state

    def _setup_mechanic_state(
        self, state: EMTOMGameState, mech_type: str, binding: Dict[str, Any]
    ) -> None:
        """Set up state for a specific mechanic binding."""
        trigger = binding.get("trigger_object")
        target_obj = binding.get("target_object")
        target_state = binding.get("target_state", "is_open")

        if mech_type == "inverse_state":
            if trigger:
                state.inverse_objects.add(trigger)

        elif mech_type == "remote_control":
            if trigger and target_obj:
                state.remote_mappings[trigger] = (target_obj, target_state)

        elif mech_type == "state_mirroring":
            if trigger and target_obj:
                state.mirror_pairs.append((trigger, target_obj, target_state))

        elif mech_type == "counting_state":
            if trigger:
                state.interaction_counts[trigger] = 0
                count = binding.get("required_count", binding.get("count", 3))
                state.object_properties.setdefault(trigger, {})["required_count"] = count

        elif mech_type == "delayed_effect":
            if trigger:
                delay = binding.get("delay_steps", 3)
                state.object_properties.setdefault(trigger, {})["delay_steps"] = delay

        elif mech_type == "decaying_state":
            if trigger:
                decay = binding.get("decay_steps", 3)
                state.object_properties.setdefault(trigger, {})["decay_steps"] = decay

        elif mech_type == "conditional_unlock":
            if trigger:
                prereq = binding.get("prerequisite_object")
                if prereq:
                    state.object_properties.setdefault(trigger, {})["prerequisite"] = prereq
                    state.object_properties.setdefault(prereq, {})["unlocks"] = trigger

        elif mech_type == "sequence_lock":
            if trigger:
                state.sequence_progress[trigger] = 0
                sequence = binding.get("sequence", [])
                state.object_properties.setdefault(trigger, {})["sequence_length"] = len(sequence)
                # Set up sequence steps
                for step in sequence:
                    step_obj = step.get("object") if isinstance(step, dict) else step
                    if step_obj:
                        state.object_properties.setdefault(step_obj, {})["advances_sequence"] = trigger

    def sync_from_habitat(self, state: Optional[EMTOMGameState] = None) -> EMTOMGameState:
        """
        Sync state from Habitat simulator.

        Args:
            state: State to update. If None, uses self.state.

        Returns:
            Updated game state
        """
        if state is None:
            state = self.state

        if self.env is None:
            return state

        try:
            world_graph = self.env.world_graph
        except AttributeError:
            return state

        # Sync agent positions and rooms
        agent_positions = {}
        agent_rooms = {}

        for agent in world_graph.get("agents", []):
            agent_id = agent.get("id") or agent.get("name")
            if agent_id:
                pos = agent.get("position")
                if pos:
                    agent_positions[agent_id] = tuple(pos) if isinstance(pos, list) else pos
                room = agent.get("room") or agent.get("location")
                if room:
                    agent_rooms[agent_id] = room

        # Sync object states
        object_states = {}
        entities = []

        for entity in world_graph.get("entities", []):
            entity_id = entity.get("id") or entity.get("name")
            entities.append(entity)

            if entity_id:
                states = {}
                for key, value in entity.items():
                    if key.startswith("is_"):
                        states[key] = value
                if states:
                    object_states[entity_id] = states

        new_state = copy.copy(state)
        new_state.agent_positions = agent_positions
        new_state.agent_rooms = agent_rooms
        new_state.object_states = object_states
        new_state.entities = entities

        self.state = new_state
        return new_state

    def spawn_key_on_table(
        self, state: Optional[EMTOMGameState] = None
    ) -> Tuple[EMTOMGameState, Optional[Dict[str, Any]]]:
        """
        Spawn a key object on a random table in the scene.

        The key will only spawn on furniture that has 'table' in its name.

        Args:
            state: State to update. If None, uses self.state.

        Returns:
            (new_state, spawn_info) where spawn_info contains key location,
            or None if no tables found
        """
        import random
        import time

        if state is None:
            state = self.state

        entities = getattr(state, 'entities', [])
        if not entities:
            return state, None

        # Find all tables (furniture with 'table' in name)
        tables = []
        for e in entities:
            name = e.get("name", e.get("id", ""))
            e_type = e.get("type", "")
            if e_type == "furniture" and "table" in name.lower():
                tables.append(name)

        if not tables:
            return state, None

        # Use a time-based seed for truly random table selection
        # (avoids being affected by global simulation seed)
        rng = random.Random(int(time.time() * 1000000))
        rng.shuffle(tables)
        chosen_table = tables[0]

        # Create key spawn info
        key_id = "exploration_key"
        spawn_info = {
            "key_id": key_id,
            "location": chosen_table,
            "spawned_at_step": state.current_step,
        }

        # Add key to world_objects (objects spawned on furniture)
        new_state = copy.copy(state)
        new_state.world_objects = copy.copy(state.world_objects)
        new_state.world_objects[key_id] = {
            "type": "object",
            "name": "key",
            "location": chosen_table,
            "pickable": True,
        }

        # Also add to entities so it shows up in world description
        new_entities = list(state.entities)
        new_entities.append({
            "id": key_id,
            "name": "key",
            "type": "object",
            "location": chosen_table,
            "is_on_table": True,
        })
        new_state.entities = new_entities

        self.state = new_state
        return new_state, spawn_info

    def auto_bind_mechanics(
        self, state: Optional[EMTOMGameState] = None
    ) -> Tuple[EMTOMGameState, Dict[str, Any]]:
        """
        Auto-bind mechanics to random objects in the scene.

        Call this after sync_from_habitat() to bind mechanics to real objects.

        Returns:
            (new_state, bindings_info) where bindings_info shows what was bound
        """
        import random

        if state is None:
            state = self.state

        bindings_info = {}

        # Get entities from state
        entities = getattr(state, 'entities', [])
        if not entities:
            return state, {"error": "No entities found - call sync_from_habitat first"}

        # Categorize entities
        articulated = []  # Furniture that can open/close (doors, drawers, cabinets)
        furniture = []    # All furniture
        objects = []      # Small objects
        shakeable = []    # Things you could shake (furniture + objects)

        for e in entities:
            name = e.get("name", e.get("id", ""))
            e_type = e.get("type", "")
            is_art = e.get("is_articulated", False)

            if is_art or any(k in name.lower() for k in ["door", "drawer", "cabinet", "fridge"]):
                articulated.append(name)
            if e_type == "furniture":
                furniture.append(name)
                shakeable.append(name)  # Can shake furniture
            elif e_type == "object":
                objects.append(name)
                shakeable.append(name)  # Can shake objects

        # Shuffle for random selection
        random.shuffle(articulated)
        random.shuffle(furniture)
        random.shuffle(objects)
        random.shuffle(shakeable)

        new_state = copy.copy(state)

        # Bind each active mechanic
        for mech_type in state.active_mechanics:
            if mech_type == "inverse_state":
                # Bind to an articulated object
                if articulated:
                    target = articulated.pop(0)
                    new_state.inverse_objects.add(target)
                    bindings_info["inverse_state"] = {"target": target}

            elif mech_type == "remote_control":
                # Bind trigger -> target pair
                if len(articulated) >= 2:
                    trigger = articulated.pop(0)
                    target = articulated.pop(0)
                    new_state.remote_mappings[trigger] = (target, "is_open")
                    bindings_info["remote_control"] = {"trigger": trigger, "target": target}
                elif articulated and furniture:
                    trigger = articulated.pop(0)
                    target = furniture[0] if furniture else trigger
                    new_state.remote_mappings[trigger] = (target, "is_open")
                    bindings_info["remote_control"] = {"trigger": trigger, "target": target}

            elif mech_type == "counting_state":
                # Bind to an articulated object, require 3 interactions
                if articulated:
                    target = articulated.pop(0)
                    new_state.interaction_counts[target] = 0
                    new_state.object_properties.setdefault(target, {})["required_count"] = 3
                    bindings_info["counting_state"] = {"target": target, "required_count": 3}

            elif mech_type == "delayed_effect":
                # Bind to an articulated object, delay by 2 steps
                if articulated:
                    target = articulated.pop(0)
                    new_state.object_properties.setdefault(target, {})["delay_steps"] = 2
                    bindings_info["delayed_effect"] = {"target": target, "delay_steps": 2}

            elif mech_type == "state_mirroring":
                # Bind pair of articulated objects
                if len(articulated) >= 2:
                    obj1 = articulated.pop(0)
                    obj2 = articulated.pop(0)
                    new_state.mirror_pairs.append((obj1, obj2, "is_open"))
                    bindings_info["state_mirroring"] = {"pair": [obj1, obj2]}

            elif mech_type == "conditional_unlock":
                # First must interact with prereq, then can use target
                if len(articulated) >= 2:
                    prereq = articulated.pop(0)
                    target = articulated.pop(0)
                    new_state.object_properties.setdefault(target, {})["prerequisite"] = prereq
                    new_state.object_properties.setdefault(prereq, {})["unlocks"] = target
                    bindings_info["conditional_unlock"] = {"prerequisite": prereq, "target": target}

        # Set up hidden items for Shake action (use synthetic items not in world graph)
        if shakeable:
            # Pick a shakeable item that's not already used by other mechanics
            container = None
            for s in shakeable:
                if s not in new_state.inverse_objects and s not in new_state.remote_mappings:
                    container = s
                    break
            if container:
                # Use synthetic hidden items (keys, notes, etc.) not in world graph
                hidden_items_pool = ["hidden_key", "secret_note", "small_coin", "tiny_gem", "old_map"]
                hidden_item = random.choice(hidden_items_pool)
                new_state = new_state.set_object_property(container, "hidden_inside", hidden_item)
                bindings_info["hidden_items"] = {container: hidden_item}

        self.state = new_state
        return new_state, bindings_info

    def apply_action(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        state: Optional[EMTOMGameState] = None,
    ) -> Tuple[EMTOMGameState, ActionExecutionResult]:
        """
        Apply an action, running it through active mechanics.

        Args:
            action_name: Name of the action (e.g., "open", "Shake")
            agent_id: Agent performing the action
            target: Target of the action
            state: State to apply to. If None, uses self.state.

        Returns:
            (new_state, result) tuple
        """
        if state is None:
            state = self.state

        # Apply mechanics first
        mech_result = apply_mechanics(action_name, agent_id, target, state)
        state = mech_result.state

        # If mechanic didn't handle it (or partially handled), apply built-in actions
        if not mech_result.applies or mech_result.success:
            state, builtin_result = self._apply_builtin_action(
                action_name, agent_id, target, state, mech_result
            )
            if builtin_result:
                mech_result = builtin_result

        # Record the action
        record = ActionRecord(
            step=state.current_step,
            agent_id=agent_id,
            action_name=action_name,
            target=target,
            success=mech_result.success,
            observation=mech_result.observation,
            effects=mech_result.effects,
        )
        state = state.record_action(record)
        state = state.add_observation(agent_id, mech_result.observation)

        result = ActionExecutionResult(
            state=state,
            observation=mech_result.observation,
            success=mech_result.success,
            effects=mech_result.effects,
            surprise_trigger=mech_result.surprise_trigger,
        )

        self.state = state
        return state, result

    def _apply_builtin_action(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        state: EMTOMGameState,
        mech_result: HandlerResult,
    ) -> Tuple[EMTOMGameState, Optional[HandlerResult]]:
        """Apply built-in action effects (Shake, Hide, etc.)."""

        if action_name == "Shake" and target:
            hidden = state.get_object_property(target, "hidden_inside")
            if hidden:
                # Add hidden item to agent's inventory
                new_state = copy.copy(state)
                if agent_id not in new_state.agent_inventory:
                    new_state.agent_inventory[agent_id] = []
                new_state.agent_inventory[agent_id].append(hidden)

                # Clear the hidden item from container
                new_state = new_state.set_object_property(target, "hidden_inside", None)

                return new_state, HandlerResult(
                    applies=True,
                    state=new_state,
                    observation=f"You shake {target}. A {hidden} falls out and you pick it up!",
                    success=True,
                    effects=[f"found={hidden}", f"inventory+={hidden}"],
                )
            else:
                return state, HandlerResult(
                    applies=True,
                    state=state,
                    observation=f"You shake {target}. Nothing falls out.",
                    success=True,
                    effects=[],
                )

        elif action_name == "Hide" and target:
            new_hidden = copy.copy(state.hidden_objects)
            new_hidden.add(target)
            state = copy.copy(state)
            state.hidden_objects = new_hidden

            return state, HandlerResult(
                applies=True,
                state=state,
                observation=f"You hide {target}. It is no longer visible to others.",
                success=True,
                effects=[f"hidden={target}"],
            )

        elif action_name == "Inspect" and target:
            # Get object info
            obj_states = state.object_states.get(target, {})
            obj_props = state.object_properties.get(target, {})

            details = []
            for k, v in obj_states.items():
                if k.startswith("is_"):
                    readable = k.replace("is_", "").replace("_", " ")
                    details.append(f"{readable}: {'yes' if v else 'no'}")

            if details:
                obs = f"You examine {target}. You observe: {', '.join(details)}."
            else:
                obs = f"You examine {target}. It appears normal."

            return state, HandlerResult(
                applies=True,
                state=state,
                observation=obs,
                success=True,
                effects=[],
            )

        return state, None

    def tick(self, state: Optional[EMTOMGameState] = None) -> Tuple[EMTOMGameState, List[str]]:
        """
        Advance time by one step.

        Processes pending effects and increments step counter.

        Args:
            state: State to tick. If None, uses self.state.

        Returns:
            (new_state, triggered_effects) tuple
        """
        if state is None:
            state = self.state

        triggered_descriptions = []
        remaining = []

        for effect in state.pending_effects:
            new_steps = effect.steps_remaining - 1
            if new_steps <= 0:
                triggered_descriptions.append(effect.description)
                # Apply the effect
                state = state.set_object_property(
                    effect.target,
                    effect.property_name,
                    effect.new_value,
                )
            else:
                remaining.append(PendingEffect(
                    effect_id=effect.effect_id,
                    target=effect.target,
                    property_name=effect.property_name,
                    new_value=effect.new_value,
                    steps_remaining=new_steps,
                    triggered_by=effect.triggered_by,
                    triggered_at_step=effect.triggered_at_step,
                    description=effect.description,
                ))

        new_state = copy.copy(state)
        new_state.pending_effects = remaining
        new_state = new_state.increment_step()

        self.state = new_state
        return new_state, triggered_descriptions

    def check_goals(self, state: Optional[EMTOMGameState] = None) -> List[Goal]:
        """
        Check which goals have been completed.

        Returns:
            List of newly completed goals
        """
        if state is None:
            state = self.state

        newly_completed = []

        for goal in state.goals:
            if goal.goal_id in state.completed_goals:
                continue

            completed = False

            if goal.goal_type == "find_item" or goal.goal_type == "reveal_item":
                for item in state.spawned_items:
                    if item.item_id == goal.target:
                        completed = True
                        break

            elif goal.goal_type == "change_state":
                if goal.target and goal.target_state:
                    obj_states = state.object_states.get(goal.target, {})
                    obj_props = state.object_properties.get(goal.target, {})
                    all_states = {**obj_states, **obj_props}
                    completed = all(
                        all_states.get(k) == v
                        for k, v in goal.target_state.items()
                    )

            if completed:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at_step = state.current_step
                newly_completed.append(goal)
                state.completed_goals.add(goal.goal_id)

        self.state = state
        return newly_completed

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about current state."""
        state = self.state
        return {
            "current_step": state.current_step,
            "active_mechanics": state.active_mechanics,
            "inverse_objects": list(state.inverse_objects),
            "remote_mappings": {k: list(v) for k, v in state.remote_mappings.items()},
            "mirror_pairs": state.mirror_pairs,
            "interaction_counts": state.interaction_counts,
            "sequence_progress": state.sequence_progress,
            "unlocked_targets": list(state.unlocked_targets),
            "pending_effects": len(state.pending_effects),
            "spawned_items": [s.item_id for s in state.spawned_items],
            "hidden_objects": list(state.hidden_objects),
            "goals": [
                {"id": g.goal_id, "status": g.status.value}
                for g in state.goals
            ],
        }

    def get_state(self) -> EMTOMGameState:
        """Get current game state."""
        return self.state

    def set_state(self, state: EMTOMGameState) -> None:
        """Set current game state."""
        self.state = state
