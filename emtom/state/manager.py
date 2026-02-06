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
    ActionRecord,
    Goal,
    GoalStatus,
)
from emtom.state.items import BaseItem
from emtom.state.item_registry import ItemRegistry
from emtom.mechanics.handlers import (
    apply_mechanics,
    HandlerResult,
    MECHANIC_INFO,
)

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from emtom.task_gen.dag import DAGProgress


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

        # Story context from scenario system (for prompts)
        self._story_context: Optional[str] = None
        self._bindings_info: Optional[Dict[str, Any]] = None

        # State history for temporal evaluation (PARTNR-style)
        self._state_history: List[EMTOMGameState] = []
        self._success_condition: Optional[Dict[str, Any]] = None

        # DAG progress tracking for subtask completion
        self._dag_progress: Optional["DAGProgress"] = None
        self._subtasks: List[Any] = []  # Store subtasks for reference

    def initialize_from_task(self, task_data: Dict[str, Any]) -> EMTOMGameState:
        """
        Initialize game state from a task definition.

        Args:
            task_data: Task definition dict containing:
                - mechanics: List of mechanic bindings
                - items: Item definitions with inside/container placements
                - goals: Task goals

        Returns:
            Initialized game state
        """
        state = EMTOMGameState()

        # Set active mechanics from bindings
        # Support both "mechanics" (old) and "mechanic_bindings" (new) field names
        mechanics = task_data.get("mechanic_bindings", task_data.get("mechanics", []))
        state.active_mechanics = task_data.get("active_mechanics", []).copy()
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

        # Load item definitions
        items_data = task_data.get("items", [])
        for item_data in items_data:
            item_id = item_data.get("item_id")
            if item_id:
                # Ensure base_id and instance_id are set for proper class lookup
                enriched_data = dict(item_data)
                if "base_id" not in enriched_data:
                    enriched_data["base_id"] = ItemRegistry.get_base_id(item_id)
                if "instance_id" not in enriched_data:
                    enriched_data["instance_id"] = item_id
                state.item_definitions[item_id] = enriched_data
                # If item is inside a container, add to items_inside.
                # Backward-compat: accept legacy "hidden_in" from older tasks.
                # Items are found via Open and go directly to inventory
                inside = item_data.get("inside") or item_data.get("hidden_in")
                if inside:
                    current = state.get_object_property(inside, "items_inside", [])
                    if not isinstance(current, list):
                        current = [current] if current else []
                    if item_id not in current:
                        current = current + [item_id]
                    state = state.set_object_property(inside, "items_inside", current)

        # Set up locked containers (require key to open)
        locked_containers = task_data.get("locked_containers", {})
        for container_id, key_type in locked_containers.items():
            state = state.set_object_property(container_id, "is_locked", True)
            state = state.set_object_property(container_id, "required_key", key_type)

        # Apply initial states (unified format: object_id -> {property: value})
        # This allows tasks to start with doors open, objects dirty, etc.
        initial_states = task_data.get("initial_states", {})
        for object_id, properties in initial_states.items():
            if isinstance(properties, dict):
                for prop_name, prop_value in properties.items():
                    state = state.set_object_property(object_id, prop_name, prop_value)

        # Load termination condition (for competitive tasks)
        termination = task_data.get("termination")
        if termination:
            state.termination_condition = termination

        # Load team membership (defaults to team_N -> [agent_N] if not specified)
        team_members = task_data.get("team_members", {})
        if team_members:
            state.team_members = team_members

        self.state = state

        # Store success condition for evaluation
        self._success_condition = task_data.get("success_condition")

        # Reset state history and record initial state
        self._state_history = [copy.deepcopy(state)]

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

        elif mech_type == "conditional_unlock":
            if trigger:
                prereq = binding.get("prerequisite_object")
                if prereq:
                    state.object_properties.setdefault(trigger, {})["prerequisite"] = prereq
                    state.object_properties.setdefault(prereq, {})["unlocks"] = trigger

        elif mech_type == "room_restriction":
            restricted_rooms = binding.get("restricted_rooms", [])
            for_agents = binding.get("for_agents", [])
            for agent_id in for_agents:
                if agent_id not in state.restricted_rooms:
                    state.restricted_rooms[agent_id] = set()
                state.restricted_rooms[agent_id].update(restricted_rooms)

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
            world_graph_dict = self.env.world_graph
        except AttributeError:
            return state

        # Sync agent positions and rooms from WorldGraph objects
        agent_positions = {}
        agent_rooms = {}

        # world_graph_dict is Dict[int, WorldGraph] - use first available graph to get all agents
        wg = None
        for uid, graph in world_graph_dict.items():
            if graph is not None:
                wg = graph
                break

        if wg is not None:
            try:
                agents = wg.get_agents()
                for agent_node in agents:
                    # Get agent name from node
                    agent_name = agent_node.name if hasattr(agent_node, 'name') else None
                    if not agent_name:
                        continue

                    # Normalize to agent_N format
                    if agent_name.startswith("agent_"):
                        agent_id = agent_name
                    elif agent_name.isdigit():
                        agent_id = f"agent_{agent_name}"
                    else:
                        # Try to extract number or use as-is
                        agent_id = agent_name

                    # Get position if available
                    if hasattr(agent_node, 'position'):
                        agent_positions[agent_id] = agent_node.position

                    # Get room for this agent
                    try:
                        rooms = wg.get_room_for_entity(agent_node)
                        if rooms and len(rooms) > 0:
                            room_node = rooms[0]
                            room_name = room_node.name if hasattr(room_node, 'name') else str(room_node)
                            agent_rooms[agent_id] = room_name
                    except (ValueError, AttributeError):
                        pass
            except (ValueError, AttributeError):
                # WorldGraph might not have agents yet
                pass

        # Sync object states from entities in world graph
        object_states = {}
        entities = []

        room_by_entity: Dict[str, str] = {}
        for uid, wg in world_graph_dict.items():
            if wg is None:
                continue
            try:
                # Get all object nodes from the graph
                objects = wg.get_all_objects() if hasattr(wg, 'get_all_objects') else []
                for obj_node in objects:
                    entity_id = obj_node.name if hasattr(obj_node, 'name') else str(obj_node)
                    entity_dict = {"id": entity_id, "name": entity_id}

                    # Extract is_* states
                    states = {}
                    for attr in dir(obj_node):
                        if attr.startswith("is_") and not callable(getattr(obj_node, attr, None)):
                            try:
                                states[attr] = getattr(obj_node, attr)
                            except:
                                pass
                    if states:
                        object_states[entity_id] = states
                        entity_dict.update(states)

                    # Try to resolve room for this entity
                    try:
                        room = wg.get_room_for_entity(obj_node)
                        room_name = room.name if hasattr(room, "name") else str(room)
                        entity_dict["room"] = room_name
                        room_by_entity[entity_id] = room_name
                    except Exception:
                        pass

                    entities.append(entity_dict)
            except (ValueError, AttributeError):
                pass
            # Only need to process one agent's world graph for entities
            break

        new_state = copy.copy(state)
        new_state.agent_positions = agent_positions
        new_state.agent_rooms = agent_rooms
        new_state.object_states = object_states
        new_state.entities = entities

        # Persist room mapping into object_properties for quick lookup
        for entity_id, room_name in room_by_entity.items():
            new_state = new_state.set_object_property(entity_id, "room", room_name)

        self.state = new_state
        return new_state

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

        # Set up scenario-based items for Open action
        if shakeable:
            # Load scenario system
            from emtom.task_gen.scenario_system import get_compatible_scenario, ScenarioInstantiator
            from emtom.task_gen.clue_generator import ClueGenerator

            # Get a scenario compatible with active mechanics
            scenario = get_compatible_scenario(state.active_mechanics)

            if scenario:
                bindings_info["scenario"] = {
                    "id": scenario.id,
                    "theme": scenario.theme,
                    "title": scenario.title_template,
                    "requires_collaboration": scenario.requires_collaboration,
                }

                # Get available containers (not used by other mechanics)
                available_containers = [
                    s for s in shakeable
                    if s not in new_state.inverse_objects and s not in new_state.remote_mappings
                ]

                if available_containers:
                    # Bind items based on scenario requirements
                    item_locations = {}
                    items_inside = {}

                    for i, item_needed in enumerate(scenario.items_needed):
                        if i >= len(available_containers):
                            break

                        container = available_containers[i]

                        # Get the appropriate item from registry
                        item = None
                        if ItemRegistry.is_registered(item_needed):
                            item = ItemRegistry.instantiate(item_needed, i + 1)
                        else:
                            # Default to small key for unknown items
                            item = ItemRegistry.instantiate("small_key", i + 1)

                        if item:
                            # Register item definition in state
                            new_state.item_definitions[item.instance_id] = item.to_dict()
                            # Add to items_inside list on container
                            current = new_state.get_object_property(container, "items_inside", [])
                            if not isinstance(current, list):
                                current = [current] if current else []
                            current = current + [item.instance_id]
                            new_state = new_state.set_object_property(
                                container, "items_inside", current
                            )
                            item_locations[item.instance_id] = container
                            items_inside[container] = current

                    # Generate clues for item locations
                    clue_gen = ClueGenerator()
                    clues = []
                    for item_id, container in item_locations.items():
                        # Determine room (simplified - use first room or generic)
                        room = "this area"
                        container_clues = clue_gen.generate_all_clues(container, room)
                        clues.extend(container_clues)

                    # Build scene inventory for instantiation
                    scene_inventory = {
                        "furniture": furniture,
                        "objects": objects,
                        "rooms": ["room"],  # Would need scene graph for real rooms
                    }

                    # Instantiate scenario
                    instantiator = ScenarioInstantiator(clue_gen)
                    instantiated = instantiator.instantiate(
                        scenario,
                        scene_inventory,
                        item_locations,
                        primary_room="this area",
                    )

                    # Lock some containers (creates key puzzles)
                    locked_containers = {}
                    available_for_locking = [
                        f for f in articulated
                        if f not in items_inside  # Don't lock containers with items inside
                    ]
                    if available_for_locking and item_locations:
                        # Lock 1-2 containers
                        num_to_lock = min(2, len(available_for_locking))
                        containers_to_lock = random.sample(available_for_locking, num_to_lock)
                        for container in containers_to_lock:
                            new_state = new_state.set_object_property(container, "is_locked", True)
                            new_state = new_state.set_object_property(container, "required_key", "small_key")
                            locked_containers[container] = "small_key"

                    # Store all scenario info in bindings
                    bindings_info["items_inside"] = items_inside
                    bindings_info["item_locations"] = item_locations
                    bindings_info["item_definitions"] = {
                        item_id: new_state.item_definitions[item_id].get("name", item_id)
                        for item_id in item_locations
                    }
                    bindings_info["clues"] = clues
                    bindings_info["story_context"] = instantiated.story_context
                    bindings_info["suggested_locations"] = instantiated.suggested_locations
                    if locked_containers:
                        bindings_info["locked_containers"] = locked_containers
                    if instantiated.agent_secrets:
                        bindings_info["agent_secrets"] = instantiated.agent_secrets

        self.state = new_state

        # Store bindings for later retrieval (e.g., by exploration prompts)
        self._bindings_info = bindings_info

        # Story context only comes from YAML scenario templates (if loaded above)
        # No fallbacks - story is optional for exploration mode
        self._story_context = bindings_info.get("story_context")

        return new_state, bindings_info

    def get_story_context(self) -> Optional[str]:
        """Get the story context from the current scenario (if any)."""
        return self._story_context

    def get_bindings_info(self) -> Optional[Dict[str, Any]]:
        """Get the full bindings info from auto_bind_mechanics."""
        return self._bindings_info

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
            action_name: Name of the action (e.g., "Open", "Pick")
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

        # Record state snapshot for temporal evaluation
        self._state_history.append(copy.deepcopy(state))

        return state, result

    def _apply_builtin_action(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        state: EMTOMGameState,
        mech_result: HandlerResult,
    ) -> Tuple[EMTOMGameState, Optional[HandlerResult]]:
        """Apply built-in action effects (Hide, etc.)."""

        if action_name == "Hide" and target:
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

        return state, None

    def tick(self, state: Optional[EMTOMGameState] = None) -> Tuple[EMTOMGameState, List[str]]:
        """
        Advance time by one step.

        Increments step counter.

        Args:
            state: State to tick. If None, uses self.state.

        Returns:
            (new_state, triggered_effects) tuple
        """
        if state is None:
            state = self.state

        new_state = state.increment_step()
        self.state = new_state
        return new_state, []

    def check_termination(self) -> bool:
        """
        Check if episode should terminate based on termination condition.

        Call this after each step. For competitive tasks, predicates like
        has_most are only evaluated after termination.

        Returns:
            True if episode is terminated (or just became terminated)
        """
        if self.state.is_terminated:
            return True

        condition = self.state.termination_condition
        if not condition:
            return False

        term_type = condition.get("type")
        value = condition.get("value")

        if term_type == "max_steps":
            if self.state.current_step >= value:
                self.state.is_terminated = True
                self.state.termination_reason = f"Reached step limit ({value})"
                return True

        elif term_type == "all_collected":
            # value is the item_type to check (e.g., "item_gold_coin")
            item_type = value
            if not item_type.startswith("item_"):
                item_type = f"item_{item_type}"

            # Count total defined items of this type
            total_defined = sum(
                1 for item_id in self.state.item_definitions
                if ItemRegistry.get_base_id(item_id) == item_type
            )

            # Count total collected across all agents
            total_collected = sum(
                self.count_items_by_type(agent_id, item_type)
                for agent_id in self.state.agent_inventory.keys()
            )

            if total_collected >= total_defined and total_defined > 0:
                self.state.is_terminated = True
                self.state.termination_reason = f"All {item_type} collected ({total_collected}/{total_defined})"
                return True

        elif term_type == "any_at_location":
            # value is the room/location to check
            room = value
            for agent_id, agent_room in self.state.agent_rooms.items():
                if agent_room == room:
                    self.state.is_terminated = True
                    self.state.termination_reason = f"{agent_id} reached {room}"
                    return True

        elif term_type == "all_at_location":
            # value is the room/location to check
            room = value
            all_agents = list(self.state.agent_rooms.keys())
            if all_agents and all(
                self.state.agent_rooms.get(aid) == room for aid in all_agents
            ):
                self.state.is_terminated = True
                self.state.termination_reason = f"All agents reached {room}"
                return True

        return False

    def is_terminated(self) -> bool:
        """Check if episode is terminated."""
        return self.state.is_terminated

    def get_termination_reason(self) -> Optional[str]:
        """Get the reason for termination, if terminated."""
        return self.state.termination_reason

    # ========== Stun Methods ==========

    def is_agent_stunned(self, agent_id: str) -> bool:
        """
        Check if an agent is currently stunned.

        Args:
            agent_id: Agent to check

        Returns:
            True if agent has remaining stun turns
        """
        return self.state.stunned_agents.get(agent_id, 0) > 0

    def get_stun_turns_remaining(self, agent_id: str) -> int:
        """
        Get remaining stun turns for an agent.

        Args:
            agent_id: Agent to check

        Returns:
            Number of turns remaining (0 if not stunned)
        """
        return self.state.stunned_agents.get(agent_id, 0)

    def stun_agent(self, agent_id: str, turns: int = 1) -> None:
        """
        Apply stun to an agent.

        Args:
            agent_id: Agent to stun
            turns: Number of turns to skip (default 1)
        """
        self.state.stunned_agents[agent_id] = turns

    def clear_stun(self, agent_id: str) -> None:
        """
        Remove stun from an agent.

        Args:
            agent_id: Agent to clear stun from
        """
        if agent_id in self.state.stunned_agents:
            del self.state.stunned_agents[agent_id]

    def check_and_process_stun(self, agent_id: str) -> Optional[str]:
        """
        Check if agent is stunned and process the stun effect.

        Call this BEFORE processing an agent's action each turn.
        If stunned, decrements the stun counter and returns a skip message.
        If stun counter reaches 0, removes the stun.

        Args:
            agent_id: Agent about to act

        Returns:
            Skip message if stunned (action should be skipped),
            None if agent can act normally
        """
        if not self.is_agent_stunned(agent_id):
            return None

        turns_left = self.state.stunned_agents[agent_id]

        # Decrement stun counter
        self.state.stunned_agents[agent_id] = turns_left - 1

        # Clean up if stun is over
        if self.state.stunned_agents[agent_id] <= 0:
            del self.state.stunned_agents[agent_id]

        return f"You are stunned and cannot act this turn! ({turns_left} turn(s) of stun remaining before this)"

    def decrement_all_stuns(self) -> Dict[str, str]:
        """
        Decrement stun counters for all stunned agents.

        Alternative to check_and_process_stun - call this once per game tick
        instead of per-agent. Returns messages for any agents whose stun ended.

        Returns:
            Dict of agent_id -> message for agents whose stun ended
        """
        messages = {}
        to_remove = []

        for agent_id, turns in list(self.state.stunned_agents.items()):
            new_turns = turns - 1
            if new_turns <= 0:
                to_remove.append(agent_id)
                messages[agent_id] = "You are no longer stunned."
            else:
                self.state.stunned_agents[agent_id] = new_turns

        for agent_id in to_remove:
            del self.state.stunned_agents[agent_id]

        return messages

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
                    completed = self._check_required_states(
                        [{"entity": goal.target, **goal.target_state}],
                        state
                    )

            if completed:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at_step = state.current_step
                newly_completed.append(goal)
                state.completed_goals.add(goal.goal_id)

        self.state = state
        return newly_completed

    def _check_required_states(
        self,
        required_states: List[Dict[str, Any]],
        state: EMTOMGameState,
    ) -> bool:
        """
        Check if all required states are satisfied.

        Supports:
        - entity object states (is_open, is_on, etc.)
        - agent inventory checks (property="has_item")

        Args:
            required_states: List of {entity, property, value} dicts
            state: Game state to check

        Returns:
            True if all required states are satisfied
        """
        for req in required_states:
            entity = req.get("entity")
            prop = req.get("property")
            value = req.get("value")

            # Check if this is an inventory check
            if prop == "has_item":
                # Entity should be agent_id (e.g., "agent_0")
                if not self.agent_has_item(entity, value):
                    return False

            else:
                # Standard object state check
                obj_states = state.object_states.get(entity, {})
                obj_props = state.object_properties.get(entity, {})
                all_states = {**obj_states, **obj_props}

                if all_states.get(prop) != value:
                    return False

        return True

    def check_success_condition(
        self,
        success_condition: Dict[str, Any],
        state: Optional[EMTOMGameState] = None,
    ) -> bool:
        """
        Check if a task's success condition is met.

        Args:
            success_condition: Dict with "required_states" key
            state: Game state to check

        Returns:
            True if success condition is satisfied
        """
        if state is None:
            state = self.state

        required_states = success_condition.get("required_states", [])
        return self._check_required_states(required_states, state)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about current state."""
        state = self.state

        # Collect items_inside (container -> item mappings)
        items_inside = {}
        for obj_id, props in state.object_properties.items():
            i_items = props.get("items_inside", [])
            if i_items:
                items_inside[obj_id] = i_items

        return {
            "current_step": state.current_step,
            "active_mechanics": state.active_mechanics,
            "inverse_objects": list(state.inverse_objects),
            "remote_mappings": {k: list(v) for k, v in state.remote_mappings.items()},
            "mirror_pairs": state.mirror_pairs,
            "unlocked_targets": list(state.unlocked_targets),
            "spawned_items": [s.item_id for s in state.spawned_items],
            "hidden_objects": list(state.hidden_objects),
            "items_inside": items_inside,  # container -> item_id mappings (found via Open)
            "item_definitions": state.item_definitions,  # item definitions from task
            "agent_inventory": state.agent_inventory,  # per-agent inventory
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

    # ========== Inventory Methods ==========

    def grant_item(
        self,
        agent_id: str,
        item_id: str,
        source: Optional[str] = None,
        state: Optional[EMTOMGameState] = None,
    ) -> Tuple[EMTOMGameState, bool, str]:
        """
        Grant an item to an agent's inventory.

        Args:
            agent_id: Agent receiving the item
            item_id: Item ID to grant
            source: What granted this item (object, mechanic, etc.)
            state: State to modify (uses self.state if None)

        Returns:
            (new_state, success, message)
        """
        if state is None:
            state = self.state

        # Get item info (from state definitions or registry)
        item = self.get_item(item_id)
        item_name = item.name if item else item_id

        # Check if already have
        if item_id in state.agent_inventory.get(agent_id, set()):
            return state, False, f"Already have {item_name}."

        # Add to inventory (using sets for O(1) lookup)
        new_state = copy.copy(state)
        new_inv = {k: set(v) for k, v in state.agent_inventory.items()}
        if agent_id not in new_inv:
            new_inv[agent_id] = set()
        new_inv[agent_id].add(item_id)
        new_state.agent_inventory = new_inv

        self.state = new_state

        # Call item's on_acquire hook if available
        if item and hasattr(item, 'on_acquire'):
            msg = item.on_acquire(self, agent_id)
            return new_state, True, msg

        return new_state, True, f"Obtained {item_name}!"

    def remove_item(
        self,
        agent_id: str,
        item_id: str,
        state: Optional[EMTOMGameState] = None,
    ) -> Tuple[EMTOMGameState, bool, str]:
        """
        Remove an item from an agent's inventory.

        Args:
            agent_id: Agent to remove item from
            item_id: Item ID to remove
            state: State to modify (uses self.state if None)

        Returns:
            (new_state, success, message)
        """
        if state is None:
            state = self.state

        # Check if agent has item
        if agent_id not in state.agent_inventory or item_id not in state.agent_inventory[agent_id]:
            item = self.get_item(item_id)
            item_name = item.name if item else item_id
            return state, False, f"Don't have {item_name}."

        # Remove from inventory
        new_state = copy.copy(state)
        new_inv = {k: set(v) for k, v in state.agent_inventory.items()}
        new_inv[agent_id].discard(item_id)
        new_state.agent_inventory = new_inv

        self.state = new_state
        item = self.get_item(item_id)
        item_name = item.name if item else item_id
        return new_state, True, f"Removed {item_name}."

    def agent_has_item(self, agent_id: str, item_id: str) -> bool:
        """
        Check if an agent has a specific item.

        O(1) lookup using sets.

        Args:
            agent_id: Agent to check
            item_id: Item to check for

        Returns:
            True if agent has the item
        """
        return item_id in self.state.agent_inventory.get(agent_id, set())

    def get_item(self, item_id: str) -> Optional[BaseItem]:
        """
        Get a BaseItem instance for an item ID.

        First checks state definitions, then falls back to registry.

        Args:
            item_id: The item instance ID (e.g., "small_key_1")

        Returns:
            BaseItem instance or None
        """
        # Check if defined in state
        item_data = self.state.item_definitions.get(item_id)
        if item_data:
            return BaseItem.from_dict(item_data)

        # Try to get from registry by base ID
        item_cls = ItemRegistry.get_by_instance_id(item_id)
        if item_cls:
            instance = item_cls()
            instance.instance_id = item_id
            instance.base_id = ItemRegistry.get_base_id(item_id)
            return instance

        return None

    def get_agent_inventory(self, agent_id: str) -> List[BaseItem]:
        """
        Get all items in agent's inventory.

        Args:
            agent_id: Agent to get inventory for

        Returns:
            List of BaseItem objects
        """
        item_ids = self.state.agent_inventory.get(agent_id, set())
        items = []
        for item_id in sorted(item_ids):  # Sort for consistent ordering
            item = self.get_item(item_id)
            if item:
                items.append(item)
        return items

    def count_items_by_type(self, agent_id: str, item_type: str) -> int:
        """
        Count how many items of a base type an agent has.

        Args:
            agent_id: Agent to count for
            item_type: Base item type (e.g., "item_gold_coin" or "gold_coin")

        Returns:
            Number of items of that type in agent's inventory
        """
        # Normalize item_type to include "item_" prefix
        if not item_type.startswith("item_"):
            item_type = f"item_{item_type}"

        count = 0
        for item_id in self.state.agent_inventory.get(agent_id, set()):
            base_id = ItemRegistry.get_base_id(item_id)
            if base_id == item_type:
                count += 1
        return count

    def count_team_items_by_type(self, team_id: str, item_type: str) -> int:
        """
        Count how many items of a base type a team has (sum across all team members).

        Args:
            team_id: Team to count for (e.g., "team_0")
            item_type: Base item type (e.g., "item_gold_coin")

        Returns:
            Total items of that type across all team members
        """
        # Get team members (default: team_N -> [agent_N])
        team_members = self.state.team_members.get(team_id)
        if team_members is None:
            # Default mapping: team_0 -> agent_0, team_1 -> agent_1
            agent_num = team_id.replace("team_", "")
            team_members = [f"agent_{agent_num}"]

        total = 0
        for agent_id in team_members:
            total += self.count_items_by_type(agent_id, item_type)
        return total

    def get_all_agent_ids(self) -> List[str]:
        """Get list of all agent IDs that have inventory entries."""
        return list(self.state.agent_inventory.keys())

    def get_all_team_ids(self) -> List[str]:
        """
        Get list of all team IDs.

        Returns configured teams, or defaults to team_0, team_1 based on agents.
        """
        if self.state.team_members:
            return list(self.state.team_members.keys())
        # Default: one team per agent
        return [f"team_{i}" for i in range(len(self.state.agent_inventory))]

    def get_inventory_text(self, agent_id: str) -> str:
        """
        Format agent's inventory as text for prompt templating.

        Args:
            agent_id: Agent to format inventory for

        Returns:
            Formatted inventory string
        """
        items = self.get_agent_inventory(agent_id)
        if not items:
            return "Your inventory is empty."

        lines = ["Your inventory:"]
        for item in items:
            line = f"- {item.name}: {item.description}"
            if item.grants_action:
                line += f" (enables {item.grants_action} action)"
            if item.task_info:
                line += f" ({item.task_info})"
            if item.consumable and item.uses_remaining is not None:
                line += f" [{item.uses_remaining} uses remaining]"
            lines.append(line)

        return "\n".join(lines)

    def get_agent_passive_effects(self, agent_id: str) -> Dict[str, Any]:
        """
        Get all passive effects from items in agent's inventory.

        Collects passive_effects from all items the agent has. Use this
        to check for special abilities at action time.

        Args:
            agent_id: Agent to get passive effects for

        Returns:
            Dict of all passive effects (merged from all items)

        Example:
            effects = manager.get_agent_passive_effects("agent_0")
            if effects.get("oracle_world_graph"):
                # Agent can see full world state
        """
        effects: Dict[str, Any] = {}
        items = self.get_agent_inventory(agent_id)
        for item in items:
            if item.passive_effects:
                effects.update(item.passive_effects)
        return effects

    def use_item(
        self,
        agent_id: str,
        item_id: str,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Use an item from the agent's inventory.

        Validates args against item.use_args, then calls on_use().

        Args:
            agent_id: Agent using the item
            item_id: Item to use
            args: List of arguments for the Use action

        Returns:
            (success, message) tuple
        """
        args = args or []

        # Check if agent has item
        if not self.agent_has_item(agent_id, item_id):
            return False, f"You don't have that item."

        # Get item instance
        item = self.get_item(item_id)
        if not item:
            return False, f"Unknown item: {item_id}"

        # Validate args against item.use_args
        expected_args = getattr(item, 'use_args', []) or []
        if len(args) != len(expected_args):
            if expected_args:
                usage = f"UseItem[{item_id}, {', '.join(expected_args)}]"
            else:
                usage = f"UseItem[{item_id}]"
            return False, f"Wrong number of arguments. Usage: {usage}"

        # Call item's on_use method
        success, message = item.on_use(self, agent_id, args)

        # Handle consumable items
        if success and item.consumable:
            if item.uses_remaining is not None and item.uses_remaining > 1:
                # Decrement uses
                item.uses_remaining -= 1
                # Update in state
                self.state.item_definitions[item_id] = item.to_dict()
            else:
                # Single use or last use - remove item
                self.remove_item(agent_id, item_id)

        return success, message

    def _check_has_item(self, entity: str, value: str) -> bool:
        """
        Check if an entity (agent) has an item.

        Used for success criteria checking with property="has_item".

        Args:
            entity: Agent ID (e.g., "agent_0")
            value: Item ID to check for

        Returns:
            True if agent has the item
        """
        return self.agent_has_item(entity, value)

    def _check_is_unlocked(self, entity: str) -> bool:
        """
        Check if a container is unlocked.

        Used for success criteria checking with property="is_unlocked".

        Args:
            entity: Container ID (e.g., "cabinet_45")

        Returns:
            True if container is NOT locked
        """
        # Prefer explicit is_unlocked flag if present (e.g., remote unlock mechanics)
        is_unlocked = self.state.get_object_property(entity, "is_unlocked", None)
        if is_unlocked is not None:
            return bool(is_unlocked)

        is_locked = self.state.get_object_property(entity, "is_locked", False)
        return not is_locked

    def _check_is_used(self, entity: str) -> bool:
        """
        Check if an item has been used/consumed.

        Used for success criteria checking with property="is_used".
        An item is "used" if it was granted to some agent but is no longer
        in any agent's inventory (consumed).

        Args:
            entity: Item ID (e.g., "item_small_key_1")

        Returns:
            True if item was granted and is now consumed
        """
        # Check if item is in any agent's inventory
        for agent_id, inventory in self.state.agent_inventory.items():
            if entity in inventory:
                return False  # Still in inventory, not used

        # Item not in any inventory - check if it was ever granted
        # by looking at action history
        for record in self.state.action_history:
            # Check if item was obtained via Open, etc.
            if entity in record.observation:
                return True  # Item was mentioned in an observation (obtained then used)

        # Also check spawned items
        for spawned in self.state.spawned_items:
            if spawned.item_id == entity and spawned.picked_up_by is not None:
                return True  # Was spawned and picked up, now gone = used

        return False

    # EMTOM game state predicates that can be used in success_condition
    GAME_STATE_PREDICATES = {"has_item", "is_unlocked", "is_used", "has_at_least", "has_most"}

    def _check_game_state_predicate(self, condition: Dict[str, Any]) -> Optional[bool]:
        """
        Check if condition is an EMTOM game state predicate and evaluate it.

        Args:
            condition: Success condition dict with entity, property, value/target

        Returns:
            True/False if this is a game state predicate, None if not handled
        """
        prop = condition.get("property")
        if prop not in self.GAME_STATE_PREDICATES:
            return None  # Not a game state predicate

        entity = condition.get("entity")
        target = condition.get("target")
        value = condition.get("value")

        if prop == "has_item":
            # entity=agent_id, target=item_id
            item_id = target or value
            if not item_id:
                return False
            return self._check_has_item(entity, item_id)

        elif prop == "is_unlocked":
            # entity=container_id
            result = self._check_is_unlocked(entity)
            # Handle value=False (checking if locked)
            if value is False:
                return not result
            return result

        elif prop == "is_used":
            # entity=item_id
            result = self._check_is_used(entity)
            # Handle value=False (checking if NOT used)
            if value is False:
                return not result
            return result

        elif prop == "has_at_least":
            # entity=agent_id or team_id, target=item_type, value=count
            if not target:
                return False
            count = value if isinstance(value, int) else 1
            if entity and entity.startswith("team_"):
                actual = self.count_team_items_by_type(entity, target)
            else:
                actual = self.count_items_by_type(entity, target) if entity else 0
            return actual >= count

        elif prop == "has_most":
            # entity=agent_id or team_id, target=item_type
            if not target or not entity:
                return False
            is_team = entity.startswith("team_")
            if is_team:
                my_count = self.count_team_items_by_type(entity, target)
                others = self.get_all_team_ids()
            else:
                my_count = self.count_items_by_type(entity, target)
                others = self.get_all_agent_ids()

            for other in others:
                if other == entity:
                    continue
                other_count = (
                    self.count_team_items_by_type(other, target)
                    if is_team
                    else self.count_items_by_type(other, target)
                )
                if other_count >= my_count:
                    return False
            return True

        return None

    # ========== PARTNR-style Evaluation ==========

    def evaluate_task(
        self,
        success_condition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion with PARTNR-style predicates.

        Uses the actual Habitat simulator state for ground-truth evaluation.

        Args:
            success_condition: Task's success condition dict. If None, uses
                              the one stored from initialize_from_task().

        Returns:
            Dict with:
                - percent_complete: float (0.0 to 1.0)
                - success: bool
                - failure_explanations: List[str]
                - proposition_status: Dict[str, bool]
        """
        from emtom.evaluation import evaluate_task

        condition = success_condition or self._success_condition
        if not condition:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": ["No success condition defined"],
                "proposition_status": {},
            }

        if not self.env:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": ["No environment interface available"],
                "proposition_status": {},
            }

        try:
            sim = self.env.sim
            result = evaluate_task(condition, sim)
            return result.to_dict()
        except Exception as e:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": [f"Evaluation error: {str(e)}"],
                "proposition_status": {},
            }

    def get_state_history(self) -> List[EMTOMGameState]:
        """Get the full state history for temporal analysis."""
        return self._state_history

    def clear_state_history(self) -> None:
        """Clear state history (useful for resetting)."""
        self._state_history = [copy.deepcopy(self.state)]

    # =========================================================================
    # DAG Progress Tracking
    # =========================================================================

    def initialize_dag_progress(self, subtasks: List[Any]) -> None:
        """
        Initialize DAG progress tracking from task subtasks.

        Args:
            subtasks: List of Subtask objects with success_conditions
        """
        from emtom.task_gen.dag import DAGProgress

        self._subtasks = subtasks
        if subtasks:
            self._dag_progress = DAGProgress.from_subtasks(subtasks)

    def update_dag_progress(self) -> Dict[str, Any]:
        """
        Update DAG progress by checking subtask conditions against current state.

        Should be called after each action to latch completed subtasks.

        Returns:
            Dict with completed, newly_completed, percent_complete, success
        """
        if not self._dag_progress:
            return {
                "completed": [],
                "newly_completed": [],
                "percent_complete": 0.0,
                "success": False,
                "error": "No DAG progress initialized",
            }

        def check_condition(subtask) -> bool:
            """Check if a subtask's success_condition is satisfied."""
            condition = subtask.success_condition
            if not condition:
                return False

            # First, check if this is an EMTOM game state predicate
            # (has_item, is_unlocked, is_used) - these don't need simulator
            game_state_result = self._check_game_state_predicate(condition)
            if game_state_result is not None:
                return game_state_result

            # Fall back to simulator-based predicates (is_on_top, is_open, etc.)
            from emtom.evaluation import evaluate_task

            # Wrap single condition in required_states format
            success_cond = {
                "description": subtask.description,
                "required_states": [condition],
            }

            if not self.env:
                return False

            try:
                sim = self.env.sim
                result = evaluate_task(success_cond, sim)
                return result.success
            except Exception:
                return False

        return self._dag_progress.update(check_condition)

    def get_dag_status(self) -> Dict[str, Any]:
        """Get current DAG progress status without updating."""
        if not self._dag_progress:
            return {
                "completed": [],
                "percent_complete": 0.0,
                "success": False,
                "remaining": [],
            }
        return self._dag_progress.get_status()

    def reset_dag_progress(self) -> None:
        """Reset DAG progress (clear all completed subtasks)."""
        if self._dag_progress:
            self._dag_progress.reset()
