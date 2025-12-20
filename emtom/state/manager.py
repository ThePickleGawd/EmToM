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
from emtom.state.items import ItemDefinition, ItemType
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

        # Story context from scenario system (for prompts)
        self._story_context: Optional[str] = None
        self._bindings_info: Optional[Dict[str, Any]] = None

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

        # Load item definitions
        items_data = task_data.get("items", [])
        for item_data in items_data:
            item_id = item_data.get("item_id")
            if item_id:
                state.item_definitions[item_id] = item_data
                # If item is hidden_in a container, set up the hidden_inside property
                hidden_in = item_data.get("hidden_in")
                if hidden_in:
                    state = state.set_object_property(hidden_in, "hidden_inside", item_id)

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

        # Set up scenario-based items for Shake/Search action
        if shakeable:
            # Load scenario system
            from emtom.task_gen.scenario_system import get_compatible_scenario, ScenarioInstantiator
            from emtom.task_gen.clue_generator import ClueGenerator
            from emtom.state.item_catalog import get_random_small_keys, get_big_key, get_radio

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
                    hidden_items = {}

                    for i, item_needed in enumerate(scenario.items_needed):
                        if i >= len(available_containers):
                            break

                        container = available_containers[i]

                        # Get the appropriate item from catalog
                        if item_needed == "small_key":
                            items = get_random_small_keys(1)
                            if items:
                                item = items[0]
                                item.item_id = f"small_key_{i+1}"
                        elif item_needed == "big_key":
                            item = get_big_key()
                        elif item_needed == "radio":
                            item = get_radio()
                        else:
                            # Default to small key for unknown items
                            items = get_random_small_keys(1)
                            item = items[0] if items else None

                        if item:
                            # Register item definition in state
                            new_state.item_definitions[item.item_id] = item.to_dict()
                            # Set hidden_inside property on container
                            new_state = new_state.set_object_property(
                                container, "hidden_inside", item.item_id
                            )
                            item_locations[item.item_id] = container
                            hidden_items[container] = item.item_id

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
                        if f not in hidden_items  # Don't lock containers with hidden items
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
                    bindings_info["hidden_items"] = hidden_items
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

        # Load story context from scenario OR from scraped .txt files
        story_context = bindings_info.get("story_context")
        if not story_context:
            # Fall back to scraped scenarios if no YAML scenario was loaded
            story_context = self._load_random_scraped_scenario()
            if story_context:
                bindings_info["story_context"] = story_context
                bindings_info["story_source"] = "scraped"

        self._story_context = story_context

        return new_state, bindings_info

    def _load_random_scraped_scenario(self) -> Optional[str]:
        """Load a random scenario from scraped .txt files."""
        import random
        import time
        from pathlib import Path

        scenarios_dir = Path(__file__).parent.parent.parent / "data" / "emtom" / "scenarios" / "scraped"
        if not scenarios_dir.exists():
            return None

        txt_files = list(scenarios_dir.glob("*.txt"))
        if not txt_files:
            return None

        # Use time-based randomness to avoid being affected by fixed seeds
        rng = random.Random(time.time_ns())
        chosen_file = rng.choice(txt_files)

        try:
            with open(chosen_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"[GameStateManager] Loaded scenario: {chosen_file.name}")
            return content
        except Exception as e:
            print(f"[GameStateManager] Error loading scenario {chosen_file}: {e}")
            return None

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
                # Grant item through inventory system
                new_state, success, msg = self.grant_item(agent_id, hidden, source=f"Shake:{target}", state=state)
                # Clear the hidden item from container
                new_state = new_state.set_object_property(target, "hidden_inside", None)

                # Get item name for observation
                item_def = self.get_item_definition(hidden)
                item_name = item_def.name if item_def else hidden

                return new_state, HandlerResult(
                    applies=True,
                    state=new_state,
                    observation=f"You shake {target}. A {item_name} falls out! {msg}",
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

        # Collect hidden items (container -> item mappings)
        hidden_items = {}
        for obj_id, props in state.object_properties.items():
            hidden = props.get("hidden_inside")
            if hidden:
                hidden_items[obj_id] = hidden

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
            "hidden_items": hidden_items,  # container -> item_id mappings
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

    def get_item_definition(self, item_id: str) -> Optional[ItemDefinition]:
        """
        Get the ItemDefinition for an item.

        Args:
            item_id: The item ID to look up

        Returns:
            ItemDefinition or None if not found
        """
        item_data = self.state.item_definitions.get(item_id)
        if item_data:
            return ItemDefinition.from_dict(item_data)
        return None

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

        # Check if item is defined
        item_def = self.get_item_definition(item_id)
        item_name = item_def.name if item_def else item_id

        # Add to inventory
        new_state = copy.copy(state)
        new_inv = copy.copy(state.agent_inventory)
        if agent_id not in new_inv:
            new_inv[agent_id] = []
        else:
            new_inv[agent_id] = list(new_inv[agent_id])

        if item_id not in new_inv[agent_id]:
            new_inv[agent_id].append(item_id)
            new_state.agent_inventory = new_inv
            self.state = new_state
            return new_state, True, f"Obtained {item_name}!"
        else:
            return state, False, f"Already have {item_name}."

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

        new_state = copy.copy(state)
        new_inv = copy.copy(state.agent_inventory)

        if agent_id not in new_inv or item_id not in new_inv[agent_id]:
            item_def = self.get_item_definition(item_id)
            item_name = item_def.name if item_def else item_id
            return state, False, f"Don't have {item_name}."

        new_inv[agent_id] = [i for i in new_inv[agent_id] if i != item_id]
        new_state.agent_inventory = new_inv

        self.state = new_state
        item_def = self.get_item_definition(item_id)
        item_name = item_def.name if item_def else item_id
        return new_state, True, f"Removed {item_name}."

    def agent_has_item(self, agent_id: str, item_id: str) -> bool:
        """
        Check if an agent has a specific item.

        Args:
            agent_id: Agent to check
            item_id: Item to check for

        Returns:
            True if agent has the item
        """
        return item_id in self.state.agent_inventory.get(agent_id, [])

    def get_agent_inventory(self, agent_id: str) -> List[ItemDefinition]:
        """
        Get all ItemDefinitions for items in agent's inventory.

        Args:
            agent_id: Agent to get inventory for

        Returns:
            List of ItemDefinition objects
        """
        item_ids = self.state.agent_inventory.get(agent_id, [])
        items = []
        for item_id in item_ids:
            item_def = self.get_item_definition(item_id)
            if item_def:
                items.append(item_def)
        return items

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
            if item.item_type == ItemType.TOOL and item.grants_action:
                line += f" (enables {item.grants_action} action)"
            if item.consumable and item.uses_remaining is not None:
                line += f" [{item.uses_remaining} uses remaining]"
            lines.append(line)

        return "\n".join(lines)

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
