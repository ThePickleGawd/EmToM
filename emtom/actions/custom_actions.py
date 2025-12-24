"""
Custom EMTOM Actions.

These actions extend the partnr Tool interface directly and can be affected by mechanics.
Each action has:
- A normal expected behavior
- Can be transformed by mechanics (inverse, remote control, counting, etc.)
- Produces observations that may differ per agent (theory of mind)

To add a new action:
1. Create a class that extends EMTOMAction
2. Decorate it with @register_action("ActionName")
3. The action will automatically be available in exploration, generation, and benchmark
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from habitat_llm.tools.tool import Tool

from emtom.actions.registry import register_action, ActionRegistry

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


@dataclass
class ActionResult:
    """Result of executing a custom action."""
    success: bool
    observation: str  # What the acting agent observes
    effect: Optional[str] = None  # What actually changed
    other_observations: Dict[str, str] = field(default_factory=dict)  # What other agents observe
    surprise_trigger: Optional[str] = None  # If this should trigger surprise detection
    spawned_items: List[str] = field(default_factory=list)  # Items revealed/spawned by action


class EMTOMAction(Tool):
    """
    Base class for EMTOM custom actions.

    Extends the partnr Tool interface directly so actions can be used
    by agents in the evaluation framework without a wrapper layer.
    """

    action_name: str = "base_action"
    action_description: str = "Base action"

    def __init__(self, agent_uid: int = 0):
        super().__init__(self.action_name, agent_uid)
        self.env_interface: Optional["EnvironmentInterface"] = None
        self._game_manager = None

    def set_environment(self, env_interface: "EnvironmentInterface"):
        """Set the environment interface for this action."""
        self.env_interface = env_interface

    def set_game_manager(self, game_manager):
        """Set the GameStateManager for this action to access game state."""
        self._game_manager = game_manager

    def to(self, device):
        """Compatibility method for device placement."""
        pass

    def get_state_description(self) -> str:
        """Method to get a string describing the state for this action."""
        return "Standing"

    @property
    def description(self) -> str:
        return self.action_description

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE"]

    def _build_world_state(self) -> Dict[str, Any]:
        """Build world state dict for action execution."""
        if not self.env_interface:
            return {}

        world_state = {
            "agent_location": "unknown",
            "rooms": [],
            "entities": [],
            "entity_details": {},
            "other_agents": [],
        }

        try:
            wg = self.env_interface.world_graph.get(self.agent_uid)
            if wg:
                world_state["rooms"] = [r.name for r in wg.get_all_rooms()]
                for node in wg.graph.nodes():
                    if hasattr(node, 'name'):
                        entity = {
                            "name": node.name,
                            "type": getattr(node, 'node_type', 'unknown'),
                            "properties": {},
                            "states": {},
                        }
                        world_state["entities"].append(entity)
                        world_state["entity_details"][node.name] = {
                            "properties": entity["properties"],
                            "states": entity["states"],
                        }
        except Exception:
            pass

        return world_state

    def process_high_level_action(
        self, input_query: str, observations: Any
    ) -> Tuple[Optional[Any], str]:
        """
        Execute the EMTOM action (Tool interface).

        Args:
            input_query: The target for the action (e.g., object name)
            observations: Current observations (unused for EMTOM actions)

        Returns:
            Tuple of (low_level_action, response_text)
        """
        world_state = self._build_world_state()
        result = self.execute(
            agent_id=f"agent_{self.agent_uid}",
            target=input_query,
            world_state=world_state,
        )
        return None, result.observation

    @abstractmethod
    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """
        Execute the action.

        Args:
            agent_id: The agent performing the action
            target: Optional target for the action
            world_state: Current world state info

        Returns:
            ActionResult with observation and effects
        """
        pass

    def get_available_targets(self, world_state: Dict[str, Any]) -> List[str]:
        """Get valid targets for this action in current state."""
        return []


@register_action("Use")
class UseAction(EMTOMAction):
    """
    Use an inventory item.

    Format: Use[item_id, arg1, arg2, ...]
    - Items define their required args via use_args field
    - Keys require 1 arg: Use[item_small_key_1, container]
    - Some items require 0 args: Use[item_flashlight_1]

    Item IDs always start with "item_" prefix.
    """

    action_name = "Use"
    action_description = (
        "Use[item_id, args...]: Use an item from your inventory. "
        "Example: Use[item_small_key_1, cabinet_42] to unlock."
    )

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE", "FURNITURE_INSTANCE"]

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to use. Format: Use[item_id, args...]",
            )

        if not self._game_manager:
            return ActionResult(
                success=False,
                observation="Cannot use: game manager not available.",
            )

        # Parse comma-separated arguments
        parts = [p.strip() for p in target.split(",")]
        item_id = parts[0]
        item_args = parts[1:] if len(parts) > 1 else []

        # Check if item_id looks like an item (has item_ prefix)
        if not item_id.startswith("item_"):
            return ActionResult(
                success=False,
                observation=f"'{item_id}' is not an item. Item IDs start with 'item_' (e.g., item_small_key_1).",
            )

        # Check if agent has the item
        if not self._game_manager.agent_has_item(agent_id, item_id):
            return ActionResult(
                success=False,
                observation=f"You don't have {item_id}.",
            )

        # Get item to check if it's a key (special unlock handling)
        item = self._game_manager.get_item(item_id)
        if item and "key" in item_id.lower() and item_args:
            # Key usage - delegate to unlock helper
            container = item_args[0]
            from emtom.actions.tool_wrapper import try_unlock_with_key
            success, message = try_unlock_with_key(
                self._game_manager, agent_id, item_id, container
            )
            return ActionResult(
                success=success,
                observation=message,
                effect=f"unlock={container}" if success else None,
            )

        # Generic item use - delegate to manager's use_item (validates use_args)
        success, msg = self._game_manager.use_item(agent_id, item_id, item_args)
        return ActionResult(
            success=success,
            observation=msg,
            effect=f"used={item_id}" if success else None,
        )

    def get_available_targets(self, world_state: Dict[str, Any]) -> List[str]:
        targets = [e.get("name", e.get("id")) for e in world_state.get("entities", [])]
        return targets[:10]


@register_action("Search")
class SearchAction(EMTOMAction):
    """
    Search a location or container for hidden items.

    Unlike Shake (which may require physical interaction), Search is a careful
    examination that can find hidden items within furniture or containers.

    Adds found items to the agent's inventory.
    """

    action_name = "Search"
    action_description = (
        "Search[target]: Carefully search a location or container for hidden items. "
        "May find items that aren't visible at first glance."
    )

    @property
    def argument_types(self) -> List[str]:
        return ["FURNITURE_INSTANCE", "OBJECT_INSTANCE"]

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to search.",
            )

        if not self._game_manager:
            return ActionResult(
                success=False,
                observation="Cannot search: no game manager available.",
            )

        state = self._game_manager.get_state()

        # Check if container is locked
        is_locked = state.get_object_property(target, "is_locked", False)
        if is_locked:
            return ActionResult(
                success=False,
                observation=f"You try to search {target}, but it's locked. Unlock it first.",
            )

        # Check if container is closed (only for openable containers)
        # is_open defaults to None for non-openable objects
        is_open = state.get_object_property(target, "is_open", None)
        if is_open is False:
            return ActionResult(
                success=False,
                observation=f"You try to search {target}, but it's closed. Open it first.",
            )

        hidden_items = state.get_object_property(target, "hidden_items", [])

        if hidden_items:
            new_state = state
            found_names = []
            spawned = []

            for item_id in hidden_items:
                new_state, success, msg = self._game_manager.grant_item(
                    agent_id, item_id, source=f"Search:{target}", state=new_state
                )
                item = self._game_manager.get_item(item_id)
                item_name = item.name if item else item_id
                found_names.append(item_name)
                spawned.append(item_id)

            # Clear hidden items from container
            new_state = new_state.set_object_property(target, "hidden_items", [])
            self._game_manager.set_state(new_state)

            items_text = ", ".join(found_names)
            return ActionResult(
                success=True,
                observation=f"You search {target} carefully. You find: {items_text}!",
                effect=f"found_items={','.join(spawned)}",
                spawned_items=spawned,
            )

        return ActionResult(
            success=True,
            observation=f"You search {target} carefully but find nothing hidden.",
        )

    def get_available_targets(self, world_state: Dict[str, Any]) -> List[str]:
        targets = [
            e.get("name", e.get("id"))
            for e in world_state.get("entities", [])
            if e.get("type") in ("furniture", "FURNITURE") or "cabinet" in e.get("name", "").lower()
        ]
        return targets[:10]


@register_action("Inspect")
class InspectAction(EMTOMAction):
    """
    Carefully inspect an object to learn about its properties.

    Normal behavior: Reveals detailed information about the object.
    Can be affected by:
    - Mechanics may hide some information
    - Different agents may see different things
    """

    action_name = "Inspect"
    action_description = (
        "Inspect[object]: Carefully examine an object to learn about its "
        "properties and current state."
    )

    MEANINGFUL_STATES = {"is_open", "is_powered", "is_clean", "is_filled", "is_locked", "is_on"}

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE", "FURNITURE_INSTANCE"]

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to inspect.",
            )

        entity_info = world_state.get("entity_details", {}).get(target, {})

        if not entity_info:
            observation = f"You look closely at {target}. It appears to be a normal object."
        else:
            states = entity_info.get("states", {})
            details = []
            for k, v in states.items():
                if any(k.startswith(prefix) for prefix in self.MEANINGFUL_STATES):
                    readable_name = k.replace("is_", "").replace("_", " ")
                    state_word = "yes" if v else "no"
                    details.append(f"{readable_name}: {state_word}")

            if details:
                observation = f"You examine {target} closely. You observe: {', '.join(details)}."
            else:
                observation = f"You examine {target}. It appears normal with no unusual properties."

        return ActionResult(
            success=True,
            observation=observation,
        )

    def get_available_targets(self, world_state: Dict[str, Any]) -> List[str]:
        targets = [e.get("name", e.get("id")) for e in world_state.get("entities", [])]
        return targets[:10]


class DynamicItemTool(EMTOMAction):
    """
    A tool dynamically created from an inventory item.

    TOOL-type items grant new actions when obtained. This class creates
    those actions based on the item definition.

    Not registered in the action registry since these are created dynamically.

    Room-based restrictions:
        If allowed_rooms is set, the tool only works when the agent is in
        one of the specified rooms. This enables Theory of Mind tasks where
        agents must navigate to specific locations to use certain abilities.
    """

    def __init__(
        self,
        name: str,
        description: str,
        item_id: str,
        argument_types: Optional[List[str]] = None,
        consumable: bool = False,
        agent_uid: int = 0,
        allowed_rooms: Optional[List[str]] = None,
    ):
        self.action_name = name
        self.action_description = description
        self._item_id = item_id
        self._argument_types = argument_types or ["OBJECT_INSTANCE", "FURNITURE_INSTANCE"]
        self._consumable = consumable
        self._allowed_rooms = allowed_rooms  # None means no restriction
        super().__init__(agent_uid)

    @property
    def argument_types(self) -> List[str]:
        return self._argument_types

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not self._game_manager:
            return ActionResult(
                success=False,
                observation="Tool not properly configured.",
            )

        # Check agent still has the item
        if not self._game_manager.agent_has_item(agent_id, self._item_id):
            item = self._game_manager.get_item(self._item_id)
            item_name = item.name if item else self._item_id
            return ActionResult(
                success=False,
                observation=f"You no longer have the {item_name}.",
            )

        # Execute the tool's effect
        result = self._execute_tool_effect(agent_id, target, world_state)

        # If consumable and success, decrement uses
        if self._consumable and result.success:
            self._decrement_uses(agent_id)

        return result

    def _execute_tool_effect(
        self,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """
        Execute tool-specific effect based on action name.

        Currently supported:
        - Communicate: Send a message via radio to other agents

        Room restrictions:
        - If allowed_rooms is set, checks agent's current room before execution
        - Returns failure with helpful message if not in allowed room
        """
        item = self._game_manager.get_item(self._item_id)
        item_name = item.name if item else self._item_id
        action = self.action_name

        if not target:
            return ActionResult(
                success=False,
                observation=f"You need to specify a message for {action}.",
            )

        # Check room restrictions if set
        if self._allowed_rooms is not None:
            agent_room = self._game_manager.state.agent_rooms.get(agent_id)
            if agent_room not in self._allowed_rooms:
                # Provide helpful message about where tool works
                if len(self._allowed_rooms) == 1:
                    rooms_hint = f"the {self._allowed_rooms[0]}"
                else:
                    rooms_hint = f"one of: {', '.join(self._allowed_rooms)}"
                return ActionResult(
                    success=False,
                    observation=f"The {item_name} doesn't work here. You need to be in {rooms_hint} to use it.",
                    effect=f"tool_blocked_by_room={agent_room}",
                )

        # Communicate: Send a radio message to other agents
        if action == "Communicate":
            return ActionResult(
                success=True,
                observation=f"You speak into the radio: \"{target}\"",
                effect=f"communicated={target}",
            )

        # Default fallback for unknown tool types
        return ActionResult(
            success=True,
            observation=f"You use the {item_name}: {target}",
            effect=f"used_{self._item_id}",
        )

    def _decrement_uses(self, agent_id: str) -> None:
        """Decrement remaining uses of a consumable item."""
        item = self._game_manager.get_item(self._item_id)
        if not item or item.uses_remaining is None:
            return

        new_uses = item.uses_remaining - 1

        if new_uses <= 0:
            # Remove item from inventory when uses exhausted
            self._game_manager.remove_item(agent_id, self._item_id)
        else:
            # Update remaining uses in state
            item.uses_remaining = new_uses
            self._game_manager.state.item_definitions[self._item_id] = item.to_dict()


def get_all_actions() -> Dict[str, EMTOMAction]:
    """Get all registered EMTOM actions (instantiated)."""
    return ActionRegistry.instantiate_all()


# For backwards compatibility - dynamically gets all registered actions
EMTOM_ACTIONS: Dict[str, EMTOMAction] = get_all_actions()


def get_emtom_tools(agent_uid: int = 0) -> Dict[str, EMTOMAction]:
    """
    Get all EMTOM actions instantiated for a given agent.

    This is the main entry point for getting actions to use with agents.

    Args:
        agent_uid: The agent ID to create actions for

    Returns:
        Dict mapping action names to action instances
    """
    from emtom.actions.registry import get_registry
    actions = {}
    for name, action_cls in get_registry().items():
        actions[name] = action_cls(agent_uid=agent_uid)
    return actions


class EMTOMActionExecutor:
    """
    Executor for EMTOM custom actions.

    Integrates custom actions with the Habitat environment and mechanics system.
    Uses the ActionRegistry to automatically discover all registered actions.
    """

    def __init__(
        self,
        env_interface: "EnvironmentInterface",
        mechanics: Optional[List[Any]] = None,
    ):
        self.env = env_interface
        self.mechanics = mechanics or []
        self.actions = ActionRegistry.instantiate_all()

    def get_available_actions(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of available custom actions with their targets."""
        available = []
        for name, action in self.actions.items():
            targets = action.get_available_targets(world_state)
            available.append({
                "name": name,
                "description": action.description,
                "targets": targets,
            })
        return available

    def execute(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """Execute a custom action, applying any relevant mechanics."""
        if action_name not in self.actions:
            return ActionResult(
                success=False,
                observation=f"Unknown action: {action_name}",
            )

        action = self.actions[action_name]
        result = action.execute(agent_id, target, world_state)

        for mechanic in self.mechanics:
            if hasattr(mechanic, 'transform_action_result'):
                result = mechanic.transform_action_result(
                    action_name, agent_id, target, result, world_state
                )

        return result

    def register_action(self, action: EMTOMAction) -> None:
        """Register a new custom action."""
        self.actions[action.name] = action
