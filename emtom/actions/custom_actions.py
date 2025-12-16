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
    Use an item or interact with an object.

    This is the generic interaction action. What happens depends on the item's
    defined behavior (to be configured separately).

    Can be affected by mechanics (inverse_state, remote_control, etc.)
    """

    action_name = "Use"
    action_description = (
        "Use[object]: Use an item or interact with an object. "
        "The effect depends on what the item does. Example: Use[key_1]"
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
                observation="You need to specify what to use.",
            )

        entities = world_state.get("entities", [])
        entity_names = [e.get("name", e.get("id")) for e in entities]
        if target not in entity_names:
            return ActionResult(
                success=False,
                observation=f"You don't see {target} here.",
            )

        # TODO: Look up item behavior definition and execute accordingly
        observation = f"You use {target}."

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"used={target}",
        )

    def get_available_targets(self, world_state: Dict[str, Any]) -> List[str]:
        targets = [e.get("name", e.get("id")) for e in world_state.get("entities", [])]
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
        "properties and current state. Returns detailed information about "
        "the object. Example: Inspect[cabinet_57]"
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
