"""
EMTOM Tools - partnr-compatible wrappers for EMTOM custom actions.

These tools implement the partnr Tool interface so they can be used by agents
in the DecentralizedEvaluationRunner.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from habitat_llm.tools.tool import Tool

from emtom.actions import ActionRegistry, ActionResult

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


class EMTOMTool(Tool):
    """
    Base class for EMTOM tools that wrap custom actions.

    These tools provide a partnr-compatible interface for EMTOM actions,
    allowing them to be used by agents in the evaluation framework.
    """

    action_name: str = "base"

    def __init__(self, name_arg: str = None, agent_uid_arg: int = 0):
        super().__init__(name_arg or self.action_name, agent_uid_arg)
        self.env_interface: Optional["EnvironmentInterface"] = None
        self._action = None
        self._game_manager = None

    def set_environment(self, env_interface: "EnvironmentInterface"):
        """Set the environment interface for this tool."""
        self.env_interface = env_interface

    def set_game_manager(self, game_manager):
        """Set the GameStateManager for this tool to access game state."""
        self._game_manager = game_manager

    def to(self, device):
        """Compatibility method for device placement."""
        pass

    def get_state_description(self) -> str:
        """Method to get a string describing the state for this tool."""
        # EMTOM tools are instant actions, so just return standing
        return "Standing"

    def _get_action(self):
        """Get the EMTOM action instance."""
        if self._action is None:
            self._action = ActionRegistry.instantiate(self.action_name)
        return self._action

    def _build_world_state(self) -> Dict[str, Any]:
        """Build world state dict for action execution."""
        if not self.env_interface:
            return {}

        # Get basic info from environment
        world_state = {
            "agent_location": "unknown",
            "rooms": [],
            "entities": [],
            "entity_details": {},
            "other_agents": [],
            "hidden_items": {},
        }

        try:
            # Try to get world graph info
            wg = self.env_interface.world_graph.get(self.agent_uid)
            if wg:
                # Get rooms
                world_state["rooms"] = [r.name for r in wg.get_all_rooms()]

                # Get entities (objects and furniture)
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

        # Get hidden_items from GameStateManager if available
        if self._game_manager:
            try:
                state = self._game_manager.get_state()
                # Build hidden_items dict from object_properties
                for obj_id, props in state.object_properties.items():
                    if "hidden_inside" in props:
                        world_state["hidden_items"][obj_id] = {
                            "contains": props["hidden_inside"]
                        }
            except Exception:
                pass

        return world_state

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE"]

    def process_high_level_action(
        self, input_query: str, observations: Any
    ) -> Tuple[Optional[Any], str]:
        """
        Execute the EMTOM action.

        Args:
            input_query: The target for the action (e.g., object name)
            observations: Current observations (unused for EMTOM actions)

        Returns:
            Tuple of (low_level_action, response_text)
        """
        action = self._get_action()
        world_state = self._build_world_state()

        # Execute the action
        result: ActionResult = action.execute(
            agent_id=f"agent_{self.agent_uid}",
            target=input_query,
            env_interface=self.env_interface,
            world_state=world_state,
        )

        # Return result as text observation
        return None, result.observation


class HideTool(EMTOMTool):
    """Tool for hiding objects from the scene."""

    action_name = "Hide"

    @property
    def description(self) -> str:
        return (
            "Hide[object]: Hide an object so other agents cannot see it. "
            "The object is removed from the visible scene. Useful for testing "
            "what other agents know (theory of mind). Example: Hide[apple_1]"
        )

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE"]


class InspectTool(EMTOMTool):
    """Tool for inspecting objects to learn their properties."""

    action_name = "Inspect"

    @property
    def description(self) -> str:
        return (
            "Inspect[object]: Carefully examine an object to learn about its "
            "properties and current state. Returns detailed information about "
            "the object. Example: Inspect[cabinet_57]"
        )

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE", "FURNITURE_INSTANCE"]


class WriteMessageTool(EMTOMTool):
    """Tool for writing messages on surfaces."""

    action_name = "WriteMessage"

    @property
    def description(self) -> str:
        return (
            "WriteMessage[furniture]: Write a message or leave a note on a "
            "surface. Other agents can read it. Useful for communicating "
            "information between agents. Example: WriteMessage[table_1]"
        )

    @property
    def argument_types(self) -> List[str]:
        return ["FURNITURE_INSTANCE"]


class ShakeTool(EMTOMTool):
    """Tool for shaking objects to reveal hidden items."""

    action_name = "Shake"

    @property
    def description(self) -> str:
        return (
            "Shake[object]: Shake an object to reveal any hidden items inside. "
            "Items will fall out and can be picked up. Like finding a key in "
            "a vase. Example: Shake[vase_1]"
        )

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE", "FURNITURE_INSTANCE"]


# Registry of EMTOM tools
EMTOM_TOOLS = {
    "Hide": HideTool,
    "Inspect": InspectTool,
    "WriteMessage": WriteMessageTool,
    "Shake": ShakeTool,
}


def get_emtom_tools(agent_uid: int = 0) -> Dict[str, EMTOMTool]:
    """
    Get all EMTOM tools instantiated for a given agent.

    Args:
        agent_uid: The agent ID to create tools for

    Returns:
        Dict mapping tool names to tool instances
    """
    return {
        name: cls(agent_uid_arg=agent_uid)
        for name, cls in EMTOM_TOOLS.items()
    }
