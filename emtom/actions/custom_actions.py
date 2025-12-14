"""
Custom EMTOM Actions.

These actions provide rich interactions that can be affected by mechanics.
Each action has:
- A normal expected behavior
- Can be transformed by mechanics (inverse, remote control, counting, etc.)
- Produces observations that may differ per agent (theory of mind)

To add a new action:
1. Create a class that extends EMTOMAction
2. Decorate it with @register_action("ActionName")
3. The action will automatically be available in exploration, generation, and benchmark
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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


class EMTOMAction(ABC):
    """Base class for EMTOM custom actions."""

    name: str = "base_action"
    description: str = "Base action"

    @abstractmethod
    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """
        Execute the action.

        Args:
            agent_id: The agent performing the action
            target: Optional target for the action
            env_interface: Habitat environment interface
            world_state: Current world state info

        Returns:
            ActionResult with observation and effects
        """
        pass

    def get_available_targets(
        self,
        _env_interface: "EnvironmentInterface",
        _world_state: Dict[str, Any],
    ) -> List[str]:
        """Get valid targets for this action in current state."""
        return []


@register_action("Hide")
class HideAction(EMTOMAction):
    """
    Hide an object from the scene.

    Normal behavior: Removes the object from visible scene graph.
    This is useful for Theory of Mind - other agents won't know the object
    was hidden unless they witnessed it.

    Can be affected by:
    - inverse_state: Hiding actually reveals, revealing actually hides
    - remote_control: Hiding X actually hides Y
    """

    name = "Hide"
    description = "Hide an object so other agents cannot see it. The object is removed from the visible scene."

    def execute(
        self,
        _agent_id: str,
        target: Optional[str],
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to hide.",
            )

        # Check if target exists
        entity_info = world_state.get("entity_details", {}).get(target, {})
        if not entity_info and target not in [e.get("name") for e in world_state.get("entities", [])]:
            return ActionResult(
                success=False,
                observation=f"You don't see {target} here.",
            )

        observation = f"You hide {target}. It is no longer visible to others."

        # Other agents don't see what happened (unless they were watching)
        other_observations = {}
        for other_agent in world_state.get("other_agents", []):
            other_observations[other_agent] = ""  # They observe nothing

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"hidden={target}",
            other_observations=other_observations,
        )

    def get_available_targets(
        self,
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        # Can hide any object (not furniture)
        objects = []
        for entity in world_state.get("entities", []):
            if entity.get("type") == "object":
                objects.append(entity.get("name", entity.get("id")))
        return objects[:10]  # Limit to 10


@register_action("Inspect")
class InspectAction(EMTOMAction):
    """
    Carefully inspect an object to learn about its properties.

    Normal behavior: Reveals detailed information about the object.
    Can be affected by:
    - Mechanics may hide some information
    - Different agents may see different things
    """

    name = "Inspect"
    description = "Carefully examine an object to learn about its properties and current state."

    # Properties to ignore (technical/internal data)
    IGNORED_PROPERTIES = {
        "type", "translation", "rotation", "scale", "sim_handle", "handle",
        "id", "name", "node_type", "category", "semantic_id", "object_id",
        "position", "orientation", "aabb", "bounds", "mesh", "material",
    }

    # Meaningful state prefixes to show
    MEANINGFUL_STATES = {"is_open", "is_powered", "is_clean", "is_filled", "is_locked", "is_on"}

    def execute(
        self,
        _agent_id: str,
        target: Optional[str],
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to inspect.",
            )

        # Get entity info
        entity_info = world_state.get("entity_details", {}).get(target, {})

        if not entity_info:
            observation = f"You look closely at {target}. It appears to be a normal object."
        else:
            states = entity_info.get("states", {})

            # Only show meaningful states (filter out technical data)
            details = []
            for k, v in states.items():
                # Check if this is a meaningful state
                if any(k.startswith(prefix) for prefix in self.MEANINGFUL_STATES):
                    # Format nicely: is_open -> open, is_powered_on -> powered on
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

    def get_available_targets(
        self,
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        # Can inspect any entity
        targets = [e.get("name", e.get("id")) for e in world_state.get("entities", [])]
        return targets[:10]  # Limit to 10


@register_action("WriteMessage")
class WriteMessageAction(EMTOMAction):
    """
    Write a message on a surface or leave a note.

    Normal behavior: Creates a visible message that other agents can read.
    Useful for Theory of Mind - communicating information between agents.

    Can be affected by:
    - inverse_state: Message says the opposite
    - remote_control: Message appears somewhere else
    """

    name = "WriteMessage"
    description = "Write a message or leave a note on a surface. Other agents can read it."

    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        location = target or world_state.get("agent_location", "here")

        # For now, generate a simple exploration-related message
        observation = f"You write a message on {location}: 'I was here - {agent_id}'"

        # Other agents can see the message if they're in the same location
        other_observations = {}
        for other_agent in world_state.get("other_agents", []):
            other_observations[other_agent] = f"You notice a message on {location}."

        return ActionResult(
            success=True,
            observation=observation,
            effect=f"message_written={location}",
            other_observations=other_observations,
        )

    def get_available_targets(
        self,
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        # Can write on furniture surfaces
        targets = []
        for entity in world_state.get("entities", []):
            if entity.get("type") == "furniture":
                targets.append(entity.get("name", entity.get("id")))
        return targets[:10]  # Limit to 10


@register_action("Shake")
class ShakeAction(EMTOMAction):
    """
    Shake an object to reveal hidden items inside.

    Like finding a key in a vase in Zelda - shaking reveals what's hidden.
    The hidden item is "spawned" into the scene and can then be picked up.

    Game state property:
    - hidden_items[object_id] = {"contains": "key_1"}
    - After shaking: item spawns, property is cleared

    Can be affected by:
    - inverse_state: Shaking hides items instead of revealing
    - remote_control: Shaking X reveals item from Y
    """

    name = "Shake"
    description = "Shake an object to reveal any hidden items inside. Items will fall out and can be picked up."

    def execute(
        self,
        _agent_id: str,
        target: Optional[str],
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False,
                observation="You need to specify what to shake.",
            )

        # Check if target exists
        entities = world_state.get("entities", [])
        entity_names = [e.get("name", e.get("id")) for e in entities]
        if target not in entity_names:
            return ActionResult(
                success=False,
                observation=f"You don't see {target} here.",
            )

        # Check if this object has hidden items (stored in game state)
        hidden_items = world_state.get("hidden_items", {})
        hidden_info = hidden_items.get(target, {})
        contained_item = hidden_info.get("contains")

        if contained_item:
            # Item found! Spawn it
            observation = f"You shake {target}. A {contained_item} falls out!"

            # Other agents nearby see the item fall out
            other_observations = {}
            for other_agent in world_state.get("other_agents", []):
                if world_state.get(f"{other_agent}_location") == world_state.get("agent_location"):
                    other_observations[other_agent] = f"You see a {contained_item} fall out of {target}."

            return ActionResult(
                success=True,
                observation=observation,
                effect=f"revealed={contained_item}",
                other_observations=other_observations,
                spawned_items=[contained_item],
            )
        else:
            # Nothing hidden
            observation = f"You shake {target}. Nothing happens."
            return ActionResult(
                success=True,
                observation=observation,
            )

    def get_available_targets(
        self,
        _env_interface: "EnvironmentInterface",
        world_state: Dict[str, Any],
    ) -> List[str]:
        # Can shake small objects (not heavy furniture)
        targets = []
        for entity in world_state.get("entities", []):
            entity_type = entity.get("type", "")
            # Can shake objects, and small furniture like vases, boxes, containers
            if entity_type == "object":
                targets.append(entity.get("name", entity.get("id")))
            elif entity_type == "furniture":
                # Only shakeable furniture (containers, small items)
                furniture_name = entity.get("name", "").lower()
                shakeable_types = ["vase", "box", "container", "basket", "jar", "pot", "bin"]
                if any(t in furniture_name for t in shakeable_types):
                    targets.append(entity.get("name", entity.get("id")))
        return targets[:10]


def get_all_actions() -> Dict[str, EMTOMAction]:
    """Get all registered EMTOM actions (instantiated)."""
    return ActionRegistry.instantiate_all()


# For backwards compatibility - dynamically gets all registered actions
EMTOM_ACTIONS: Dict[str, EMTOMAction] = get_all_actions()


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
        # Get all registered actions from the registry
        self.actions = ActionRegistry.instantiate_all()

    def get_available_actions(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of available custom actions with their targets."""
        available = []

        for name, action in self.actions.items():
            targets = action.get_available_targets(self.env, world_state)
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
        """
        Execute a custom action, applying any relevant mechanics.
        """
        if action_name not in self.actions:
            return ActionResult(
                success=False,
                observation=f"Unknown action: {action_name}",
            )

        action = self.actions[action_name]

        # Execute the base action
        result = action.execute(agent_id, target, self.env, world_state)

        # Apply mechanics transformations
        for mechanic in self.mechanics:
            if hasattr(mechanic, 'transform_action_result'):
                result = mechanic.transform_action_result(
                    action_name, agent_id, target, result, world_state
                )

        return result

    def register_action(self, action: EMTOMAction) -> None:
        """Register a new custom action."""
        self.actions[action.name] = action
