"""
Stateless Mechanic Handlers.

Each handler is a pure function that:
- Takes (action, agent, target, state)
- Returns (applies: bool, new_state, result)

Handlers don't store any state - all state lives in EMTOMGameState.
"""

from typing import Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import copy

from emtom.state.game_state import EMTOMGameState


@dataclass
class HandlerResult:
    """Result from a mechanic handler."""
    applies: bool  # Whether this mechanic applied
    state: EMTOMGameState
    observation: str
    success: bool
    effects: list
    surprise_trigger: Optional[str] = None
    # What action should actually be executed (for transforms like inverse_state)
    actual_action: Optional[str] = None
    actual_target: Optional[str] = None
    # Whether the action should be blocked (not executed in Habitat)
    blocked: bool = False
    # Which mechanic type was applied (for event logging)
    mechanic_type: Optional[str] = None


# Type alias for handler functions
MechanicHandler = Callable[
    [str, str, Optional[str], EMTOMGameState],
    HandlerResult
]


# =============================================================================
# Mechanic Definitions
# =============================================================================

MECHANIC_INFO = {
    "inverse_state": {
        "description": "Actions have opposite effects (open becomes close)",
        "category": "state_transform",
        "setup_keys": ["trigger_object"],
        "agent_observation": "You try to open {trigger}, but it closes instead!",
        "tom_use": "One agent discovers the inversion, must warn the other",
        "example_binding": {"mechanic_type": "inverse_state", "trigger_object": "drawer_52"},
    },
    "remote_control": {
        "description": "Acting on trigger affects a remote target",
        "category": "hidden_mapping",
        "setup_keys": ["trigger_object", "target_object", "target_state"],
        "agent_observation": "You hear something happen to {target}! (It opened)",
        "tom_use": "Agent A flips switch, Agent B's door unlocks - they must communicate",
        "example_binding": {"mechanic_type": "remote_control", "trigger_object": "lamp_12", "target_object": "cabinet_45", "target_state": "is_open"},
        "recommended_for_tom": True,
    },
    "conditional_unlock": {
        "description": "Object blocked until prerequisite action or item obtained",
        "category": "conditional",
        "setup_keys": ["trigger_object", "prerequisite_object", "requires_item"],
        "agent_observation": "You try to open {trigger}, but it won't budge. Something seems to be blocking it.",
        "tom_use": "One agent knows the prerequisite, other is stuck",
        "example_binding": {"mechanic_type": "conditional_unlock", "trigger_object": "chest_45", "prerequisite_object": "lever_12"},
        "alt_example": {"mechanic_type": "conditional_unlock", "trigger_object": "vault_1", "requires_item": "item_magic_orb_1"},
        "recommended_for_tom": True,
    },
    "state_mirroring": {
        "description": "Two objects always have the same state (open one, both open)",
        "category": "hidden_mapping",
        "setup_keys": ["trigger_object", "target_object", "target_state"],
        "agent_observation": "{target} opens too!",
        "tom_use": "Agents in different rooms see linked effects",
        "example_binding": {"mechanic_type": "state_mirroring", "trigger_object": "drawer_1", "target_object": "drawer_2", "target_state": "is_open"},
    },
    # === Theory of Mind Mechanics ===
    "location_change": {
        "description": "Object is moved while an agent is absent (classic false belief)",
        "category": "belief_tracking",
        "setup_keys": ["object_id", "original_location", "new_location", "moving_agent"],
        "agent_observation": "You move {object} from {original} to {new_location}.",
        "tom_use": "Agent A moves object while B is away. B returns with false belief about location.",
        "example_binding": {"mechanic_type": "location_change", "object_id": "apple_3", "original_location": "basket_1", "new_location": "box_2", "moving_agent": "agent_0"},
        "recommended_for_tom": True,
    },
    "container_swap": {
        "description": "Contents of two containers are secretly swapped",
        "category": "belief_tracking",
        "setup_keys": ["container_a", "container_b", "swapping_agent"],
        "agent_observation": "You swap the contents of {container_a} and {container_b}.",
        "tom_use": "Agent A swaps contents while B is away. B has false belief about which container holds what.",
        "example_binding": {"mechanic_type": "container_swap", "container_a": "basket_1", "container_b": "box_2", "swapping_agent": "agent_0"},
        "recommended_for_tom": True,
    },
    "state_change_unseen": {
        "description": "Object state changes while an agent cannot observe it",
        "category": "belief_tracking",
        "setup_keys": ["object_id", "property", "new_value", "changing_agent"],
        "agent_observation": "You change {object}'s {property} to {new_value}.",
        "tom_use": "Agent A changes state (locks door) while B is away. B has false belief about state.",
        "example_binding": {"mechanic_type": "state_change_unseen", "object_id": "door_5", "property": "is_locked", "new_value": True, "changing_agent": "agent_0"},
        "recommended_for_tom": True,
    },
}


def get_mechanic_info(name: str) -> Dict[str, Any]:
    """Get info about a mechanic."""
    return MECHANIC_INFO.get(name, {})


def get_mechanics_for_task_generation() -> str:
    """
    Get comprehensive mechanic descriptions for task generation prompts.

    Dynamically generates from MECHANIC_INFO, including:
    - What each mechanic does
    - What agents observe when it triggers
    - The exact JSON binding format with examples

    Returns:
        Formatted string for LLM prompts
    """
    import json

    lines = ["Available mechanics for creating puzzle complexity:\n"]
    recommended = []

    for mech_name, info in MECHANIC_INFO.items():
        lines.append(f"## {mech_name}")
        lines.append(f"**Effect**: {info['description']}")

        if info.get("agent_observation"):
            lines.append(f"**Agent sees**: \"{info['agent_observation']}\"")

        if info.get("tom_use"):
            lines.append(f"**ToM use**: {info['tom_use']}")

        if info.get("example_binding"):
            lines.append("```json")
            lines.append(json.dumps(info["example_binding"]))
            lines.append("```")

        # Show alternative example if present (e.g., conditional_unlock has two forms)
        if info.get("alt_example"):
            lines.append("Or:")
            lines.append("```json")
            lines.append(json.dumps(info["alt_example"]))
            lines.append("```")

        if info.get("recommended_for_tom"):
            recommended.append(mech_name)

        lines.append("")  # blank line between mechanics

    if recommended:
        lines.append(f"**Best for ToM**: {', '.join(recommended)} (create cross-agent dependencies)")

    return "\n".join(lines)


def list_mechanics() -> list:
    """List all available mechanics."""
    return list(MECHANIC_INFO.keys())


# =============================================================================
# Helper Functions
# =============================================================================

def no_effect(state: EMTOMGameState) -> HandlerResult:
    """Return a non-applying result."""
    return HandlerResult(
        applies=False,
        state=state,
        observation="",
        success=True,
        effects=[],
    )


# =============================================================================
# Mechanic Handlers
# =============================================================================

def handle_inverse_state(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Inverse State: Actions have opposite effects.

    Setup in state:
        state.inverse_objects = {"door_1", "drawer_2"}
    """
    if not target or target not in state.inverse_objects:
        return no_effect(state)

    # Map both lowercase and capitalized action names
    inverse_map = {
        "open": "close",
        "close": "open",
        "turn_on": "turn_off",
        "turn_off": "turn_on",
        "lock": "unlock",
        "unlock": "lock",
        "Open": "Close",
        "Close": "Open",
    }

    action_lower = action_name.lower()
    if action_lower not in ["open", "close", "turn_on", "turn_off", "lock", "unlock"]:
        return no_effect(state)

    # Get inverted action, preserving case
    inverted = inverse_map.get(action_name, inverse_map.get(action_lower, action_name))
    # Capitalize for Habitat tools
    inverted_capitalized = inverted.capitalize() if inverted.islower() else inverted

    return HandlerResult(
        applies=True,
        state=state,
        observation=f"You try to {action_lower} {target}, but it {inverted.lower()}s instead!",
        success=True,
        effects=[f"inverted={action_name}->{inverted}"],
        surprise_trigger=f"{target} did the opposite of expected",
        actual_action=inverted_capitalized,
        actual_target=target,
    )


def handle_remote_control(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Remote Control: Acting on trigger affects a remote target.

    The physical action is executed on the original target in Habitat.
    The remote effect is applied to game state (changes remote target's property).

    Only triggers on state-changing actions (Open, Close, etc.), not Navigate, Pick, etc.

    Setup in state:
        state.remote_mappings = {"switch_1": ("light_1", "is_open")}
        - When you interact with switch_1, light_1's is_open property toggles
    """
    if not target or target not in state.remote_mappings:
        return no_effect(state)

    # Only trigger on state-changing actions, not movement/manipulation
    action_lower = action_name.lower()
    state_changing_actions = {"open", "close", "turn_on", "turn_off", "lock", "unlock", "use"}
    if action_lower not in state_changing_actions:
        return no_effect(state)

    remote_target, remote_property = state.remote_mappings[target]
    action_lower = action_name.lower()

    # Determine new value based on action (toggle or set based on action type)
    # For open/close, turn_on/turn_off: derive from action
    # Otherwise: toggle the current value
    current_value = state.get_object_property(remote_target, remote_property, False)

    if action_lower in ("open", "turn_on", "unlock"):
        new_value = True
    elif action_lower in ("close", "turn_off", "lock"):
        new_value = False
    else:
        # Toggle for other actions
        new_value = not current_value

    # Apply the state change to the remote target
    new_state = state.set_object_property(remote_target, remote_property, new_value)

    # Describe what happened
    if remote_property == "is_open":
        effect_desc = "opened" if new_value else "closed"
    elif remote_property == "is_on":
        effect_desc = "turned on" if new_value else "turned off"
    else:
        effect_desc = f"changed to {new_value}"

    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"You hear something happen to {remote_target}! (It {effect_desc})",
        success=True,
        effects=[f"remote_effect={remote_target}.{remote_property}={new_value}"],
        surprise_trigger=f"{target} affected {remote_target} remotely",
        # Keep original action and target - don't redirect the physical action
        actual_action=None,
        actual_target=None,
    )


def handle_conditional_unlock(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Conditional Unlock: Object only works after prerequisite action or item.

    Setup in state (object prerequisite - must interact with prereq first):
        state.object_properties["chest_1"]["prerequisite"] = "lever_1"
        state.object_properties["lever_1"]["unlocks"] = "chest_1"

    Setup in state (item prerequisite - must have item in inventory):
        state.object_properties["door_1"]["requires_item"] = "magic_orb_1"
        When agent with magic_orb_1 interacts, door_1 unlocks.
    """
    if not target:
        return no_effect(state)

    # Actions that should be blocked by locks (interaction actions, not movement)
    LOCKABLE_ACTIONS = {"Open", "Close", "Search"}
    action_lower = action_name.lower()

    # Check if target is locked by item prerequisite (must have item in inventory)
    required_item = state.get_object_property(target, "requires_item")
    if required_item and target not in state.unlocked_targets:
        # Check if agent has the required item
        agent_inventory = state.agent_inventory.get(agent_id, [])
        if required_item not in agent_inventory:
            # Only block interactive actions
            if action_name in LOCKABLE_ACTIONS:
                return HandlerResult(
                    applies=True,
                    state=state,
                    observation=f"You try to {action_lower} {target}, but it won't budge. You sense you need something special.",
                    success=False,
                    effects=[],
                    surprise_trigger=f"{target} requires a special item",
                    blocked=True,
                )
        else:
            # Agent has the item - unlock the target
            new_unlocked = copy.copy(state.unlocked_targets)
            new_unlocked.add(target)
            new_state = copy.copy(state)
            new_state.unlocked_targets = new_unlocked

            return HandlerResult(
                applies=True,
                state=new_state,
                observation=f"The {required_item} glows and {target} unlocks!",
                success=True,
                effects=[f"item_unlocked={target}"],
                surprise_trigger=f"{required_item} unlocked {target}",
            )

    # Check if target is locked by object prerequisite (must interact with prereq first)
    prerequisite = state.get_object_property(target, "prerequisite")
    if prerequisite and target not in state.unlocked_targets:
        # Only block interactive actions
        if action_name in LOCKABLE_ACTIONS:
            return HandlerResult(
                applies=True,
                state=state,
                observation=f"You try to {action_lower} {target}, but it won't budge. Something seems to be blocking it.",
                success=False,
                effects=[],
                surprise_trigger=f"{target} is locked by unknown prerequisite",
                blocked=True,
            )

    # Check if this action unlocks something
    unlocks = state.get_object_property(target, "unlocks")
    if unlocks:
        new_unlocked = copy.copy(state.unlocked_targets)
        new_unlocked.add(unlocks)
        new_state = copy.copy(state)
        new_state.unlocked_targets = new_unlocked

        return HandlerResult(
            applies=True,
            state=new_state,
            observation=f"You hear a click somewhere!",
            success=True,
            effects=[f"unlocked={unlocks}"],
            surprise_trigger=f"Something was unlocked",
        )

    return no_effect(state)


def handle_state_mirroring(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    State Mirroring: Two objects always have the same state.

    Setup in state:
        state.mirror_pairs = [("drawer_1", "drawer_2", "is_open")]
    """
    if not target:
        return no_effect(state)

    # Find mirror partner and the mirrored property
    partner = None
    mirror_property = None
    for obj_a, obj_b, prop in state.mirror_pairs:
        if obj_a == target:
            partner = obj_b
            mirror_property = prop
            break
        elif obj_b == target:
            partner = obj_a
            mirror_property = prop
            break

    if not partner or not mirror_property:
        return no_effect(state)

    # Determine new value based on action
    action_lower = action_name.lower()
    if action_lower in ("open", "turn_on", "unlock"):
        new_value = True
    elif action_lower in ("close", "turn_off", "lock"):
        new_value = False
    else:
        # Toggle for other actions
        current_value = state.get_object_property(partner, mirror_property, False)
        new_value = not current_value

    # Apply the same state change to the partner
    new_state = state.set_object_property(partner, mirror_property, new_value)

    # Describe what happened
    if mirror_property == "is_open":
        effect_desc = "opens" if new_value else "closes"
    elif mirror_property == "is_on":
        effect_desc = "turns on" if new_value else "turns off"
    else:
        effect_desc = f"changes to {new_value}"

    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"{partner} {effect_desc} too!",
        success=True,
        effects=[f"mirrored={partner}.{mirror_property}={new_value}"],
        surprise_trigger=f"{target} and {partner} changed together",
    )


# =============================================================================
# Theory of Mind Mechanic Handlers
# =============================================================================

def handle_location_change(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Location Change: Track when objects are moved while agents are absent.

    This is the classic Sally-Anne false belief scenario:
    - Agent A moves object from location X to Y
    - Agent B (absent) still believes object is at X

    Setup in state via mechanic_bindings:
        {"mechanic_type": "location_change", "object_id": "apple_3",
         "original_location": "basket_1", "new_location": "box_2"}

    The mechanic triggers on Place actions and tracks:
    - Which agents were NOT in the same room (absent agents)
    - The original and new locations
    - When the move happened

    Absent agents can query their "belief" about object location.
    """
    # Only trigger on Place actions
    if action_name.lower() != "place" or not target:
        return no_effect(state)

    # Check if this object is configured for location_change tracking
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") != "location_change":
            continue
        if binding.get("object_id") != target:
            continue

        # This object should be tracked for location changes
        original_loc = binding.get("original_location", "unknown")
        new_loc = binding.get("new_location", "unknown")

        # Determine which agents are absent (not in same room as moving agent)
        moving_agent_room = state.agent_rooms.get(agent_id, "unknown")
        absent_agents = []
        for other_agent, room in state.agent_rooms.items():
            if other_agent != agent_id and room != moving_agent_room:
                absent_agents.append(other_agent)

        # Record the location change
        new_state = copy.copy(state)
        new_state.location_changes = copy.copy(state.location_changes)
        new_state.location_changes[target] = {
            "original_location": original_loc,
            "new_location": new_loc,
            "moved_by": agent_id,
            "moved_at_step": state.current_step,
            "absent_agents": absent_agents,
        }

        obs_parts = [f"You move {target} from {original_loc} to {new_loc}."]
        if absent_agents:
            obs_parts.append(f"({', '.join(absent_agents)} didn't see this.)")

        return HandlerResult(
            applies=True,
            state=new_state,
            observation=" ".join(obs_parts),
            success=True,
            effects=[f"location_change={target}:{original_loc}->{new_loc}"],
            surprise_trigger=f"{target} moved while agents absent",
            mechanic_type="location_change",
        )

    return no_effect(state)


def handle_container_swap(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Container Swap: Track when contents of containers are swapped.

    Extends false belief to container contents:
    - Agent A swaps contents of container X and Y
    - Agent B (absent) believes original contents are in original containers

    Setup in state via mechanic_bindings:
        {"mechanic_type": "container_swap", "container_a": "basket_1",
         "container_b": "box_2", "trigger_action": "swap"}

    The mechanic triggers on a special "Swap" action or UseItem with swap item.
    """
    # Trigger on UseItem with a swap-type item or special Swap action
    action_lower = action_name.lower()
    if action_lower not in ("useitem", "swap"):
        return no_effect(state)

    # Check for container_swap bindings
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") != "container_swap":
            continue

        container_a = binding.get("container_a")
        container_b = binding.get("container_b")

        # Check if target matches either container
        if target not in (container_a, container_b):
            continue

        # Determine which agents are absent
        swapping_agent_room = state.agent_rooms.get(agent_id, "unknown")
        absent_agents = []
        for other_agent, room in state.agent_rooms.items():
            if other_agent != agent_id and room != swapping_agent_room:
                absent_agents.append(other_agent)

        # Record the swap
        new_state = copy.copy(state)
        new_state.container_swaps = copy.copy(state.container_swaps)

        # Mark both containers as swapped
        swap_record = {
            "swapped_with": container_b if target == container_a else container_a,
            "swapped_by": agent_id,
            "swapped_at_step": state.current_step,
            "absent_agents": absent_agents,
        }
        new_state.container_swaps[container_a] = {**swap_record, "swapped_with": container_b}
        new_state.container_swaps[container_b] = {**swap_record, "swapped_with": container_a}

        obs_parts = [f"You swap the contents of {container_a} and {container_b}."]
        if absent_agents:
            obs_parts.append(f"({', '.join(absent_agents)} didn't see this.)")

        return HandlerResult(
            applies=True,
            state=new_state,
            observation=" ".join(obs_parts),
            success=True,
            effects=[f"container_swap={container_a}<->{container_b}"],
            surprise_trigger=f"Contents of {container_a} and {container_b} swapped",
            mechanic_type="container_swap",
        )

    return no_effect(state)


def handle_state_change_unseen(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    State Change Unseen: Track when object state changes while agents can't observe.

    - Agent A changes state of object (locks door, closes cabinet)
    - Agent B (absent or can't see) still believes old state

    Setup in state via mechanic_bindings:
        {"mechanic_type": "state_change_unseen", "object_id": "door_5",
         "tracked_properties": ["is_locked", "is_open"]}

    Triggers on state-changing actions (Open, Close, Lock, Unlock).
    """
    if not target:
        return no_effect(state)

    # Only track state-changing actions
    action_lower = action_name.lower()
    state_actions = {"open", "close", "lock", "unlock", "turn_on", "turn_off"}
    if action_lower not in state_actions:
        return no_effect(state)

    # Check for state_change_unseen bindings
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") != "state_change_unseen":
            continue
        if binding.get("object_id") != target:
            continue

        # Determine the property being changed
        if action_lower in ("open", "close"):
            prop = "is_open"
            new_value = action_lower == "open"
        elif action_lower in ("lock", "unlock"):
            prop = "is_locked"
            new_value = action_lower == "lock"
        elif action_lower in ("turn_on", "turn_off"):
            prop = "is_on"
            new_value = action_lower == "turn_on"
        else:
            continue

        # Check if this property is tracked
        tracked = binding.get("tracked_properties", [prop])
        if prop not in tracked:
            continue

        # Determine which agents are absent/can't see
        acting_agent_room = state.agent_rooms.get(agent_id, "unknown")
        unaware_agents = []
        for other_agent, room in state.agent_rooms.items():
            if other_agent != agent_id and room != acting_agent_room:
                unaware_agents.append(other_agent)

        # Get old value
        old_value = state.get_object_property(target, prop, not new_value)

        # Record the unseen change
        new_state = copy.copy(state)
        new_state.unseen_state_changes = copy.copy(state.unseen_state_changes)
        new_state.unseen_state_changes[target] = {
            "property": prop,
            "old_value": old_value,
            "new_value": new_value,
            "changed_by": agent_id,
            "changed_at_step": state.current_step,
            "unaware_agents": unaware_agents,
        }

        obs_parts = [f"You {action_lower} {target}."]
        if unaware_agents:
            obs_parts.append(f"({', '.join(unaware_agents)} didn't see this.)")

        return HandlerResult(
            applies=True,
            state=new_state,
            observation=" ".join(obs_parts),
            success=True,
            effects=[f"state_unseen={target}.{prop}:{old_value}->{new_value}"],
            surprise_trigger=f"{target} state changed while agents unaware",
            mechanic_type="state_change_unseen",
        )

    return no_effect(state)




# =============================================================================
# Handler Registry
# =============================================================================

MECHANIC_HANDLERS: Dict[str, MechanicHandler] = {
    # State transform mechanics
    "inverse_state": handle_inverse_state,
    "remote_control": handle_remote_control,
    "conditional_unlock": handle_conditional_unlock,
    "state_mirroring": handle_state_mirroring,
    # Belief tracking mechanics
    "location_change": handle_location_change,
    "container_swap": handle_container_swap,
    "state_change_unseen": handle_state_change_unseen,
}


def get_handler(name: str) -> Optional[MechanicHandler]:
    """Get handler function for a mechanic."""
    return MECHANIC_HANDLERS.get(name)


def apply_mechanics(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Apply all active mechanics to an action.

    Returns the first mechanic that applies, or a default result.
    """
    for mech_name in state.active_mechanics:
        handler = get_handler(mech_name)
        if handler:
            result = handler(action_name, agent_id, target, state)
            if result.applies:
                return result

    # No mechanic applied - return default
    return HandlerResult(
        applies=False,
        state=state,
        observation=f"You {action_name} {target}." if target else f"You {action_name}.",
        success=True,
        effects=[],
    )
