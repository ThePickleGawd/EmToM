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
    "room_restriction": {
        "description": "Specific agents cannot enter certain rooms, forcing collaboration",
        "category": "navigation_block",
        "setup_keys": ["restricted_rooms", "for_agents"],
        "agent_observation": "You cannot enter {room}. The area is off-limits to you.",
        "tom_use": "Agent with knowledge of item location cannot access it, must communicate with partner who can",
        "example_binding": {"mechanic_type": "room_restriction", "restricted_rooms": ["bathroom_1"], "for_agents": ["agent_0"]},
        "recommended_for_tom": True,
    },
    "limited_bandwidth": {
        "description": "Agents have a maximum number of messages they can send, forcing strategic communication",
        "category": "communication_constraint",
        "setup_keys": ["message_limits"],
        "agent_observation": "You have used {used}/{max} messages. Choose your words carefully.",
        "tom_use": "Agents must reason about what the other agent needs to know most, prioritize information, and model what the other agent can infer without being told",
        "example_binding": {"mechanic_type": "limited_bandwidth", "message_limits": {"agent_0": 3, "agent_1": 3}},
        "alt_example": {"mechanic_type": "limited_bandwidth", "message_limits": {"agent_0": 2, "agent_1": 5}},
        "recommended_for_tom": True,
    },
    "irreversible_action": {
        "description": "Once an action is performed on an object, that object is permanently locked and no further actions can be taken on it by anyone",
        "category": "state_constraint",
        "setup_keys": [],
        "agent_observation": "That action is permanent — {target} is now locked and cannot be interacted with again.",
        "tom_use": "Agents must plan carefully and communicate before acting, since mistakes cannot be undone. Creates high-stakes coordination pressure.",
        "example_binding": {"mechanic_type": "irreversible_action"},
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

    # Special-case unlock semantics: treat is_unlocked as the inverse of is_locked
    # so remote triggers can actually unlock containers.
    if remote_property == "is_unlocked":
        new_state = new_state.set_object_property(remote_target, "is_locked", not new_value)

    # Describe what happened
    if remote_property == "is_open":
        effect_desc = "opened" if new_value else "closed"
    elif remote_property == "is_unlocked":
        effect_desc = "unlocked" if new_value else "locked"
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
    LOCKABLE_ACTIONS = {"Open", "Close"}
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


def handle_room_restriction(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Room Restriction: Certain agents cannot navigate to specific rooms.

    Forces collaboration by preventing an agent from accessing locations
    they may have information about. They must communicate with partners
    who can reach those areas.

    Setup in task.json:
        mechanic_bindings: [
            {
                "mechanic_type": "room_restriction",
                "restricted_rooms": ["bathroom_1", "office_2"],
                "for_agents": ["agent_0"]
            }
        ]

    Result: agent_0 cannot Navigate to bathroom_1 or office_2.
    """
    # Only applies to Navigate actions
    if action_name.lower() != "navigate" or not target:
        return no_effect(state)

    # Check if this agent has any room restrictions
    restricted = state.restricted_rooms.get(agent_id, set())
    if not restricted:
        return no_effect(state)

    # Check if target room is restricted for this agent
    if target in restricted:
        return HandlerResult(
            applies=True,
            state=state,
            observation=f"You cannot enter {target}. The area is off-limits to you.",
            success=False,
            effects=[f"blocked_navigation={agent_id}_to_{target}"],
            surprise_trigger=f"{target} is restricted for {agent_id}",
            blocked=True,
        )

    # If navigating to an object/furniture, block if it's located in a restricted room
    target_room = state.get_object_property(target, "room", None)
    if not target_room:
        for entity in state.entities:
            name = entity.get("name") or entity.get("id")
            if name == target:
                target_room = entity.get("room")
                break

    if target_room and target_room in restricted:
        return HandlerResult(
            applies=True,
            state=state,
            observation=f"You cannot enter {target_room} to reach {target}. The area is off-limits to you.",
            success=False,
            effects=[f"blocked_navigation={agent_id}_to_{target}"],
            surprise_trigger=f"{target_room} is restricted for {agent_id}",
            blocked=True,
        )

    return no_effect(state)


def handle_limited_bandwidth(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Limited Bandwidth: Agents have a maximum number of Communicate actions.

    Forces agents to think carefully about what information to share and when.
    Creates ToM pressure: agents must model what the other agent needs to know
    most and prioritize critical information over nice-to-have details.

    Setup in task.json:
        mechanic_bindings: [
            {
                "mechanic_type": "limited_bandwidth",
                "message_limits": {"agent_0": 3, "agent_1": 3}
            }
        ]

    Result: Each agent can only send the specified number of messages.
    """
    # Only applies to Communicate actions
    if action_name.lower() != "communicate":
        return no_effect(state)

    # Check if this agent has a message limit
    limit = state.message_limits.get(agent_id)
    if limit is None:
        return no_effect(state)

    sent = state.messages_sent.get(agent_id, 0)

    if sent >= limit:
        return HandlerResult(
            applies=True,
            state=state,
            observation=f"You have used all {limit} of your allowed messages. You cannot send any more.",
            success=False,
            effects=[f"blocked_communicate={agent_id}_limit_reached"],
            surprise_trigger=f"{agent_id} has exhausted message budget",
            blocked=True,
            mechanic_type="limited_bandwidth",
        )

    # Message allowed — increment counter
    new_state = copy.copy(state)
    new_sent = dict(state.messages_sent)
    new_sent[agent_id] = sent + 1
    new_state.messages_sent = new_sent

    remaining = limit - (sent + 1)
    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"Message sent. You have {remaining} message{'s' if remaining != 1 else ''} remaining.",
        success=True,
        effects=[f"message_sent={agent_id}_{sent + 1}_of_{limit}"],
        mechanic_type="limited_bandwidth",
    )


def handle_irreversible_action(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Irreversible Action: Once an action succeeds on an object, it is permanently locked.

    Any subsequent action targeting a locked object is blocked.
    Only physical interaction actions trigger/are affected by the lock
    (Open, Close, Pick, Place, UseItem). Navigate and Communicate are unaffected.

    Setup: No per-object configuration needed. When this mechanic is active,
    ALL successful interaction actions lock their target object.

    State:
        state.locked_objects = {"cabinet_1": "Open by agent_0", ...}
    """
    if not target:
        return no_effect(state)

    # Only applies to physical interaction actions
    INTERACTION_ACTIONS = {"Open", "Close", "Pick", "Place", "UseItem"}
    if action_name not in INTERACTION_ACTIONS:
        return no_effect(state)

    # For Place targets like "apple_1, on, table_1", extract the primary object
    primary_target = target.split(",")[0].strip() if "," in target else target

    # Check if object is already locked → block
    if primary_target in state.locked_objects:
        locked_by = state.locked_objects[primary_target]
        return HandlerResult(
            applies=True,
            state=state,
            observation=f"You cannot {action_name.lower()} {primary_target} — it was permanently locked after a previous action ({locked_by}).",
            success=False,
            effects=[],
            blocked=True,
            mechanic_type="irreversible_action",
        )

    # Object is not locked yet — allow the action but mark it for locking
    new_state = copy.copy(state)
    new_locked = dict(state.locked_objects)
    new_locked[primary_target] = f"{action_name} by {agent_id}"
    new_state.locked_objects = new_locked

    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"Action performed. {primary_target} is now permanently locked — no further actions can be taken on it.",
        success=True,
        effects=[f"locked={primary_target}_by_{agent_id}"],
        mechanic_type="irreversible_action",
    )


# =============================================================================
# Handler Registry
# =============================================================================

MECHANIC_HANDLERS: Dict[str, MechanicHandler] = {
    "inverse_state": handle_inverse_state,
    "remote_control": handle_remote_control,
    "conditional_unlock": handle_conditional_unlock,
    "state_mirroring": handle_state_mirroring,
    "room_restriction": handle_room_restriction,
    "limited_bandwidth": handle_limited_bandwidth,
    "irreversible_action": handle_irreversible_action,
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
