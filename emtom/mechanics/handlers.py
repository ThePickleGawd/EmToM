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
    "delayed_information": {
        "description": "Information is revealed to an agent after a delay",
        "category": "information_asymmetry",
        "setup_keys": ["info_id", "content", "delay_turns", "target_agents"],
        "agent_observation": "You suddenly remember: {content}",
        "tom_use": "Agent A gets info first, must decide whether to share before B learns it naturally.",
        "example_binding": {"mechanic_type": "delayed_information", "info_id": "key_hint", "content": "The key is hidden in drawer_5", "delay_turns": 3, "target_agents": ["agent_1"]},
        "recommended_for_tom": True,
    },
    # === Communication Mechanics ===
    "limited_bandwidth": {
        "description": "Agents can only send N messages total during the episode",
        "category": "communication_constraint",
        "setup_keys": ["max_messages"],
        "agent_observation": "Message sent. You have {remaining} messages left.",
        "tom_use": "Must prioritize what to share - forces strategic communication decisions.",
        "example_binding": {"mechanic_type": "limited_bandwidth", "max_messages": 3},
        "recommended_for_tom": True,
    },
    "delayed_messages": {
        "description": "Messages arrive after N turns delay",
        "category": "communication_constraint",
        "setup_keys": ["delay_turns"],
        "agent_observation": "Message sent. It will arrive in {delay} turns.",
        "tom_use": "Must plan ahead for communication lag - actions and messages desync.",
        "example_binding": {"mechanic_type": "delayed_messages", "delay_turns": 2},
        "recommended_for_tom": True,
    },
    "noisy_channel": {
        "description": "Messages may be corrupted or lost entirely",
        "category": "communication_constraint",
        "setup_keys": ["corruption_rate", "drop_rate"],
        "agent_observation": "Message sent through noisy channel.",
        "tom_use": "Must use redundancy for critical info - uncertainty about what partner received.",
        "example_binding": {"mechanic_type": "noisy_channel", "corruption_rate": 0.3, "drop_rate": 0.1},
        "recommended_for_tom": True,
    },
    # === Coordination Mechanics ===
    "hidden_agenda": {
        "description": "Agents have secret, potentially conflicting goals",
        "category": "coordination",
        "setup_keys": ["agent_goals"],
        "agent_observation": "You have a secret objective: {goal}",
        "tom_use": "Agents must infer others' hidden motives from behavior. May need to cooperate despite conflict.",
        "example_binding": {"mechanic_type": "hidden_agenda", "agent_goals": {"agent_0": {"goal": "get_apple", "target": "apple_3"}, "agent_1": {"goal": "get_apple", "target": "apple_3"}}},
        "recommended_for_tom": True,
    },
    "simultaneous_action": {
        "description": "Certain actions require multiple agents to act together in the same step",
        "category": "coordination",
        "setup_keys": ["required_action", "target", "required_agents", "window_size"],
        "agent_observation": "This requires coordination - all agents must act together!",
        "tom_use": "Agents must communicate and synchronize timing. Tests coordination without explicit planning.",
        "example_binding": {"mechanic_type": "simultaneous_action", "action_id": "open_vault", "required_action": "Open", "target": "vault_1", "required_agents": ["agent_0", "agent_1"], "window_size": 1},
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


def handle_delayed_information(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Delayed Information: Reveal information to agents after a delay.

    Creates information asymmetry over time:
    - Agent A might learn info at step 0
    - Agent B learns same info at step 5
    - A must decide whether to share before B learns naturally

    Setup in state via mechanic_bindings:
        {"mechanic_type": "delayed_information", "info_id": "key_hint",
         "content": "The key is in drawer_5", "reveal_to": "agent_1",
         "reveal_at_step": 5}

    This handler checks on EVERY action whether delayed info should be revealed.
    """
    new_state = state
    revealed_info = []

    # Check all delayed info entries
    for info_id, info_data in state.delayed_info.items():
        if info_data.get("revealed", False):
            continue

        reveal_step = info_data.get("reveal_at_step", 0)
        if state.current_step < reveal_step:
            continue

        # Time to reveal!
        target_agents = info_data.get("target_agents", [])
        content = info_data.get("content", "")

        # Only reveal to the acting agent if they're a target
        if agent_id in target_agents:
            revealed_info.append((info_id, content))

            # Mark as revealed for this agent
            new_state = copy.copy(new_state)
            new_state.delayed_info = copy.copy(new_state.delayed_info)
            new_state.delayed_info[info_id] = copy.copy(info_data)
            new_state.delayed_info[info_id]["revealed"] = True
            new_state.delayed_info[info_id]["revealed_at_step"] = state.current_step

    if revealed_info:
        observations = [f"You suddenly remember: {content}" for _, content in revealed_info]
        return HandlerResult(
            applies=True,
            state=new_state,
            observation=" ".join(observations),
            success=True,
            effects=[f"info_revealed={info_id}" for info_id, _ in revealed_info],
            surprise_trigger="Delayed information revealed",
            mechanic_type="delayed_information",
        )

    return no_effect(state)


# =============================================================================
# Communication Mechanic Handlers
# =============================================================================

def handle_limited_bandwidth(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Limited Bandwidth: Agents can only send N messages total.

    Forces strategic communication decisions - agents must prioritize
    what information to share when messages are scarce.

    Setup in state via mechanic_bindings:
        {"mechanic_type": "limited_bandwidth", "max_messages": 3}

    Triggers on Communicate actions.
    """
    if action_name.lower() != "communicate":
        return no_effect(state)

    # Check for limited_bandwidth binding
    max_msgs = None
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") == "limited_bandwidth":
            max_msgs = binding.get("max_messages", 5)
            break

    if max_msgs is None:
        return no_effect(state)

    # Get current message count for this agent
    current_count = state.message_counts.get(agent_id, 0)

    # Check if agent has exceeded limit
    if current_count >= max_msgs:
        return HandlerResult(
            applies=True,
            state=state,
            observation=f"You try to send a message, but you've used all {max_msgs} of your messages!",
            success=False,
            effects=["message_blocked=bandwidth_exceeded"],
            surprise_trigger="Out of messages",
            blocked=True,
            mechanic_type="limited_bandwidth",
        )

    # Increment message count
    new_state = copy.copy(state)
    new_state.message_counts = copy.copy(state.message_counts)
    new_state.message_counts[agent_id] = current_count + 1

    remaining = max_msgs - (current_count + 1)
    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"Message sent. You have {remaining} message{'s' if remaining != 1 else ''} remaining.",
        success=True,
        effects=[f"message_count={agent_id}:{current_count + 1}/{max_msgs}"],
        mechanic_type="limited_bandwidth",
    )


def handle_delayed_messages(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Delayed Messages: Messages arrive after N turns.

    Creates temporal desync between actions and communication.
    Agents must plan ahead, accounting for the delay.

    Setup in state via mechanic_bindings:
        {"mechanic_type": "delayed_messages", "delay_turns": 2}

    Triggers on Communicate actions and also checks for pending deliveries.
    """
    # First, check for any messages that should be delivered now
    delivered_messages = []
    new_pending = []
    new_state = state

    for msg in state.pending_messages:
        if msg.get("deliver_at_step", 0) <= state.current_step:
            # This message should be delivered
            if msg.get("to_agent") != agent_id:
                # Not for this agent, keep it pending for them
                new_pending.append(msg)
            else:
                delivered_messages.append(msg)
        else:
            new_pending.append(msg)

    # If we delivered messages, update state and return that observation
    if delivered_messages:
        new_state = copy.copy(state)
        new_state.pending_messages = new_pending

        contents = [m.get("content", "") for m in delivered_messages]
        senders = [m.get("from_agent", "someone") for m in delivered_messages]
        obs_parts = []
        for sender, content in zip(senders, contents):
            obs_parts.append(f"Delayed message from {sender}: \"{content}\"")

        return HandlerResult(
            applies=True,
            state=new_state,
            observation=" | ".join(obs_parts),
            success=True,
            effects=[f"delayed_message_delivered={m.get('msg_id')}" for m in delivered_messages],
            surprise_trigger="Delayed message arrived",
            mechanic_type="delayed_messages",
        )

    # Now handle new Communicate actions
    if action_name.lower() != "communicate":
        return no_effect(state)

    # Check for delayed_messages binding
    delay = None
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") == "delayed_messages":
            delay = binding.get("delay_turns", 1)
            break

    if delay is None:
        return no_effect(state)

    # Queue the message for delayed delivery
    # target contains the message content for Communicate actions
    message_content = target if target else ""

    # Determine recipient (all other agents)
    other_agents = [a for a in state.agent_rooms.keys() if a != agent_id]

    new_state = copy.copy(state)
    new_state.pending_messages = copy.copy(state.pending_messages)

    import uuid
    msg_id = str(uuid.uuid4())[:8]

    for recipient in other_agents:
        new_state.pending_messages.append({
            "msg_id": msg_id,
            "from_agent": agent_id,
            "to_agent": recipient,
            "content": message_content,
            "sent_at_step": state.current_step,
            "deliver_at_step": state.current_step + delay,
        })

    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"Message queued. It will arrive in {delay} turn{'s' if delay != 1 else ''}.",
        success=True,
        effects=[f"message_delayed={msg_id}:+{delay}turns"],
        mechanic_type="delayed_messages",
    )


def handle_noisy_channel(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Noisy Channel: Messages may be corrupted or lost.

    Creates uncertainty about communication success:
    - corruption_rate: probability that message is garbled
    - drop_rate: probability that message is lost entirely

    Setup in state via mechanic_bindings:
        {"mechanic_type": "noisy_channel", "corruption_rate": 0.3, "drop_rate": 0.1}

    Triggers on Communicate actions.
    """
    if action_name.lower() != "communicate":
        return no_effect(state)

    # Check for noisy_channel binding
    corruption_rate = None
    drop_rate = None
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") == "noisy_channel":
            corruption_rate = binding.get("corruption_rate", 0.2)
            drop_rate = binding.get("drop_rate", 0.1)
            break

    if corruption_rate is None and drop_rate is None:
        return no_effect(state)

    import random

    message_content = target if target else ""

    # Check if message is dropped
    if random.random() < (drop_rate or 0):
        return HandlerResult(
            applies=True,
            state=state,
            observation="You send a message... but hear only static. The message was lost!",
            success=False,
            effects=["message_dropped=noise"],
            surprise_trigger="Message lost in transmission",
            blocked=True,
            mechanic_type="noisy_channel",
        )

    # Check if message is corrupted
    if random.random() < (corruption_rate or 0):
        # Corrupt the message by replacing some words with noise
        words = message_content.split()
        if len(words) > 0:
            num_corrupt = max(1, len(words) // 3)
            corrupt_indices = random.sample(range(len(words)), min(num_corrupt, len(words)))
            noise_words = ["[static]", "[garbled]", "[???]", "[noise]", "[unintelligible]"]
            for idx in corrupt_indices:
                words[idx] = random.choice(noise_words)
            corrupted_content = " ".join(words)
        else:
            corrupted_content = "[garbled]"

        # Record the corruption for tracking
        new_state = copy.copy(state)
        new_state.message_history = copy.copy(state.message_history)
        new_state.message_history.append({
            "from": agent_id,
            "original": message_content,
            "received": corrupted_content,
            "step": state.current_step,
            "corrupted": True,
        })

        return HandlerResult(
            applies=True,
            state=new_state,
            observation=f"Message sent through noisy channel. (Some interference detected)",
            success=True,
            effects=[f"message_corrupted={len(corrupt_indices)}_words"],
            surprise_trigger="Message corrupted by noise",
            mechanic_type="noisy_channel",
            # The actual_target will contain the corrupted message for the recipient
        )

    # Message went through cleanly
    new_state = copy.copy(state)
    new_state.message_history = copy.copy(state.message_history)
    new_state.message_history.append({
        "from": agent_id,
        "original": message_content,
        "received": message_content,
        "step": state.current_step,
        "corrupted": False,
    })

    return HandlerResult(
        applies=True,
        state=new_state,
        observation="Message sent through noisy channel. (Transmission clear)",
        success=True,
        effects=["message_sent=clear"],
        mechanic_type="noisy_channel",
    )


# =============================================================================
# Coordination Mechanic Handlers
# =============================================================================

def handle_hidden_agenda(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Hidden Agenda: Agents have secret, potentially conflicting goals.

    Each agent has a private objective that may conflict with others.
    Creates situations where agents must:
    - Infer others' goals from behavior
    - Decide whether to cooperate or compete
    - Balance public task with private agenda

    Setup in state via mechanic_bindings:
        {"mechanic_type": "hidden_agenda", "agent_goals": {
            "agent_0": {"goal": "get_apple", "target": "apple_3", "description": "You want the apple for yourself"},
            "agent_1": {"goal": "get_apple", "target": "apple_3", "description": "You want the apple for yourself"}
        }}

    Tracks when agents achieve their hidden goals and detects conflicts.
    """
    # Check for hidden_agenda binding
    agent_goals = None
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") == "hidden_agenda":
            agent_goals = binding.get("agent_goals", {})
            break

    if not agent_goals or agent_id not in agent_goals:
        return no_effect(state)

    my_agenda = agent_goals.get(agent_id, {})
    my_goal = my_agenda.get("goal", "")
    my_target = my_agenda.get("target", "")

    # Check if this action achieves the agent's hidden goal
    goal_achieved = False
    action_lower = action_name.lower()

    # Check various goal types
    if my_goal == "get_item" and action_lower == "pick" and target == my_target:
        goal_achieved = True
    elif my_goal == "place_item" and action_lower == "place" and target == my_target:
        goal_achieved = True
    elif my_goal == "open_container" and action_lower == "open" and target == my_target:
        goal_achieved = True
    elif my_goal == "reach_location" and action_lower == "navigate" and target == my_target:
        goal_achieved = True
    elif my_goal.startswith("get_") and action_lower == "pick" and target == my_target:
        # Generic "get_X" goal
        goal_achieved = True

    if goal_achieved:
        # Update state to mark goal as achieved
        new_state = copy.copy(state)
        new_state.hidden_agendas = copy.copy(state.hidden_agendas)
        new_state.hidden_agendas[agent_id] = {
            "goal": my_goal,
            "target": my_target,
            "achieved": True,
            "achieved_at_step": state.current_step,
        }

        # Check for conflicts - did another agent want the same thing?
        conflicts = []
        for other_agent, other_agenda in agent_goals.items():
            if other_agent != agent_id:
                if other_agenda.get("target") == my_target and other_agenda.get("goal") == my_goal:
                    conflicts.append(other_agent)

        effects = [f"hidden_goal_achieved={agent_id}:{my_goal}"]
        if conflicts:
            effects.append(f"goal_conflict_with={','.join(conflicts)}")

        return HandlerResult(
            applies=True,
            state=new_state,
            observation=f"You achieved your secret objective!",
            success=True,
            effects=effects,
            surprise_trigger="Hidden agenda achieved",
            mechanic_type="hidden_agenda",
        )

    # Track actions that might reveal intentions to observant agents
    new_state = copy.copy(state)
    new_state.hidden_agendas = copy.copy(state.hidden_agendas)
    if agent_id not in new_state.hidden_agendas:
        new_state.hidden_agendas[agent_id] = {
            "goal": my_goal,
            "target": my_target,
            "achieved": False,
            "actions_toward_goal": [],
        }

    # Check if this action is toward the goal target (reveals intent)
    if target == my_target:
        actions_list = new_state.hidden_agendas[agent_id].get("actions_toward_goal", [])
        actions_list.append({"action": action_name, "step": state.current_step})
        new_state.hidden_agendas[agent_id]["actions_toward_goal"] = actions_list

        return HandlerResult(
            applies=True,
            state=new_state,
            observation="",  # Silent tracking
            success=True,
            effects=[f"action_toward_hidden_goal={agent_id}"],
            mechanic_type="hidden_agenda",
        )

    return no_effect(state)


def handle_simultaneous_action(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Simultaneous Action: Certain actions require multiple agents to act together.

    Some objectives can only be achieved if agents coordinate to act
    in the same step (e.g., both press buttons, both pull levers).

    Setup in state via mechanic_bindings:
        {"mechanic_type": "simultaneous_action", "action_id": "open_vault",
         "required_action": "Open", "target": "vault_1",
         "required_agents": ["agent_0", "agent_1"], "window_size": 1}

    window_size: How many steps agents have to complete the synchronized action
                 (1 = must be exact same step, 2 = within 2 steps, etc.)
    """
    if not target:
        return no_effect(state)

    # Check for simultaneous_action bindings that match this action
    for binding in state.mechanic_bindings:
        if binding.get("mechanic_type") != "simultaneous_action":
            continue

        required_action = binding.get("required_action", "")
        required_target = binding.get("target", "")
        required_agents = binding.get("required_agents", [])
        window_size = binding.get("window_size", 1)
        action_id = binding.get("action_id", f"{required_action}_{required_target}")

        # Check if this action matches the requirement
        if action_name.lower() != required_action.lower():
            continue
        if target != required_target:
            continue
        if agent_id not in required_agents:
            continue

        # This agent is attempting the coordinated action
        new_state = copy.copy(state)
        new_state.simultaneous_requirements = copy.copy(state.simultaneous_requirements)
        new_state.current_step_actions = copy.copy(state.current_step_actions)

        # Record this agent's action
        new_state.current_step_actions[agent_id] = {
            "action": action_name,
            "target": target,
            "step": state.current_step,
        }

        # Initialize or update the requirement tracking
        if action_id not in new_state.simultaneous_requirements:
            new_state.simultaneous_requirements[action_id] = {
                "required_agents": required_agents,
                "action": required_action,
                "target": required_target,
                "pending_agents": [agent_id],
                "window_start": state.current_step,
                "window_size": window_size,
                "completed": False,
            }
        else:
            req = new_state.simultaneous_requirements[action_id]
            # Check if we're still within the window
            if state.current_step - req["window_start"] >= window_size:
                # Window expired, reset
                req["pending_agents"] = [agent_id]
                req["window_start"] = state.current_step
            elif agent_id not in req["pending_agents"]:
                req["pending_agents"].append(agent_id)

        # Check if all required agents have now acted
        req = new_state.simultaneous_requirements[action_id]
        pending = set(req["pending_agents"])
        required = set(required_agents)

        if pending >= required:
            # All agents acted within the window - success!
            req["completed"] = True
            req["completed_at_step"] = state.current_step

            return HandlerResult(
                applies=True,
                state=new_state,
                observation=f"Synchronized action successful! All agents acted together on {target}.",
                success=True,
                effects=[f"simultaneous_action_complete={action_id}"],
                surprise_trigger="Coordinated action succeeded",
                mechanic_type="simultaneous_action",
            )
        else:
            # Still waiting for other agents
            waiting_for = required - pending
            return HandlerResult(
                applies=True,
                state=new_state,
                observation=f"You attempt to {action_name.lower()} {target}, but it requires coordination. Waiting for: {', '.join(waiting_for)}",
                success=False,
                effects=[f"simultaneous_action_waiting={action_id}"],
                surprise_trigger="Waiting for coordinated action",
                blocked=True,  # Block the action until all agents participate
                mechanic_type="simultaneous_action",
            )

    return no_effect(state)


# =============================================================================
# Handler Registry
# =============================================================================

MECHANIC_HANDLERS: Dict[str, MechanicHandler] = {
    # Original mechanics
    "inverse_state": handle_inverse_state,
    "remote_control": handle_remote_control,
    "conditional_unlock": handle_conditional_unlock,
    "state_mirroring": handle_state_mirroring,
    # Theory of Mind mechanics
    "location_change": handle_location_change,
    "container_swap": handle_container_swap,
    "state_change_unseen": handle_state_change_unseen,
    "delayed_information": handle_delayed_information,
    # Communication mechanics
    "limited_bandwidth": handle_limited_bandwidth,
    "delayed_messages": handle_delayed_messages,
    "noisy_channel": handle_noisy_channel,
    # Coordination mechanics
    "hidden_agenda": handle_hidden_agenda,
    "simultaneous_action": handle_simultaneous_action,
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
