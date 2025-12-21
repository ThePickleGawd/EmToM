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
import uuid

from emtom.state.game_state import EMTOMGameState, PendingEffect


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
    },
    "remote_control": {
        "description": "Acting on one object affects a different object",
        "category": "hidden_mapping",
        "setup_keys": ["trigger_object", "target_object", "target_state"],
    },
    "counting_state": {
        "description": "Object only responds after N interactions",
        "category": "conditional",
        "setup_keys": ["trigger_object", "required_count"],
    },
    "delayed_effect": {
        "description": "Actions take effect after N steps",
        "category": "time_delayed",
        "setup_keys": ["trigger_object", "delay_steps"],
    },
    "decaying_state": {
        "description": "States automatically revert after N steps",
        "category": "time_delayed",
        "setup_keys": ["trigger_object", "decay_steps"],
    },
    "conditional_unlock": {
        "description": "Object only works after prerequisite action or having an item",
        "category": "conditional",
        "setup_keys": ["trigger_object", "prerequisite_object", "prerequisite_action", "requires_item"],
    },
    "state_mirroring": {
        "description": "Two objects always have the same state",
        "category": "hidden_mapping",
        "setup_keys": ["trigger_object", "target_object", "target_state"],
    },
    "sequence_lock": {
        "description": "Objects must be interacted in specific order",
        "category": "conditional",
        "setup_keys": ["trigger_object", "sequence"],
    },
}


def get_mechanic_info(name: str) -> Dict[str, Any]:
    """Get info about a mechanic."""
    return MECHANIC_INFO.get(name, {})


def list_mechanics() -> list:
    """List all available mechanics."""
    return list(MECHANIC_INFO.keys())


# =============================================================================
# Helper Functions
# =============================================================================

def action_to_state(action_name: str) -> str:
    """Map action name to state property."""
    mapping = {
        "open": "is_open",
        "close": "is_open",
        "turn_on": "is_on",
        "turn_off": "is_on",
        "lock": "is_locked",
        "unlock": "is_locked",
    }
    return mapping.get(action_name, "is_active")


def action_to_value(action_name: str) -> bool:
    """Map action name to resulting value."""
    mapping = {
        "open": True,
        "close": False,
        "turn_on": True,
        "turn_off": False,
        "lock": True,
        "unlock": False,
    }
    return mapping.get(action_name, True)


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


def handle_counting_state(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Counting State: Object only responds after N interactions.

    Setup in state:
        state.interaction_counts = {"door_1": 0}
        state.object_properties["door_1"]["required_count"] = 3
    """
    if not target or target not in state.interaction_counts:
        return no_effect(state)

    required_count = state.get_object_property(target, "required_count", 3)
    current_count = state.interaction_counts.get(target, 0) + 1
    action_lower = action_name.lower()

    # Update count
    new_counts = copy.copy(state.interaction_counts)
    new_counts[target] = current_count
    new_state = copy.copy(state)
    new_state.interaction_counts = new_counts

    if current_count >= required_count:
        # Reset count for next cycle
        new_counts[target] = 0
        return HandlerResult(
            applies=True,
            state=new_state,
            observation=f"This time it responds!",
            success=True,
            effects=[f"count_reached={current_count}"],
        )
    else:
        # Block the action - don't execute in Habitat
        return HandlerResult(
            applies=True,
            state=new_state,
            observation=f"You {action_lower} {target}, but nothing happens. ({current_count}/{required_count})",
            success=False,
            effects=[f"count={current_count}/{required_count}"],
            surprise_trigger=f"{target} didn't respond (attempt {current_count})",
            blocked=True,  # Don't execute in Habitat
        )


def handle_delayed_effect(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Delayed Effect: Actions take effect after N steps.

    Setup in state:
        state.object_properties["door_1"]["delay_steps"] = 3
    """
    delay_steps = state.get_object_property(target, "delay_steps")
    if not delay_steps or not target:
        return no_effect(state)

    # Only apply to state-changing actions (not Navigate, Pick, etc.)
    action_lower = action_name.lower()
    state_changing_actions = {"open", "close", "turn_on", "turn_off", "lock", "unlock"}
    if action_lower not in state_changing_actions:
        return no_effect(state)

    # Create pending effect
    effect = PendingEffect(
        effect_id=str(uuid.uuid4()),
        target=target,
        property_name=action_to_state(action_name),
        new_value=action_to_value(action_name),
        steps_remaining=delay_steps,
        triggered_by=agent_id,
        triggered_at_step=state.current_step,
        description=f"{action_name} on {target} (delayed {delay_steps} steps)",
    )
    new_state = state.add_pending_effect(effect)

    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"You {action_lower} {target}, but nothing seems to happen immediately.",
        success=True,
        effects=[f"delayed={delay_steps}_steps"],
        surprise_trigger=f"{target} didn't respond immediately",
        actual_action=action_name,
        actual_target=target,
        blocked=True,  # Don't execute immediately - effect is delayed
    )


def handle_decaying_state(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Decaying State: States automatically revert after N steps.

    Setup in state:
        state.object_properties["door_1"]["decay_steps"] = 3
    """
    decay_steps = state.get_object_property(target, "decay_steps")
    if not decay_steps or not target:
        return no_effect(state)

    # Only apply to state-changing actions (not Navigate, Pick, etc.)
    action_lower = action_name.lower()
    state_changing_actions = {"open", "close", "turn_on", "turn_off", "lock", "unlock"}
    if action_lower not in state_changing_actions:
        return no_effect(state)

    # Schedule reversion (opposite of current action)
    current_value = action_to_value(action_name)
    revert_value = not current_value

    effect = PendingEffect(
        effect_id=str(uuid.uuid4()),
        target=target,
        property_name=action_to_state(action_name),
        new_value=revert_value,
        steps_remaining=decay_steps,
        triggered_by=agent_id,
        triggered_at_step=state.current_step,
        description=f"{target} will revert in {decay_steps} steps",
    )
    new_state = state.add_pending_effect(effect)

    return HandlerResult(
        applies=True,
        state=new_state,
        observation=f"(This will revert in {decay_steps} steps)",
        success=True,
        effects=[f"will_decay_in={decay_steps}_steps"],
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

    action_lower = action_name.lower()

    # Check if target is locked by item prerequisite (must have item in inventory)
    required_item = state.get_object_property(target, "requires_item")
    if required_item and target not in state.unlocked_targets:
        # Check if agent has the required item
        agent_inventory = state.agent_inventory.get(agent_id, [])
        if required_item not in agent_inventory:
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


def handle_sequence_lock(
    action_name: str,
    agent_id: str,
    target: Optional[str],
    state: EMTOMGameState,
) -> HandlerResult:
    """
    Sequence Lock: Objects must be interacted in specific order.

    Setup in state:
        state.sequence_progress = {"chest_1": 0}
        state.object_properties["chest_1"]["sequence_length"] = 3
        state.object_properties["button_1"]["advances_sequence"] = "chest_1"
    """
    if not target:
        return no_effect(state)

    # Check if target is sequence-locked
    if target in state.sequence_progress and target not in state.sequence_unlocked:
        return HandlerResult(
            applies=True,
            state=state,
            observation=f"You try to {action_name} {target}, but it's locked. A specific sequence is required.",
            success=False,
            effects=[],
            surprise_trigger=f"{target} requires a sequence to unlock",
            blocked=True,  # Don't execute in Habitat
        )

    # Check if this action advances a sequence
    sequence_target = state.get_object_property(target, "advances_sequence")
    if sequence_target and sequence_target in state.sequence_progress:
        new_progress = copy.copy(state.sequence_progress)
        current = new_progress.get(sequence_target, 0)
        new_progress[sequence_target] = current + 1
        new_state = copy.copy(state)
        new_state.sequence_progress = new_progress

        required = state.get_object_property(sequence_target, "sequence_length", 3)

        if new_progress[sequence_target] >= required:
            new_unlocked = copy.copy(state.sequence_unlocked)
            new_unlocked.add(sequence_target)
            new_state.sequence_unlocked = new_unlocked
            return HandlerResult(
                applies=True,
                state=new_state,
                observation=f"You hear a satisfying click - something unlocked!",
                success=True,
                effects=[f"sequence_complete={sequence_target}"],
                surprise_trigger=f"Sequence completed, {sequence_target} unlocked",
            )
        else:
            return HandlerResult(
                applies=True,
                state=new_state,
                observation=f"You hear a small click.",
                success=True,
                effects=[f"sequence_progress={new_progress[sequence_target]}/{required}"],
            )

    return no_effect(state)


# =============================================================================
# Handler Registry
# =============================================================================

MECHANIC_HANDLERS: Dict[str, MechanicHandler] = {
    "inverse_state": handle_inverse_state,
    "remote_control": handle_remote_control,
    "counting_state": handle_counting_state,
    "delayed_effect": handle_delayed_effect,
    "decaying_state": handle_decaying_state,
    "conditional_unlock": handle_conditional_unlock,
    "state_mirroring": handle_state_mirroring,
    "sequence_lock": handle_sequence_lock,
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
