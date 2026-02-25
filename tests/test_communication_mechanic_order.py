from emtom.mechanics.handlers import apply_mechanics
from emtom.state.manager import GameStateManager


def _init_state(mechanic_bindings):
    manager = GameStateManager(None)
    return manager.initialize_from_task(
        {
            "active_mechanics": [],
            "mechanic_bindings": mechanic_bindings,
            "items": [],
            "locked_containers": {},
            "initial_states": {},
        }
    )


def test_disallowed_recipient_blocks_independent_of_binding_order():
    bindings_a = [
        {"mechanic_type": "limited_bandwidth", "message_limits": {"agent_0": 2}},
        {
            "mechanic_type": "restricted_communication",
            "allowed_targets": {"agent_0": ["agent_1"]},
        },
    ]
    bindings_b = list(reversed(bindings_a))

    for bindings in (bindings_a, bindings_b):
        state = _init_state(bindings)
        result = apply_mechanics("Communicate", "agent_0", '"x", agent_2', state)

        assert result.applies is True
        assert result.blocked is True
        assert result.mechanic_type == "restricted_communication"
        assert "only send messages to" in result.observation
        assert result.state.messages_sent.get("agent_0", 0) == 0


def test_bandwidth_applies_after_recipient_check():
    bindings = [
        {"mechanic_type": "limited_bandwidth", "message_limits": {"agent_0": 1}},
        {
            "mechanic_type": "restricted_communication",
            "allowed_targets": {"agent_0": ["agent_1"]},
        },
    ]
    state = _init_state(bindings)

    first = apply_mechanics("Communicate", "agent_0", '"x", agent_1', state)
    assert first.blocked is False
    assert first.state.messages_sent.get("agent_0", 0) == 1

    second = apply_mechanics("Communicate", "agent_0", '"x", agent_1', first.state)
    assert second.blocked is True
    assert second.mechanic_type == "limited_bandwidth"
    assert "used all 1" in second.observation


def test_unreliable_is_checked_after_budget_and_consumes_message():
    bindings = [
        {"mechanic_type": "unreliable_communication", "failure_probability": {"agent_0": 1.0}},
        {"mechanic_type": "limited_bandwidth", "message_limits": {"agent_0": 2}},
        {
            "mechanic_type": "restricted_communication",
            "allowed_targets": {"agent_0": ["agent_1"]},
        },
    ]
    state = _init_state(bindings)

    result = apply_mechanics("Communicate", "agent_0", '"x", agent_1', state)
    assert result.applies is True
    assert result.blocked is True
    assert result.mechanic_type == "unreliable_communication"
    assert "may or may not have been delivered" in result.observation
    assert result.state.messages_sent.get("agent_0", 0) == 1
