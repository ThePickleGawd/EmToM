from emtom.task_gen.spec_validator import validate_blocking_spec


def _base_task() -> dict:
    return {
        "task_id": "test-task",
        "title": "Test Task",
        "task": "This is a sufficiently long task description.",
        "episode_id": "episode_1",
        "num_agents": 2,
        "mechanic_bindings": [],
        "agent_secrets": {
            "agent_0": [],
            "agent_1": [],
        },
        "agent_actions": {
            "agent_0": ["Communicate", "Wait", "Open"],
            "agent_1": ["Communicate", "Wait", "Open"],
        },
    }


def test_communicate_recipient_counts_as_active_agent() -> None:
    task = _base_task()
    task["golden_trajectory"] = [
        {
            "actions": [
                {"agent": "agent_0", "action": 'Communicate["stand_1 is open", agent_1]'},
                {"agent": "agent_1", "action": "Wait[]"},
            ]
        }
    ]

    errors = validate_blocking_spec(task)

    assert not any("only one active agent" in error for error in errors)


def test_single_actor_without_recipient_still_fails_multi_agent_guard() -> None:
    task = _base_task()
    task["golden_trajectory"] = [
        {
            "actions": [
                {"agent": "agent_0", "action": "Open[stand_1]"},
                {"agent": "agent_1", "action": "Wait[]"},
            ]
        }
    ]

    errors = validate_blocking_spec(task)

    assert any("only one active agent" in error for error in errors)
