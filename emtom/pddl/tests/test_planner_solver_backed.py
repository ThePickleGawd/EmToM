"""Regression tests for the solver-backed golden trajectory generator."""

from emtom.pddl.planner import generate_deterministic_trajectory


def _actions(result):
    return [
        action["action"]
        for step in result["trajectory"]
        for action in step["actions"]
        if action["action"] != "Wait[]"
    ]


def test_or_goal_uses_feasible_branch():
    task_data = {
        "task_id": "or_branch",
        "title": "OR branch",
        "category": "competitive",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 2,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem or_branch)\n"
            "  (:domain emtom)\n"
            "  (:objects\n"
            "    agent_0 agent_1 - agent\n"
            "    cabinet_27 table_5 - furniture\n"
            "    kitchen_1 bedroom_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 bedroom_1)\n"
            "    (agent_in_room agent_1 bedroom_1)\n"
            "    (is_in_room cabinet_27 kitchen_1)\n"
            "    (is_in_room table_5 bedroom_1)\n"
            "    (is_closed cabinet_27)\n"
            "    (is_restricted agent_0 kitchen_1)\n"
            "  )\n"
            "  (:goal (or (agent_in_room agent_0 kitchen_1) (is_open cabinet_27)))\n"
            ")"
        ),
        "items": [],
        "locked_containers": {},
        "initial_states": {},
    }

    result = generate_deterministic_trajectory(task_data)
    actions = _actions(result)

    assert "Open[cabinet_27]" in actions
    assert result["ignored_literals"] == []


def test_use_item_plan_translates_directly():
    task_data = {
        "task_id": "unlock_with_item",
        "title": "Unlock With Item",
        "category": "cooperative",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 1,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem unlock_with_item)\n"
            "  (:domain emtom)\n"
            "  (:objects\n"
            "    agent_0 - agent\n"
            "    item_small_key_1 - item\n"
            "    cabinet_27 - furniture\n"
            "    kitchen_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 kitchen_1)\n"
            "    (is_in_room cabinet_27 kitchen_1)\n"
            "    (has_item agent_0 item_small_key_1)\n"
            "    (requires_item cabinet_27 item_small_key_1)\n"
            "  )\n"
            "  (:goal (is_unlocked cabinet_27))\n"
            ")"
        ),
        "items": [],
        "locked_containers": {},
        "initial_states": {},
    }

    result = generate_deterministic_trajectory(task_data)

    assert _actions(result) == [
        "Navigate[cabinet_27]",
        "UseItem[item_small_key_1, cabinet_27]",
    ]
    assert result["ignored_literals"] == []


def test_rejects_solver_plans_that_move_furniture():
    task_data = {
        "task_id": "invalid_pick",
        "title": "Invalid Pick",
        "category": "cooperative",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 2,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem invalid_pick)\n"
            "  (:domain emtom)\n"
            "  (:objects\n"
            "    agent_0 agent_1 - agent\n"
            "    mug_1 - object\n"
            "    source_table handoff_table shelf_1 - furniture\n"
            "    kitchen_1 living_room_1 bedroom_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 kitchen_1)\n"
            "    (agent_in_room agent_1 bedroom_1)\n"
            "    (is_in_room mug_1 kitchen_1)\n"
            "    (is_in_room source_table kitchen_1)\n"
            "    (is_in_room handoff_table living_room_1)\n"
            "    (is_in_room shelf_1 bedroom_1)\n"
            "    (is_on_top mug_1 source_table)\n"
            "    (is_restricted agent_0 bedroom_1)\n"
            "    (is_restricted agent_1 kitchen_1)\n"
            "  )\n"
            "  (:goal (is_on_top mug_1 shelf_1))\n"
            ")"
        ),
        "items": [],
        "locked_containers": {},
        "initial_states": {},
    }

    try:
        generate_deterministic_trajectory(task_data)
    except RuntimeError as exc:
        assert "runtime only supports movable objects" in str(exc)
    else:
        raise AssertionError("expected invalid furniture-moving plan to be rejected")
