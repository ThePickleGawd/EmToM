from emtom.pddl.planner import generate_deterministic_trajectory


def _actions(result):
    return [
        action["action"]
        for step in result["trajectory"]
        for action in step["actions"]
        if action["action"] != "Wait[]"
    ]


def test_open_inserts_target_navigation_before_interaction():
    task_data = {
        "task_id": "open_target_nav",
        "title": "Open target navigation",
        "category": "cooperative",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 1,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem open_target_nav)\n"
            "  (:domain emtom)\n"
            "  (:objects\n"
            "    agent_0 - agent\n"
            "    chest_of_drawers_27 - furniture\n"
            "    bedroom_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 bedroom_1)\n"
            "    (is_in_room chest_of_drawers_27 bedroom_1)\n"
            "    (is_closed chest_of_drawers_27)\n"
            "  )\n"
            "  (:goal (is_open chest_of_drawers_27))\n"
            ")"
        ),
        "items": [],
        "locked_containers": {},
        "initial_states": {},
    }

    result = generate_deterministic_trajectory(task_data)

    assert _actions(result) == [
        "Navigate[chest_of_drawers_27]",
        "Open[chest_of_drawers_27]",
    ]


def test_use_item_inserts_target_navigation_before_interaction():
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
