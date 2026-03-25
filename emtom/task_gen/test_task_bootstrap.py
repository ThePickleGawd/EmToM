from emtom.pddl.problem_pddl import parse_problem_pddl, validate_problem_pddl_self_contained
from emtom.task_gen.task_bootstrap import build_scene_bootstrap_problem_pddl


def _scene(agent_spawns):
    return {
        "scene_id": "102817140",
        "episode_id": "1940",
        "rooms": ["kitchen_1", "hallway_2"],
        "furniture": ["table_11", "cabinet_18"],
        "objects": ["laptop_0", "mug_2", "orphan_9"],
        "articulated_furniture": ["cabinet_18"],
        "furniture_in_rooms": {
            "kitchen_1": ["table_11"],
            "hallway_2": ["cabinet_18"],
        },
        "objects_on_furniture": {
            "table_11": ["laptop_0"],
            "cabinet_18": ["mug_2"],
        },
        "agent_spawns": agent_spawns,
    }


def test_bootstrap_problem_is_self_contained_when_spawn_rooms_are_available() -> None:
    problem_pddl = build_scene_bootstrap_problem_pddl(
        _scene({"agent_0": "kitchen_1", "agent_1": "hallway_2"}),
        2,
        problem_name="scene_102817140",
    )

    parsed = parse_problem_pddl(problem_pddl)
    errors = validate_problem_pddl_self_contained(parsed, num_agents=2)

    assert errors == []
    assert "orphan_9" not in problem_pddl
    assert "(agent_in_room agent_0 kitchen_1)" in problem_pddl
    assert "(agent_in_room agent_1 hallway_2)" in problem_pddl
    assert "(is_in_room cabinet_18 hallway_2)" in problem_pddl
    assert "(is_in_room laptop_0 kitchen_1)" in problem_pddl
    assert "(is_on_top mug_2 cabinet_18)" in problem_pddl


def test_bootstrap_problem_falls_back_to_scene_rooms_for_position_spawns() -> None:
    problem_pddl = build_scene_bootstrap_problem_pddl(
        _scene(
            {
                "agent_0": {"position": [0, 0, 0], "rotation": 0.0},
                "agent_1": {"position": [1, 0, 1], "rotation": 1.57},
            }
        ),
        2,
        problem_name="scene_102817140",
    )

    parsed = parse_problem_pddl(problem_pddl)
    errors = validate_problem_pddl_self_contained(parsed, num_agents=2)

    assert errors == []
    assert "(agent_in_room agent_0 kitchen_1)" in problem_pddl
    assert "(agent_in_room agent_1 hallway_2)" in problem_pddl
