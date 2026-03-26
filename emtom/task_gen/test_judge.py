from emtom.task_gen.judge import _analyze_id_leakage


def test_analyze_id_leakage_flags_public_task_and_ignorance_secret_object_ids():
    task = {
        "task": "Put vase_0 on the living-room table.",
        "agent_secrets": {
            "agent_0": ["You do not know where vase_0 currently is."],
            "agent_1": ["Before the task starts, you saw that vase_0 is on bed_7."],
        },
        "problem_pddl": """(define (problem leak-test)
  (:domain emtom)
  (:objects
    agent_0 agent_1 - agent
    living_room_1 bedroom_1 - room
    vase_0 - object
    table_9 bed_7 - furniture
  )
  (:init)
  (:goal (and (is_on_top vase_0 table_9)))
)""",
    }

    leakage = _analyze_id_leakage(task)

    assert leakage["public_task_object_ids"] == ["vase_0"]
    assert leakage["ignorance_secret_ids"] == ["vase_0"]
