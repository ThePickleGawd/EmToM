from emtom.task_gen.prompts import build_external_taskgen_prompt


def test_build_external_taskgen_prompt_inlines_runtime_context():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=4,
        subtasks_min=3,
        subtasks_max=6,
        query="Keep it grounded in the scene.",
        verification_feedback={
            "overall_reasoning": "The prior task leaked the plan in secrets.",
            "required_fixes": ["Remove prescriptive language."],
        },
        calibration_stats={
            "model": "gpt-5.2",
            "target_rate": 0.20,
            "rate": 0.45,
        },
        current_k_level=2,
        seed_tasks_dir="/tmp/taskgen/sampled_tasks",
        seed_pass_ratio=0.20,
        seed_fail_ratio=0.80,
    )

    assert "## User Requirements" in prompt
    assert "## Previous ToM Verification Failed" in prompt
    assert prompt.count("## Dataset Calibration") == 1
    assert "## Required K-Level: 2" in prompt
    assert "## Sampled Task Context" in prompt
    assert "## Required Commands" in prompt
    assert "1. Run `taskgen status`." in prompt
    assert "2. Run `taskgen new_scene N`" in prompt
    assert "Read all 10 `task_*_fields.json`" in prompt
    assert "`task`, `active_mechanics`, `mechanic_bindings`, `agent_secrets`, `agent_actions`, `problem_pddl`, and `num_agents`" in prompt
    assert "Open the matching raw `task_*.json` only" in prompt
    assert "extra_sections" not in prompt


def test_build_external_taskgen_prompt_omits_sampled_task_context_when_evolution_removed():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="mixed",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=5,
        calibration_stats={},
        current_k_level=1,
        seed_tasks_dir="/tmp/taskgen/sampled_tasks",
        skip_steps=["task-evolution"],
    )

    assert "## Sampled Task Context" not in prompt
    assert "task_*_fields.json" not in prompt


def test_build_external_taskgen_prompt_does_not_push_easy_tasks():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=4,
        difficulty="easy",
    )

    assert "## Difficulty: EASY" not in prompt
    assert "Generate SIMPLE tasks" not in prompt
    assert "do not weaken secrets" in prompt


def test_build_external_taskgen_prompt_warns_against_hidden_object_id_leaks():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=4,
    )

    assert "do NOT reveal the exact runtime object ID" in prompt
    assert "Do NOT leak hidden target object IDs" in prompt
    assert "NEVER add ignorance lines like 'You do not know where ...'" in prompt
    assert "NEVER add epistemic coaching like 'By the end, you must be confident ...'" in prompt
