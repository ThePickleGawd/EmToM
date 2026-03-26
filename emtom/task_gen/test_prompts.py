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
    assert "Read `/tmp/taskgen/sampled_tasks/SUMMARY.md` first" not in prompt
