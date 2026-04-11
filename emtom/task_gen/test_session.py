from emtom.task_gen.session import (
    _aggregate_competitive_baseline_phase_results,
    _build_submission_verification_feedback,
    _build_test_task_failure_feedback,
    _build_test_task_retry_guidance,
    build_mode_comparison,
)


def test_build_mode_comparison_uses_count_based_calibration_gate():
    comparison = build_mode_comparison(
        category="cooperative",
        standard={"evaluation": {"success": True, "percent_complete": 1.0}},
        baseline={"evaluation": {"success": True, "percent_complete": 1.0}},
        current_rate=0.43,
        target_rate=0.10,
        current_passed=51,
        current_failed=66,
    )

    assert comparison["standard_requirement"] == "must_fail"
    assert comparison["gate_passed"] is True
    assert comparison["next_standard_pass_rate_if_fail"] < comparison["next_standard_pass_rate_if_pass"]
    assert "Calibration note: standard passed" in comparison["reasons"][0]


def test_build_mode_comparison_can_require_a_pass_when_that_moves_toward_target():
    comparison = build_mode_comparison(
        category="cooperative",
        standard={"evaluation": {"success": False, "percent_complete": 0.0}},
        baseline={"evaluation": {"success": True, "percent_complete": 1.0}},
        current_rate=0.0,
        target_rate=0.10,
        current_passed=0,
        current_failed=20,
    )

    assert comparison["standard_requirement"] == "must_pass"
    assert comparison["gate_passed"] is True
    assert "Calibration note: standard failed" in comparison["reasons"][0]


def test_test_task_retry_guidance_returns_generic_retry_for_non_baseline_failures():
    guidance = _build_test_task_retry_guidance(
        {
            "standard_requirement": "must_fail",
            "standard_passed": True,
            "baseline_passed": True,
        }
    )

    assert "Do not call `taskgen fail`" in guidance
    assert "run `taskgen judge` -> `taskgen test_task` again" in guidance


def test_test_task_retry_guidance_explains_unsolved_baseline_case():
    guidance = _build_test_task_retry_guidance(
        {
            "standard_requirement": "either",
            "standard_passed": False,
            "baseline_passed": False,
        }
    )

    assert "Baseline also failed" in guidance
    assert "physically broken" in guidance


def test_competitive_baseline_requires_both_single_team_phases():
    baseline = _aggregate_competitive_baseline_phase_results(
        {
            "team_0_only": {
                "active_team": "team_0",
                "idle_team": "team_1",
                "done": True,
                "steps": 12,
                "turns": 4,
                "evaluation": {
                    "winner": "team_0",
                    "team_status": {"team_0": True, "team_1": False},
                    "team_progress": {"team_0": 1.0, "team_1": 0.0},
                },
            },
            "team_1_only": {
                "active_team": "team_1",
                "idle_team": "team_0",
                "done": True,
                "steps": 15,
                "turns": 5,
                "evaluation": {
                    "winner": None,
                    "team_status": {"team_0": False, "team_1": False},
                    "team_progress": {"team_0": 0.0, "team_1": 0.4},
                },
            },
        }
    )

    assert baseline["evaluation"]["success"] is False
    assert baseline["evaluation"]["percent_complete"] == 0.4
    assert baseline["evaluation"]["phase_results"]["team_0_only"]["passed"] is True
    assert baseline["evaluation"]["phase_results"]["team_1_only"]["passed"] is False


def test_build_mode_comparison_reports_failed_competitive_baseline_phases():
    comparison = build_mode_comparison(
        category="competitive",
        standard={"evaluation": {"winner": None, "percent_complete": 0.0}},
        baseline={
            "evaluation": {
                "success": False,
                "percent_complete": 0.5,
                "phase_results": {
                    "team_0_only": {"passed": True},
                    "team_1_only": {"passed": False},
                },
            }
        },
        current_rate=0.20,
        target_rate=0.10,
        current_passed=1,
        current_failed=4,
    )

    assert comparison["baseline_passed"] is False
    assert comparison["baseline_failed_phases"] == ["team_1_only"]
    assert "both solo-team phases" in comparison["reasons"][0]


def test_build_test_task_failure_feedback_highlights_competitive_phase_failures():
    feedback = _build_test_task_failure_feedback(
        "competitive",
        {
            "baseline_passed": False,
            "standard_passed": True,
            "standard_progress": 1.0,
            "baseline_progress": 0.75,
            "standard_turns": 3,
            "baseline_turns": 28,
            "baseline_failed_phases": ["team_1_only"],
            "baseline_phase_results": {
                "team_0_only": {"passed": True, "progress": 1.0, "turns": 8},
                "team_1_only": {"passed": False, "progress": 0.75, "turns": 20},
            },
            "reasons": [
                "Competitive baseline must pass in both solo-team phases (team_0_only and team_1_only). Failed phases: team_1_only."
            ],
        },
        "/tmp/taskgen/run_3",
    )

    assert feedback["source_gate"] == "test_task"
    assert "Competitive solo-team baseline failed" in feedback["summary"]
    assert any("Failed phases to unblock first: team_1_only." == fix for fix in feedback["required_fixes"])
    assert any("team_1_only: passed=False" in item for item in feedback["evidence"])
    assert feedback["artifact_paths"]["comparison_json"].endswith("/tmp/taskgen/run_3/comparison.json")


def test_build_submission_verification_feedback_reports_passing_models():
    feedback = _build_submission_verification_feedback(
        "cooperative",
        {
            "required_failures": 1,
            "message": "Submission verification failed. Need at least 1/3 verification models to fail.",
            "passed_models": ["gpt-5.4", "claude-sonnet-4-6", "gemini-flash"],
            "failed_models": [],
            "trajectory_dir": "/tmp/taskgen/verification_2",
            "models": {
                "gpt-5.4": {"passed": True, "progress": 1.0, "turns": 5},
                "claude-sonnet-4-6": {"passed": True, "progress": 0.6, "turns": 20},
                "gemini-flash": {"passed": True, "progress": 1.0, "turns": 8},
            },
        },
    )

    assert feedback["source_gate"] == "verify_task"
    assert "3/3 verification models still solved it" in feedback["summary"]
    assert any("gpt-5.4: passed=True" in item for item in feedback["evidence"])
    assert any("gemini-flash solved the task quickly" in fix for fix in feedback["required_fixes"])
    assert feedback["artifact_paths"]["verification_summary_json"].endswith(
        "/tmp/taskgen/verification_2/verification_summary.json"
    )
