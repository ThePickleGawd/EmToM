from emtom.task_gen.session import (
    _aggregate_competitive_baseline_phase_results,
    _build_test_task_retry_guidance,
    build_mode_comparison,
)


def test_build_mode_comparison_uses_count_based_calibration_gate():
    comparison = build_mode_comparison(
        category="cooperative",
        standard={"evaluation": {"success": True, "percent_complete": 1.0}},
        baseline={"evaluation": {"success": True, "percent_complete": 1.0}},
        current_rate=0.43,
        target_rate=0.20,
        current_passed=51,
        current_failed=66,
    )

    assert comparison["standard_requirement"] == "must_fail"
    assert comparison["gate_passed"] is False
    assert comparison["next_standard_pass_rate_if_fail"] < comparison["next_standard_pass_rate_if_pass"]


def test_build_mode_comparison_can_require_a_pass_when_that_moves_toward_target():
    comparison = build_mode_comparison(
        category="cooperative",
        standard={"evaluation": {"success": False, "percent_complete": 0.0}},
        baseline={"evaluation": {"success": True, "percent_complete": 1.0}},
        current_rate=0.0,
        target_rate=0.20,
        current_passed=0,
        current_failed=2,
    )

    assert comparison["standard_requirement"] == "must_pass"
    assert comparison["gate_passed"] is False


def test_test_task_retry_guidance_explains_too_easy_case():
    guidance = _build_test_task_retry_guidance(
        {
            "standard_requirement": "must_fail",
            "standard_passed": True,
            "baseline_passed": True,
        }
    )

    assert "Do not call `taskgen fail`" in guidance
    assert "too easy for standard mode" in guidance
    assert "non-binary" in guidance


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
        target_rate=0.20,
        current_passed=1,
        current_failed=4,
    )

    assert comparison["baseline_passed"] is False
    assert comparison["baseline_failed_phases"] == ["team_1_only"]
    assert "both solo-team phases" in comparison["reasons"][0]
