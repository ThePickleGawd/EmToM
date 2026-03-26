from emtom.task_gen.session import build_mode_comparison, _build_test_task_retry_guidance


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
