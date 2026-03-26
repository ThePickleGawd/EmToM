from emtom.task_gen.session import build_mode_comparison


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
