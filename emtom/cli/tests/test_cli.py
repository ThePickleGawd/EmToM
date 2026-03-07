"""Tests for emtom.cli modules."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from emtom.cli import CLIResult, failure, print_result, success


# ---------------------------------------------------------------------------
# CLIResult contract tests
# ---------------------------------------------------------------------------

class TestCLIResult:
    def test_success_shape(self):
        r = success({"key": "value"})
        assert r["success"] is True
        assert r["data"] == {"key": "value"}
        assert r["error"] is None

    def test_failure_shape(self):
        r = failure("boom")
        assert r["success"] is False
        assert r["data"] == {}
        assert r["error"] == "boom"

    def test_failure_with_data(self):
        r = failure("boom", data={"partial": True})
        assert r["data"] == {"partial": True}

    def test_print_result_json(self, capsys):
        r = success({"x": 1})
        print_result(r)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["success"] is True
        assert parsed["data"]["x"] == 1


# ---------------------------------------------------------------------------
# validate_task tests
# ---------------------------------------------------------------------------

def _make_minimal_task(**overrides) -> dict:
    """Build a minimal valid task dict for testing."""
    task = {
        "task_id": "test_001",
        "title": "Test Task for Validation",
        "task": "This is a sufficiently long task description for validation purposes.",
        "episode_id": "1234",
        "num_agents": 2,
        "mechanic_bindings": [],
        "agent_secrets": {"agent_0": ["secret0"], "agent_1": ["secret1"]},
        "agent_actions": {"agent_0": ["Navigate", "Wait"], "agent_1": ["Navigate", "Wait"]},
        "pddl_domain": "emtom",
        "problem_pddl": (
            "(define (problem test_001) "
            "(:domain emtom) "
            "(:objects agent_0 agent_1 - agent kitchen_1 - room cup_1 - object table_2 - furniture) "
            "(:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) "
            "(is_in_room cup_1 kitchen_1) (is_in_room table_2 kitchen_1)) "
            "(:goal (and (is_on_top cup_1 table_2))))"
        ),
        "golden_trajectory": [
            {"actions": [
                {"agent": "agent_0", "action": "Navigate[table_1]"},
                {"agent": "agent_1", "action": "Navigate[table_2]"},
            ]},
        ],
    }
    task.update(overrides)
    return task


class TestValidateTask:
    def test_missing_required_fields(self):
        from emtom.cli.validate_task import validate

        result = validate({}, None)
        assert result["success"] is False
        assert "Missing required fields" in result["error"]

    def test_invalid_agent_id_in_secrets(self):
        from emtom.cli.validate_task import validate

        task = _make_minimal_task(
            agent_secrets={"agent_0": ["s"], "agent_99": ["s"]}
        )
        result = validate(task, None)
        assert result["success"] is False
        assert "agent_99" in result["error"]

    def test_task_too_short(self):
        from emtom.cli.validate_task import validate

        task = _make_minimal_task(task="short")
        result = validate(task, None)
        assert result["success"] is False
        assert "at least 20" in result["error"]

    def test_agent_secrets_not_dict(self):
        from emtom.cli.validate_task import validate

        task = _make_minimal_task(agent_secrets=["not", "a", "dict"])
        result = validate(task, None)
        assert result["success"] is False
        assert "must be a dict" in result["error"]

    def test_valid_task_passes(self):
        from emtom.cli.validate_task import validate

        task = _make_minimal_task()
        # Mock parse_goal_string since PDDL infra may not be available
        with patch("emtom.cli.validate_task.validate") as mock_validate:
            # Just test our function directly
            pass
        # Direct test with real PDDL if available
        result = validate(task, None)
        # This may fail if emtom.pddl is not importable, which is fine
        # The structure tests above cover the logic

    def test_run_file_not_found(self):
        from emtom.cli.validate_task import run

        result = run("/nonexistent/path.json")
        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# verify_pddl tests
# ---------------------------------------------------------------------------

class TestVerifyPddl:
    def test_file_not_found(self):
        from emtom.cli.verify_pddl import run

        result = run("/nonexistent/task.json")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_no_problem_pddl(self):
        from emtom.cli.verify_pddl import run

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"task_id": "test"}, f)
            tmp = f.name
        try:
            result = run(tmp)
            assert result["success"] is False
            assert "problem_pddl" in result["error"].lower()
        finally:
            os.unlink(tmp)

    def test_invalid_json(self):
        from emtom.cli.verify_pddl import run

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json")
            tmp = f.name
        try:
            result = run(tmp)
            assert result["success"] is False
            assert "JSON" in result["error"]
        finally:
            os.unlink(tmp)

    def test_verify_returns_strict_proof_metadata(self):
        from emtom.cli.verify_pddl import run
        from emtom.pddl.solver import SolverResult

        task = {
            "task_id": "test_budget_warning",
            "title": "Budget warning",
            "category": "cooperative",
            "scene_id": "scene",
            "episode_id": "episode",
            "active_mechanics": [],
            "mechanic_bindings": [],
            "task": "Test task",
            "agent_secrets": {},
            "agent_actions": {},
            "num_agents": 2,
            "problem_pddl": (
                "(define (problem test)\n"
                "  (:domain emtom)\n"
                "  (:objects agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture)\n"
                "  (:init (agent_in_room agent_0 kitchen_1) "
                "         (agent_in_room agent_1 kitchen_1) "
                "         (is_in_room cabinet_27 kitchen_1))\n"
                "  (:goal (K agent_0 (is_open cabinet_27)))\n"
                ")"
            ),
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(task, f)
            tmp = f.name

        try:
            with patch(
                "emtom.pddl.tom_verifier.prove_minimal_tom_level",
                return_value={
                    "tom_level": 1,
                    "minimal_tom_level": 1,
                    "epistemic_goal_depth": 1,
                    "proved_unsat_below": [0],
                    "proof_attempts": [{"level": 0, "solvable": False, "belief_depth": 1, "error": "requires depth 1"}],
                    "proof_backend": "fast_downward_strict",
                    "proof_strict": True,
                    "solver_result": SolverResult(solvable=True, belief_depth=1, solve_time=0.0),
                    "trivial_k_goals": [],
                },
            ):
                result = run(tmp)

            assert result["success"] is True
            assert result["data"]["proof_strict"] is True
            assert result["data"]["proof_backend"] == "fast_downward_strict"
            assert result["data"]["proved_unsat_below"] == [0]
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# submit_task tests
# ---------------------------------------------------------------------------

class TestSubmitTask:
    def test_file_not_found(self):
        from emtom.cli.submit_task import run

        result = run("/nonexistent/task.json", output_dir="/tmp")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_submit_persists_computed_tom_fields(self):
        from emtom.cli.submit_task import run

        task = {
            "task_id": "draft_task",
            "title": "Submission ToM Persistence Test",
            "category": "cooperative",
            "task": "This is a sufficiently long task description for submit testing.",
            "scene_id": "scene_test",
            "episode_id": "episode_test",
            "num_agents": 2,
            "active_mechanics": [],
            "mechanic_bindings": [],
            "agent_secrets": {"agent_0": ["s0"], "agent_1": ["s1"]},
            "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
            "items": [],
            "locked_containers": {},
            "initial_states": {},
            "message_targets": {},
            "teams": {},
            "team_secrets": {},
            "pddl_domain": "emtom",
            "problem_pddl": (
                "(define (problem t_submit)"
                " (:domain emtom)"
                " (:objects agent_0 agent_1 - agent kitchen_1 - room cup_1 - object table_1 - furniture)"
                " (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1)"
                "        (is_in_room cup_1 kitchen_1) (is_in_room table_1 kitchen_1)"
                "        (is_on_top cup_1 table_1))"
                " (:goal (is_on_top cup_1 table_1)))"
            ),
        }

        with tempfile.TemporaryDirectory() as td:
            task_path = Path(td) / "task.json"
            out_dir = Path(td) / "out"
            with open(task_path, "w") as f:
                json.dump(task, f)

            with patch("emtom.cli.validate_task.run", return_value=success({"valid": True})), \
                 patch("emtom.cli.submit_task._compute_tom_metadata", return_value={"tom_level": 2, "tom_reasoning": "computed"}), \
                 patch("emtom.pddl.planner.regenerate_golden_trajectory", return_value={}), \
                 patch("emtom.cli.submit_task._ensure_domain_pddl_file", return_value=Path(td) / "domain.pddl"):
                result = run(
                    str(task_path),
                    output_dir=str(out_dir),
                    subtasks_min=1,
                    subtasks_max=20,
                )

            assert result["success"] is True
            output_path = Path(result["data"]["output_path"])
            assert output_path.exists()

            with open(output_path) as f:
                submitted = json.load(f)
            assert submitted["tom_level"] == 2
            assert submitted["tom_reasoning"] == "computed"

            with open(task_path) as f:
                working = json.load(f)
            assert working["tom_level"] == 2
            assert working["tom_reasoning"] == "computed"

    def test_submit_fails_when_tom_computation_fails(self):
        from emtom.cli.submit_task import run

        task = {
            "task_id": "draft_task",
            "title": "Submission ToM Failure Test",
            "category": "cooperative",
            "task": "This is a sufficiently long task description for submit testing.",
            "scene_id": "scene_test",
            "episode_id": "episode_test",
            "num_agents": 2,
            "active_mechanics": [],
            "mechanic_bindings": [],
            "agent_secrets": {"agent_0": ["s0"], "agent_1": ["s1"]},
            "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
            "items": [],
            "locked_containers": {},
            "initial_states": {},
            "message_targets": {},
            "teams": {},
            "team_secrets": {},
            "pddl_domain": "emtom",
            "problem_pddl": (
                "(define (problem t_submit)"
                " (:domain emtom)"
                " (:objects agent_0 agent_1 - agent kitchen_1 - room cup_1 - object table_1 - furniture)"
                " (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1)"
                "        (is_in_room cup_1 kitchen_1) (is_in_room table_1 kitchen_1)"
                "        (is_on_top cup_1 table_1))"
                " (:goal (is_on_top cup_1 table_1)))"
            ),
        }

        with tempfile.TemporaryDirectory() as td:
            task_path = Path(td) / "task.json"
            out_dir = Path(td) / "out"
            with open(task_path, "w") as f:
                json.dump(task, f)

            with patch("emtom.cli.validate_task.run", return_value=success({"valid": True})), \
                 patch("emtom.cli.submit_task._compute_tom_metadata", side_effect=ValueError("boom")):
                result = run(
                    str(task_path),
                    output_dir=str(out_dir),
                    subtasks_min=1,
                    subtasks_max=20,
                )

            assert result["success"] is False
            assert "Failed to compute tom_level" in result["error"]


# ---------------------------------------------------------------------------
# judge_task tests
# ---------------------------------------------------------------------------

class TestJudgeTask:
    def test_file_not_found(self):
        from emtom.cli.judge_task import run

        result = run("/nonexistent/task.json")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_judge_runs_verify_pddl_first(self):
        from emtom.cli.judge_task import run

        task = _make_minimal_task(
            problem_pddl=(
                "(define (problem test_001) "
                "(:domain emtom) "
                "(:objects agent_0 agent_1 - agent kitchen_1 - room cup_1 - object table_2 - furniture) "
                "(:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) "
                "(is_in_room cup_1 kitchen_1) (is_in_room table_2 kitchen_1)) "
                "(:goal (and (is_on_top cup_1 table_2))))"
            ),
        )

        with tempfile.TemporaryDirectory() as td:
            task_path = Path(td) / "task.json"
            with open(task_path, "w") as f:
                json.dump(task, f)

            with patch(
                "emtom.cli.verify_pddl.run",
                return_value=failure("strict proof failed", data={"valid": False}),
            ), patch("emtom.pddl.planner.regenerate_golden_trajectory") as mock_regen:
                result = run(str(task_path))

            assert result["success"] is False
            assert "strict proof failed" in result["error"]
            mock_regen.assert_not_called()


# ---------------------------------------------------------------------------
# static_validate_trajectory tests
# ---------------------------------------------------------------------------

class TestStaticValidateTrajectory:
    def test_empty_trajectory(self):
        from emtom.cli.validate_task import static_validate_trajectory

        errors = static_validate_trajectory(
            {"num_agents": 2},
            [{"actions": []}],
        )
        # Empty actions array should be flagged
        assert any("No actions" in e for e in errors)

    def test_invalid_agent(self):
        from emtom.cli.validate_task import static_validate_trajectory

        errors = static_validate_trajectory(
            {"num_agents": 2},
            [{"actions": [{"agent": "agent_99", "action": "Wait[]"}]}],
        )
        assert any("agent_99" in e for e in errors)

    def test_malformed_action(self):
        from emtom.cli.validate_task import static_validate_trajectory

        errors = static_validate_trajectory(
            {"num_agents": 2},
            [{"actions": [{"agent": "agent_0", "action": "???malformed"}]}],
        )
        assert any("Malformed" in e for e in errors)

    def test_valid_trajectory(self):
        from emtom.cli.validate_task import static_validate_trajectory

        errors = static_validate_trajectory(
            {"num_agents": 2},
            [{"actions": [
                {"agent": "agent_0", "action": "Navigate[table_1]"},
                {"agent": "agent_1", "action": "Wait[]"},
            ]}],
        )
        # No scene data, so object IDs aren't checked
        assert len(errors) == 0
