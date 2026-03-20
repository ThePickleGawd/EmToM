"""Helpers for strict task metadata checks shared across CLI surfaces."""

from __future__ import annotations

from typing import Any, Dict, Optional


def compute_strict_tom_metadata(
    task_data: Dict[str, Any],
    scene_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute authoritative ToM metadata from canonical problem_pddl."""
    from emtom.pddl.tom_verifier import (
        explain_tom_depth,
        generate_tom_reasoning,
        prove_minimal_tom_level,
    )
    from emtom.task_gen.task_generator import GeneratedTask

    generated = GeneratedTask.from_dict(task_data)
    proof = prove_minimal_tom_level(generated, scene_data=scene_data, strict=True)
    solver_result = proof.get("solver_result")
    if solver_result is None:
        last_error = (
            proof["proof_attempts"][-1]["error"]
            if proof.get("proof_attempts")
            else "unknown reason"
        )
        raise ValueError(f"PDDL goal is not solvable: {last_error}")

    tom_info = explain_tom_depth(generated, scene_data, solver_result=solver_result)
    tom_info["epistemic_goal_depth"] = proof["epistemic_goal_depth"]
    tom_info["proved_unsat_below"] = proof["proved_unsat_below"]
    tom_info["proof_backend"] = proof["proof_backend"]
    tom_info["proof_strict"] = proof["proof_strict"]
    tom_level = tom_info.get("tom_level")
    if not isinstance(tom_level, int):
        raise ValueError(f"Invalid computed tom_level: {tom_level!r}")

    information_gaps = tom_info.get("information_gaps", [])
    tom_reasoning = generate_tom_reasoning(
        task_data,
        tom_level=tom_level,
        information_gaps=information_gaps,
    )

    result: Dict[str, Any] = {
        "tom_level": tom_level,
        "epistemic_goal_depth": tom_info["epistemic_goal_depth"],
        "proved_unsat_below": tom_info["proved_unsat_below"],
        "proof_backend": tom_info["proof_backend"],
        "proof_strict": tom_info["proof_strict"],
    }
    if isinstance(tom_reasoning, str) and tom_reasoning.strip():
        result["tom_reasoning"] = tom_reasoning
    return result
