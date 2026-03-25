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
        prove_minimal_tom_level,
    )
    from emtom.task_gen.task_generator import GeneratedTask

    generated = GeneratedTask.from_dict(task_data)
    proof = prove_minimal_tom_level(generated, scene_data=scene_data, strict=False)
    # If taskgen runs with a lightweight synthetic scene (or otherwise without
    # observability grounding), the verifier cannot reliably prove minimal ToM
    # levels above 1 even when the authored goals contain deeper nesting.
    #
    # NOTE: scene_data may be either a plain dict (common in JSON workflows) or
    # a SceneData dataclass instance (taskgen internal loader). Avoid dict-only
    # APIs like `.get()`.
    def _rooms_empty(sd: Any) -> bool:
        if sd is None:
            return True
        if isinstance(sd, dict):
            return not bool(sd.get("rooms"))
        rooms = getattr(sd, "rooms", None)
        return not bool(rooms)

    if _rooms_empty(scene_data):
        depth = proof.get("epistemic_goal_depth")
        if isinstance(depth, int) and depth > 0:
            return {
                "tom_level": depth,
                "epistemic_goal_depth": depth,
                "proved_unsat_below": proof.get("proved_unsat_below", []),
                "proof_backend": "synthetic_scene_fallback",
                "proof_strict": proof.get("proof_strict", False),
                "tom_reasoning": (
                    "No scene observability data available (synthetic/minimal scene); "
                    "falling back to authored epistemic nesting depth."
                ),
            }
    solver_result = proof.get("solver_result")
    # In structural-only environments, the solver cannot establish minimal ToM level.
    # Fall back to the authored epistemic goal nesting depth as tom_level.
    if proof.get("proof_backend") == "pdkb_structural":
        depth = proof.get("epistemic_goal_depth")
        if isinstance(depth, int) and depth > 0:
            return {
                "tom_level": depth,
                "epistemic_goal_depth": depth,
                "proved_unsat_below": proof.get("proved_unsat_below", []),
                "proof_backend": proof.get("proof_backend"),
                "proof_strict": proof.get("proof_strict", False),
                "tom_reasoning": "Structural-only fallback: using authored epistemic nesting depth.",
            }
    if solver_result is None:
        # Competitive OR-goals are structurally incompatible with the epistemic
        # solver (it assumes a single cooperative objective). Fall back to the
        # syntactic epistemic nesting depth so competitive tasks can still carry
        # K-level metadata.
        category = task_data.get("category", "")
        depth = proof.get("epistemic_goal_depth")
        if category == "competitive" and isinstance(depth, int) and depth > 0:
            return {
                "tom_level": depth,
                "epistemic_goal_depth": depth,
                "proved_unsat_below": proof.get("proved_unsat_below", []),
                "proof_backend": "competitive_syntactic_fallback",
                "proof_strict": False,
                "tom_reasoning": (
                    "Competitive OR-goal structure is incompatible with epistemic "
                    "solver; using authored epistemic nesting depth."
                ),
            }
        last_error = (
            proof["proof_attempts"][-1]["error"]
            if proof.get("proof_attempts")
            else "unknown reason"
        )
        raise ValueError(f"PDDL goal is not solvable: {last_error}")

    tom_info = explain_tom_depth(
        generated,
        scene_data,
        solver_result=solver_result,
        proof=proof,
    )
    tom_info["epistemic_goal_depth"] = proof["epistemic_goal_depth"]
    tom_info["proved_unsat_below"] = proof["proved_unsat_below"]
    tom_info["proof_backend"] = proof["proof_backend"]
    tom_info["proof_strict"] = proof["proof_strict"]
    tom_level = tom_info.get("tom_level")
    if not isinstance(tom_level, int):
        raise ValueError(f"Invalid computed tom_level: {tom_level!r}")

    result: Dict[str, Any] = {
        "tom_level": tom_level,
        "epistemic_goal_depth": tom_info["epistemic_goal_depth"],
        "proved_unsat_below": tom_info["proved_unsat_below"],
        "proof_backend": tom_info["proof_backend"],
        "proof_strict": tom_info["proof_strict"],
    }
    tom_reasoning = tom_info.get("tom_reasoning")
    if isinstance(tom_reasoning, str) and tom_reasoning.strip():
        result["tom_reasoning"] = tom_reasoning
    return result
