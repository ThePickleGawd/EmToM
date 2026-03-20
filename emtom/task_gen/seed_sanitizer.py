"""Helpers for exposing seed/example tasks safely under literal-ToM semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_TITLE_PLACEHOLDER = "Rewrite Title For Current Scene"
_TASK_PLACEHOLDER = (
    "Rewrite the public task from scratch for the current scene. Describe only "
    "the functional shared objective in natural language. Do not mention K() "
    "goals, probes, or knowledge as a pass/fail requirement."
)
_AGENT_SECRET_PLACEHOLDER = (
    "Rewrite from scratch for the current scene. Mention team membership, "
    "access/communication constraints, mechanic hints, and relevant private "
    "information. Use exact scene IDs for goal-critical targets. Do not say "
    "that knowledge is required for task success."
)
_TEAM_SECRET_PLACEHOLDER = (
    "Rewrite from scratch for the current scene. Mention team-level private "
    "context if needed, using exact scene IDs when the team needs precise "
    "grounding, but do not describe K() knowledge as a runtime success "
    "condition."
)

_DERIVED_FIELDS_TO_DROP = (
    "functional_goal_pddl",
    "literal_tom_probes",
    "runtime_semantics_version",
    "golden_trajectory",
    "tom_level",
    "tom_reasoning",
    "calibration",
    "judge",
    "benchmark_results",
)


def sanitize_task_for_seeding(
    task_data: Dict[str, Any],
    *,
    num_agents: int | None = None,
) -> Dict[str, Any]:
    """Return a seed-safe task copy with natural-language fields reset.

    The generator still gets structural fields such as category, mechanics,
    communication graph, and PDDL. Public/secret language and derived runtime
    metadata are cleared so stale semantics do not leak into new generations.
    """

    sanitized = deepcopy(task_data)

    for key in _DERIVED_FIELDS_TO_DROP:
        sanitized.pop(key, None)

    if num_agents is None:
        raw_num_agents = sanitized.get("num_agents")
        num_agents = raw_num_agents if isinstance(raw_num_agents, int) and raw_num_agents > 0 else 2

    sanitized["title"] = _TITLE_PLACEHOLDER
    sanitized["task"] = _TASK_PLACEHOLDER

    sanitized["agent_secrets"] = {
        f"agent_{idx}": [_AGENT_SECRET_PLACEHOLDER]
        for idx in range(num_agents)
    }

    if "team_secrets" in sanitized:
        team_secrets = sanitized.get("team_secrets")
        if isinstance(team_secrets, dict):
            sanitized["team_secrets"] = {
                team_id: [_TEAM_SECRET_PLACEHOLDER]
                for team_id in team_secrets
            }
        else:
            sanitized["team_secrets"] = {}

    return sanitized
