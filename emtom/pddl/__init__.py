"""
PDDL + Epistemic Extensions for EmToM.

Replaces the hand-crafted subtask DAG with formal PDDL goal specifications.
ToM level and human-readable descriptions are derived from the PDDL, not stored redundantly.
"""

from emtom.pddl.dsl import (
    Type,
    Predicate,
    Param,
    Formula,
    Literal,
    And,
    Or,
    Not,
    EpistemicFormula,
    Knows,
    Believes,
    Effect,
    Action,
    Problem,
    Domain,
)
from emtom.pddl.epistemic import ObservabilityModel
from emtom.pddl.compiler import compile_task
from emtom.pddl.goal_checker import PDDLGoalChecker
from emtom.pddl.describe import describe_task
from emtom.pddl.goal_spec import GoalEntry, GoalSpec
from emtom.pddl.problem_pddl import (
    ParsedProblemPDDL,
    parse_problem_pddl,
    extract_goal_from_problem_pddl,
    replace_goal_in_problem_pddl,
)
from emtom.pddl.runtime_projection import LiteralToMProbe, RuntimeProjection

__all__ = [
    "Type",
    "Predicate",
    "Param",
    "Formula",
    "Literal",
    "And",
    "Or",
    "Not",
    "EpistemicFormula",
    "Knows",
    "Believes",
    "Effect",
    "Action",
    "Problem",
    "Domain",
    "ObservabilityModel",
    "compile_task",
    "PDDLGoalChecker",
    "describe_task",
    "GoalEntry",
    "GoalSpec",
    "ParsedProblemPDDL",
    "parse_problem_pddl",
    "extract_goal_from_problem_pddl",
    "replace_goal_in_problem_pddl",
    "LiteralToMProbe",
    "RuntimeProjection",
]
