"""
Utilities for inline task-level PDDL problem strings.

`problem_pddl` is stored inside task JSON as the single authoritative
problem specification. This module parses that string into the existing
DSL dataclasses used by the solver/checker stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

from emtom.pddl.dsl import (
    And,
    Believes,
    EpistemicFormula,
    Formula,
    Knows,
    Literal,
    Not,
    Problem,
    parse_goal_string,
)


_ProblemInitExpr = Union[Literal, Knows, Believes]


@dataclass(frozen=True)
class ParsedProblemPDDL:
    """Parsed inline problem PDDL payload."""

    problem_name: str
    domain_name: str
    objects: Dict[str, str]
    init_literals: List[Literal]
    epistemic_init: List[Union[Knows, Believes]]
    goal_formula: Formula
    goal_pddl: str
    owners: Dict[str, str] = None  # literal PDDL string -> owner ID

    def __post_init__(self):
        if self.owners is None:
            object.__setattr__(self, 'owners', {})

    def to_problem(self) -> Problem:
        """Convert to DSL `Problem` dataclass."""
        return Problem(
            name=self.problem_name,
            domain_name=self.domain_name,
            objects=self.objects,
            init=self.init_literals,
            goal=self.goal_formula,
            epistemic_init=self.epistemic_init,
        )


def parse_problem_pddl(problem_pddl: str) -> ParsedProblemPDDL:
    """
    Parse an inline PDDL problem definition from `task.json`.

    Supported sections:
    - `(:domain ...)`
    - `(:objects ...)`
    - `(:init ...)`
    - `(:goal ...)`

    Numeric fluents / assignments are intentionally unsupported in v1.
    """
    raw = _strip_comments(problem_pddl or "").strip()
    if not raw:
        raise ValueError("problem_pddl is empty")

    problem_name = _extract_problem_name(raw)
    domain_name = _extract_domain_name(raw)
    try:
        objects_text = _extract_section(raw, "objects")
    except ValueError:
        objects_text = ""  # :objects is optional
    init_text = _extract_section(raw, "init")
    goal_text = _extract_goal(raw)

    objects = _parse_objects_block(objects_text)
    init_literals, epistemic_init = _parse_init_block(init_text)
    goal_formula = parse_goal_string(goal_text)

    # Parse optional :goal-owners section
    owners = _parse_goal_owners(raw)

    return ParsedProblemPDDL(
        problem_name=problem_name,
        domain_name=domain_name,
        objects=objects,
        init_literals=init_literals,
        epistemic_init=epistemic_init,
        goal_formula=goal_formula,
        goal_pddl=goal_text,
        owners=owners,
    )


def extract_goal_from_problem_pddl(problem_pddl: str) -> str:
    """Extract raw `:goal` expression from an inline problem string."""
    raw = _strip_comments(problem_pddl or "").strip()
    if not raw:
        raise ValueError("problem_pddl is empty")
    return _extract_goal(raw)


def collect_object_ids_from_formula(formula: Formula) -> Set[str]:
    """Collect grounded object IDs referenced in a formula."""
    out: Set[str] = set()

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            for arg in node.args:
                if not arg.startswith("?"):
                    out.add(arg)
            return
        if isinstance(node, EpistemicFormula):
            _walk(node.inner)
            return
        if hasattr(node, "operands"):
            for op in getattr(node, "operands", []) or []:
                _walk(op)
            return
        if isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)

    _walk(formula)
    return out


def _strip_comments(text: str) -> str:
    # PDDL line comments start with ';'
    return re.sub(r";[^\n]*", "", text)


def _extract_problem_name(text: str) -> str:
    m = re.search(r"\(\s*define\s+\(\s*problem\s+([^\s\)]+)\s*\)", text, flags=re.IGNORECASE)
    if not m:
        return "task_problem"
    return m.group(1)


def _extract_domain_name(text: str) -> str:
    m = re.search(r"\(\s*:domain\s+([^\s\)]+)\s*\)", text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("problem_pddl is missing (:domain ...)")
    return m.group(1)


def _extract_section(text: str, section: str) -> str:
    """
    Extract section payload from `(:<section> ...)`.

    Returns the raw content inside the section (excluding wrapper parens).
    """
    lower = text.lower()
    needle = f"(:{section.lower()}"
    idx = lower.find(needle)
    if idx < 0:
        raise ValueError(f"problem_pddl is missing (:{section} ...)")

    start = idx + len(needle)
    end = _find_matching_paren(text, idx)
    return text[start:end].strip()


def _extract_goal(text: str) -> str:
    lower = text.lower()
    needle = "(:goal"
    idx = lower.find(needle)
    if idx < 0:
        raise ValueError("problem_pddl is missing (:goal ...)")

    pos = idx + len(needle)
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "(":
        raise ValueError("(:goal ...) must contain a parenthesized formula")

    end = _find_matching_paren(text, pos)
    return text[pos : end + 1].strip()


def _find_matching_paren(text: str, open_idx: int) -> int:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        raise ValueError("Internal parser error: expected '(' at open index")

    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
            if depth < 0:
                break
    raise ValueError("Unbalanced parentheses in problem_pddl")


def _split_top_level_s_exprs(text: str) -> List[str]:
    exprs: List[str] = []
    depth = 0
    start: Optional[int] = None

    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                exprs.append(text[start : i + 1].strip())
                start = None
            if depth < 0:
                raise ValueError("Unbalanced parentheses while parsing init block")

    if depth != 0:
        raise ValueError("Unbalanced parentheses while parsing init block")

    return [e for e in exprs if e]


def _parse_objects_block(text: str) -> Dict[str, str]:
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return {}

    objects: Dict[str, str] = {}
    pending: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-":
            if not pending:
                raise ValueError("Invalid :objects block: '-' without preceding object names")
            if i + 1 >= len(tokens):
                raise ValueError("Invalid :objects block: missing type after '-'")
            typ = tokens[i + 1]
            for name in pending:
                objects[name] = typ
            pending = []
            i += 2
            continue
        pending.append(tok)
        i += 1

    # Untyped tails default to object.
    for name in pending:
        objects[name] = "object"

    return objects


def _parse_init_expr(expr: str) -> _ProblemInitExpr:
    parsed = parse_goal_string(expr)
    if isinstance(parsed, Literal):
        return parsed
    if isinstance(parsed, (Knows, Believes)):
        return parsed
    if isinstance(parsed, Not) and isinstance(parsed.operand, Literal):
        inner = parsed.operand
        return Literal(predicate=inner.predicate, args=inner.args, negated=not inner.negated)
    raise ValueError(f"Unsupported init expression in problem_pddl: {expr}")


def _parse_init_block(text: str) -> Tuple[List[Literal], List[Union[Knows, Believes]]]:
    literals: List[Literal] = []
    epistemic: List[Union[Knows, Believes]] = []

    for expr in _split_top_level_s_exprs(text):
        parsed = _parse_init_expr(expr)
        if isinstance(parsed, Literal):
            literals.append(parsed)
        else:
            epistemic.append(parsed)

    return literals, epistemic


def _parse_goal_owners(text: str) -> Dict[str, str]:
    """Parse optional (:goal-owners ...) section from problem PDDL.

    Format::

        (:goal-owners
          (team_0 (is_inside trophy_1 cabinet_10))
          (team_1 (is_inside trophy_1 cabinet_20)))

    Returns mapping from PDDL literal string to owner ID.
    """
    lower = text.lower()
    needle = "(:goal-owners"
    idx = lower.find(needle)
    if idx < 0:
        return {}

    start = idx + len(needle)
    end = _find_matching_paren(text, idx)
    body = text[start:end].strip()

    owners: Dict[str, str] = {}
    for entry in _split_top_level_s_exprs(body):
        # Each entry is (owner_id formula)
        # Strip outer parens
        inner = entry.strip()
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1].strip()

        # First token is owner, rest is the PDDL formula
        parts = inner.split(None, 1)
        if len(parts) != 2:
            continue
        owner_id = parts[0]
        formula_str = parts[1].strip()
        # Normalize the formula via parse+serialize for consistent keys.
        # If formula is compound (and A B C), decompose into individual
        # literals so each one maps to the owner separately.
        try:
            formula = parse_goal_string(formula_str)
            if isinstance(formula, And):
                for operand in formula.operands:
                    owners[operand.to_pddl()] = owner_id
            else:
                owners[formula.to_pddl()] = owner_id
        except ValueError:
            # Best-effort: use raw string
            owners[formula_str] = owner_id

    return owners


def strip_goal_owners_pddl(pddl_str: str) -> str:
    """Remove (:goal-owners ...) section from a PDDL string.

    Used before passing to planners that don't understand this extension.
    """
    lower = pddl_str.lower()
    needle = "(:goal-owners"
    idx = lower.find(needle)
    if idx < 0:
        return pddl_str

    end = _find_matching_paren(pddl_str, idx)
    return pddl_str[:idx] + pddl_str[end + 1:]
