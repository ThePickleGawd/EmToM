"""
PDDL goal checker — replaces DAGProgress for runtime evaluation.

Evaluates PDDL goal formulas against simulator state with latching behavior
(once a conjunct is satisfied, it stays satisfied).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

from emtom.pddl.dsl import Formula, Literal, And, Or, Not, parse_goal_string


class PDDLGoalChecker:
    """
    Evaluates PDDL goal formula against simulator state.

    Replaces DAGProgress with PDDL-native goal checking.
    Conjuncts are latched: once completed, they stay completed.

    Supports ordering constraints via pddl_ordering (replaces depends_on).
    """

    def __init__(
        self,
        goal: Formula,
        ordering: Optional[List[Dict[str, str]]] = None,
        owners: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            goal: Parsed PDDL goal formula
            ordering: List of {"before": "(pred ...)", "after": "(pred ...)"} constraints
            owners: Map from goal literal string to owner (e.g., "team_0", "agent_0")
                   Literals not in this map default to required (cooperative).
        """
        self.goal = goal
        self.conjuncts = goal.flatten()
        self.completed: Set[int] = set()  # indices into self.conjuncts

        # Build ordering constraints: index -> set of prerequisite indices
        self._prerequisites: Dict[int, Set[int]] = {}
        self._build_ordering(ordering or [])

        # Owner map: index -> owner string
        self._owners: Dict[int, str] = {}
        if owners:
            conjunct_strs = [c.to_pddl() for c in self.conjuncts]
            for literal_str, owner in owners.items():
                for idx, cs in enumerate(conjunct_strs):
                    if cs == literal_str:
                        self._owners[idx] = owner

    def _build_ordering(self, ordering: List[Dict[str, str]]) -> None:
        """Build prerequisite map from ordering constraints."""
        conjunct_strs = [c.to_pddl() for c in self.conjuncts]

        for constraint in ordering:
            before_str = constraint.get("before", "")
            after_str = constraint.get("after", "")

            before_idx = None
            after_idx = None
            for idx, cs in enumerate(conjunct_strs):
                if cs == before_str:
                    before_idx = idx
                if cs == after_str:
                    after_idx = idx

            if before_idx is not None and after_idx is not None:
                self._prerequisites.setdefault(after_idx, set()).add(before_idx)

    def update(self, check_predicate: Callable) -> Dict[str, Any]:
        """
        Check each goal conjunct, latch completed ones.

        Args:
            check_predicate: Function(predicate_name, args_tuple) -> bool

        Returns:
            Dict with percent_complete, all_complete, newly_completed
        """
        newly_completed = []

        for idx, conjunct in enumerate(self.conjuncts):
            if idx in self.completed:
                continue

            # Check ordering prerequisites
            prereqs = self._prerequisites.get(idx, set())
            if not prereqs.issubset(self.completed):
                continue

            if conjunct.evaluate(check_predicate):
                self.completed.add(idx)
                newly_completed.append(conjunct.to_pddl())

        total = len(self.conjuncts) if self.conjuncts else 1
        return {
            "completed": [self.conjuncts[i].to_pddl() for i in self.completed],
            "newly_completed": newly_completed,
            "percent_complete": len(self.completed) / total,
            "all_complete": len(self.completed) == len(self.conjuncts),
        }

    def get_required_conjuncts(self) -> List[Literal]:
        """Get conjuncts that are required for task success (no owner or required=True)."""
        return [
            c for idx, c in enumerate(self.conjuncts)
            if idx not in self._owners
        ]

    def get_team_conjuncts(self, team_id: str) -> List[Literal]:
        """Get conjuncts owned by a specific team."""
        return [
            c for idx, c in enumerate(self.conjuncts)
            if self._owners.get(idx) == team_id
        ]

    def get_agent_conjuncts(self, agent_id: str) -> List[Literal]:
        """Get conjuncts owned by a specific agent."""
        return [
            c for idx, c in enumerate(self.conjuncts)
            if self._owners.get(idx) == agent_id
        ]

    def get_all_teams(self) -> List[str]:
        """Get all team IDs that own conjuncts."""
        teams = set()
        for owner in self._owners.values():
            if owner.startswith("team_"):
                teams.add(owner)
        return sorted(teams)

    def is_conjunct_completed(self, idx: int) -> bool:
        """Check if a specific conjunct index is latched as completed."""
        return idx in self.completed

    def to_propositions(self) -> List[Dict[str, Any]]:
        """
        Convert PDDL goal conjuncts to evaluation.py proposition format.

        Returns list of {"entity": ..., "property": ..., "target": ..., "required": ...}
        """
        props = []
        for idx, conjunct in enumerate(self.conjuncts):
            if isinstance(conjunct, Literal):
                prop = conjunct.to_proposition()
                # Add ownership/required info
                owner = self._owners.get(idx)
                if owner:
                    prop["required"] = owner
                else:
                    prop["required"] = True
                props.append(prop)
        return props

    def reset(self) -> None:
        """Reset all progress."""
        self.completed.clear()

    @classmethod
    def from_task_data(cls, task_data: Dict[str, Any]) -> Optional["PDDLGoalChecker"]:
        """
        Create a PDDLGoalChecker from raw task JSON data.

        Returns None if task doesn't have pddl_goal.
        """
        pddl_goal_str = task_data.get("pddl_goal")
        if not pddl_goal_str:
            return None

        goal = parse_goal_string(pddl_goal_str)
        ordering = task_data.get("pddl_ordering", [])
        owners = task_data.get("pddl_owners", {})

        return cls(goal=goal, ordering=ordering, owners=owners)
