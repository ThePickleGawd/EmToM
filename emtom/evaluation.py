"""
EMTOM Evaluation System.

PARTNR-inspired evaluation with:
- Temporal constraint verification (BEFORE/AFTER)
- Percent complete metric (0-1 ratio)
- Failure explanations (human-readable)

Based on: https://arxiv.org/abs/2411.00081 (Section A.5)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from emtom.state.game_state import EMTOMGameState


class ConstraintType(Enum):
    """Types of temporal constraints."""
    BEFORE = "before"  # A must happen before B
    AFTER = "after"    # A must happen after B
    DURING = "during"  # A must happen while B is true


@dataclass
class Milestone:
    """
    An intermediate checkpoint that SHOULD be reached during task execution.

    Milestones help track partial progress and debug agent behavior.
    Unlike required_states, milestones don't affect success - they're informational.

    Example:
        {"id": "found_key", "entity": "agent_0", "property": "has_item", "value": "key_1"}
    """
    milestone_id: str
    entity: str
    property: str
    value: Any
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Milestone":
        return cls(
            milestone_id=data.get("id", f"{data['entity']}_{data['property']}"),
            entity=data["entity"],
            property=data["property"],
            value=data["value"],
            description=data.get("description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.milestone_id,
            "entity": self.entity,
            "property": self.property,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class Minefield:
    """
    A failure condition that should NEVER be triggered during task execution.

    Minefields catch bad agent behavior (breaking things, wrong actions).
    If triggered, they're recorded but don't necessarily fail the task.

    Example:
        {"id": "broke_vase", "entity": "vase_1", "property": "is_broken", "value": True}
    """
    minefield_id: str
    entity: str
    property: str
    value: Any
    description: Optional[str] = None
    fails_task: bool = False  # If True, triggering this fails the entire task

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Minefield":
        return cls(
            minefield_id=data.get("id", f"{data['entity']}_{data['property']}"),
            entity=data["entity"],
            property=data["property"],
            value=data["value"],
            description=data.get("description"),
            fails_task=data.get("fails_task", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.minefield_id,
            "entity": self.entity,
            "property": self.property,
            "value": self.value,
            "description": self.description,
            "fails_task": self.fails_task,
        }


@dataclass
class TemporalConstraint:
    """
    A temporal constraint between two propositions.

    Example:
        {"type": "before", "first": "unlock_chest", "then": "retrieve_item"}
        Means: unlock_chest must be satisfied before retrieve_item
    """
    constraint_type: ConstraintType
    first: str  # Proposition ID that must happen first
    then: str   # Proposition ID that must happen second

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalConstraint":
        return cls(
            constraint_type=ConstraintType(data["type"]),
            first=data["first"],
            then=data["then"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.constraint_type.value,
            "first": self.first,
            "then": self.then,
        }


@dataclass
class Proposition:
    """
    A proposition about world state.

    Can check:
    - Object properties: {"entity": "chest_1", "property": "is_open", "value": True}
    - Agent inventory: {"entity": "agent_0", "property": "has_item", "value": "key_1"}
    - Object location: {"entity": "cup_1", "property": "location", "value": "table_1"}
    """
    prop_id: str
    entity: str
    property: str
    value: Any
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prop_id: str = None) -> "Proposition":
        return cls(
            prop_id=prop_id or data.get("prop_id", f"{data['entity']}_{data['property']}"),
            entity=data["entity"],
            property=data["property"],
            value=data["value"],
            description=data.get("description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prop_id": self.prop_id,
            "entity": self.entity,
            "property": self.property,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class EvaluationResult:
    """
    Result of task evaluation.

    Attributes:
        percent_complete: Ratio of satisfied propositions (0.0 to 1.0)
        success: True if all propositions satisfied and constraints met
        failure_explanations: Human-readable list of why task failed
        proposition_status: Dict mapping prop_id -> satisfied (bool)
        constraint_status: Dict mapping constraint description -> satisfied (bool)
        satisfied_at_step: Dict mapping prop_id -> step when first satisfied
        milestones_reached: Dict mapping milestone_id -> step when reached (None if not reached)
        minefields_triggered: Dict mapping minefield_id -> step when triggered (None if not triggered)
    """
    percent_complete: float
    success: bool
    failure_explanations: List[str]
    proposition_status: Dict[str, bool] = field(default_factory=dict)
    constraint_status: Dict[str, bool] = field(default_factory=dict)
    satisfied_at_step: Dict[str, Optional[int]] = field(default_factory=dict)
    milestones_reached: Dict[str, Optional[int]] = field(default_factory=dict)
    minefields_triggered: Dict[str, Optional[int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "percent_complete": self.percent_complete,
            "success": self.success,
            "failure_explanations": self.failure_explanations,
            "proposition_status": self.proposition_status,
            "constraint_status": self.constraint_status,
            "satisfied_at_step": self.satisfied_at_step,
            "milestones_reached": self.milestones_reached,
            "minefields_triggered": self.minefields_triggered,
        }

    @property
    def milestones_summary(self) -> str:
        """Human-readable summary of milestone progress."""
        reached = sum(1 for v in self.milestones_reached.values() if v is not None)
        total = len(self.milestones_reached)
        if total == 0:
            return "No milestones defined"
        return f"{reached}/{total} milestones reached"

    @property
    def minefields_summary(self) -> str:
        """Human-readable summary of minefield status."""
        triggered = sum(1 for v in self.minefields_triggered.values() if v is not None)
        total = len(self.minefields_triggered)
        if total == 0:
            return "No minefields defined"
        if triggered == 0:
            return f"0/{total} minefields triggered (clean run)"
        return f"{triggered}/{total} minefields triggered"


class TaskEvaluator:
    """
    Evaluates task completion with temporal constraints.

    Usage:
        evaluator = TaskEvaluator(success_condition)
        result = evaluator.evaluate(state, state_history)
    """

    def __init__(self, success_condition: Dict[str, Any]):
        """
        Initialize evaluator from success_condition dict.

        Expected format:
        {
            "description": "What success looks like",
            "required_states": [
                {"entity": "chest_1", "property": "is_open", "value": True},
                {"entity": "agent_0", "property": "has_item", "value": "key_1"}
            ],
            "temporal_constraints": [
                {"type": "before", "first": "get_key", "then": "unlock_chest"}
            ],
            "time_limit": 100  # optional
        }
        """
        self.description = success_condition.get("description", "Complete the task")
        self.time_limit = success_condition.get("time_limit")

        # Parse propositions from required_states
        self.propositions: List[Proposition] = []
        for i, req in enumerate(success_condition.get("required_states", [])):
            prop_id = req.get("prop_id", f"prop_{i}")
            self.propositions.append(Proposition.from_dict(req, prop_id))

        # Parse temporal constraints
        self.constraints: List[TemporalConstraint] = []
        for constraint_data in success_condition.get("temporal_constraints", []):
            self.constraints.append(TemporalConstraint.from_dict(constraint_data))

        # Parse milestones (intermediate checkpoints)
        self.milestones: List[Milestone] = []
        for milestone_data in success_condition.get("milestones", []):
            self.milestones.append(Milestone.from_dict(milestone_data))

        # Parse minefields (failure conditions)
        self.minefields: List[Minefield] = []
        for minefield_data in success_condition.get("minefields", []):
            self.minefields.append(Minefield.from_dict(minefield_data))

    def check_proposition(
        self,
        prop: Proposition,
        state: EMTOMGameState,
        agent_inventory: Dict[str, List[str]] = None,
    ) -> bool:
        """Check if a single proposition is satisfied in the given state."""
        entity = prop.entity
        property_name = prop.property
        expected_value = prop.value

        # Handle inventory checks
        if property_name == "has_item":
            inventory = agent_inventory or state.agent_inventory
            agent_items = inventory.get(entity, [])
            return expected_value in agent_items

        # Handle location checks (object is on/in something)
        if property_name == "location":
            # Check world_objects for spawned items
            for loc, obj_info in state.world_objects.items():
                if obj_info.get("object_id") == entity:
                    return loc == expected_value
            # Check object_properties for location
            obj_loc = state.object_properties.get(entity, {}).get("location")
            if obj_loc:
                return obj_loc == expected_value
            return False

        # Handle is_unlocked (check unlocked_targets set)
        if property_name == "is_unlocked":
            is_unlocked = entity in state.unlocked_targets
            return is_unlocked == expected_value

        # Handle is_locked (inverse of unlocked)
        if property_name == "is_locked":
            is_locked = entity not in state.unlocked_targets
            # Also check object_properties
            prop_locked = state.object_properties.get(entity, {}).get("is_locked", False)
            actual_locked = is_locked or prop_locked
            return actual_locked == expected_value

        # Check Habitat object states (is_open, is_on, etc.)
        habitat_value = state.object_states.get(entity, {}).get(property_name)
        if habitat_value is not None:
            return habitat_value == expected_value

        # Check custom object properties
        custom_value = state.object_properties.get(entity, {}).get(property_name)
        if custom_value is not None:
            return custom_value == expected_value

        # Property not found - consider it not satisfied
        return False

    def find_satisfaction_step(
        self,
        prop: Proposition,
        state_history: List[EMTOMGameState],
    ) -> Optional[int]:
        """Find the step at which a proposition was first satisfied."""
        for i, state in enumerate(state_history):
            if self.check_proposition(prop, state):
                return i
        return None

    def check_milestone(
        self,
        milestone: Milestone,
        state: EMTOMGameState,
    ) -> bool:
        """Check if a milestone is reached in the given state."""
        # Reuse proposition checking logic
        prop = Proposition(
            prop_id=milestone.milestone_id,
            entity=milestone.entity,
            property=milestone.property,
            value=milestone.value,
        )
        return self.check_proposition(prop, state)

    def find_milestone_step(
        self,
        milestone: Milestone,
        state_history: List[EMTOMGameState],
    ) -> Optional[int]:
        """Find the step at which a milestone was first reached."""
        for i, state in enumerate(state_history):
            if self.check_milestone(milestone, state):
                return i
        return None

    def check_minefield(
        self,
        minefield: Minefield,
        state: EMTOMGameState,
    ) -> bool:
        """Check if a minefield is triggered in the given state."""
        # Reuse proposition checking logic
        prop = Proposition(
            prop_id=minefield.minefield_id,
            entity=minefield.entity,
            property=minefield.property,
            value=minefield.value,
        )
        return self.check_proposition(prop, state)

    def find_minefield_step(
        self,
        minefield: Minefield,
        state_history: List[EMTOMGameState],
    ) -> Optional[int]:
        """Find the step at which a minefield was first triggered."""
        for i, state in enumerate(state_history):
            if self.check_minefield(minefield, state):
                return i
        return None

    def evaluate(
        self,
        current_state: EMTOMGameState,
        state_history: Optional[List[EMTOMGameState]] = None,
    ) -> EvaluationResult:
        """
        Evaluate task completion.

        Args:
            current_state: Current game state
            state_history: List of all states during execution (for temporal checks)

        Returns:
            EvaluationResult with percent_complete, success, and failure_explanations
        """
        if state_history is None:
            state_history = [current_state]

        # Track which propositions are satisfied
        prop_status: Dict[str, bool] = {}
        satisfied_at: Dict[str, Optional[int]] = {}
        failure_explanations: List[str] = []

        # Check each proposition
        for prop in self.propositions:
            satisfied = self.check_proposition(prop, current_state)
            prop_status[prop.prop_id] = satisfied

            # Find when it was first satisfied (for temporal checks)
            if satisfied:
                step = self.find_satisfaction_step(prop, state_history)
                satisfied_at[prop.prop_id] = step
            else:
                satisfied_at[prop.prop_id] = None
                # Generate failure explanation
                explanation = self._explain_proposition_failure(prop, current_state)
                failure_explanations.append(explanation)

        # Calculate percent complete
        if self.propositions:
            satisfied_count = sum(1 for v in prop_status.values() if v)
            percent_complete = satisfied_count / len(self.propositions)
        else:
            percent_complete = 1.0  # No propositions = success

        # Check temporal constraints
        constraint_status: Dict[str, bool] = {}
        all_constraints_met = True

        for constraint in self.constraints:
            constraint_key = f"{constraint.first} {constraint.constraint_type.value} {constraint.then}"

            # Get satisfaction steps
            first_step = satisfied_at.get(constraint.first)
            then_step = satisfied_at.get(constraint.then)

            if constraint.constraint_type == ConstraintType.BEFORE:
                # first must be satisfied before then
                if first_step is None:
                    # First not satisfied, can't verify
                    constraint_met = False
                    failure_explanations.append(
                        f"Temporal constraint violated: '{constraint.first}' was never satisfied "
                        f"(required before '{constraint.then}')"
                    )
                elif then_step is None:
                    # Then not satisfied yet, constraint is vacuously true for now
                    constraint_met = True
                else:
                    constraint_met = first_step <= then_step
                    if not constraint_met:
                        failure_explanations.append(
                            f"Temporal constraint violated: '{constraint.then}' was satisfied at step {then_step} "
                            f"but '{constraint.first}' was satisfied later at step {first_step}"
                        )

            elif constraint.constraint_type == ConstraintType.AFTER:
                # first must be satisfied after then
                if then_step is None:
                    constraint_met = False
                    failure_explanations.append(
                        f"Temporal constraint violated: '{constraint.then}' was never satisfied "
                        f"(required before '{constraint.first}')"
                    )
                elif first_step is None:
                    constraint_met = True  # Vacuously true
                else:
                    constraint_met = first_step >= then_step
                    if not constraint_met:
                        failure_explanations.append(
                            f"Temporal constraint violated: '{constraint.first}' was satisfied at step {first_step} "
                            f"but should have been after '{constraint.then}' (step {then_step})"
                        )
            else:
                # DURING - more complex, skip for now
                constraint_met = True

            constraint_status[constraint_key] = constraint_met
            if not constraint_met:
                all_constraints_met = False

        # Check time limit
        if self.time_limit and current_state.current_step > self.time_limit:
            failure_explanations.append(
                f"Time limit exceeded: {current_state.current_step} steps > {self.time_limit} limit"
            )
            all_constraints_met = False

        # Check milestones (informational, doesn't affect success)
        milestones_reached: Dict[str, Optional[int]] = {}
        for milestone in self.milestones:
            step = self.find_milestone_step(milestone, state_history)
            milestones_reached[milestone.milestone_id] = step

        # Check minefields (failure conditions)
        minefields_triggered: Dict[str, Optional[int]] = {}
        minefield_failed = False
        for minefield in self.minefields:
            step = self.find_minefield_step(minefield, state_history)
            minefields_triggered[minefield.minefield_id] = step
            if step is not None:
                desc = minefield.description or f"{minefield.entity}.{minefield.property}={minefield.value}"
                failure_explanations.append(f"Minefield triggered at step {step}: {desc}")
                if minefield.fails_task:
                    minefield_failed = True

        # Success requires all propositions satisfied AND all constraints met AND no fatal minefields
        success = (percent_complete == 1.0) and all_constraints_met and not minefield_failed

        return EvaluationResult(
            percent_complete=percent_complete,
            success=success,
            failure_explanations=failure_explanations,
            proposition_status=prop_status,
            constraint_status=constraint_status,
            satisfied_at_step=satisfied_at,
            milestones_reached=milestones_reached,
            minefields_triggered=minefields_triggered,
        )

    def _explain_proposition_failure(
        self,
        prop: Proposition,
        state: EMTOMGameState,
    ) -> str:
        """Generate human-readable explanation for why a proposition failed."""
        entity = prop.entity
        property_name = prop.property
        expected = prop.value

        # Get actual value
        if property_name == "has_item":
            actual_items = state.agent_inventory.get(entity, [])
            if actual_items:
                return f"{entity} has items {actual_items} but not '{expected}'"
            else:
                return f"{entity} has no items (expected '{expected}')"

        if property_name == "location":
            # Find actual location
            for loc, obj_info in state.world_objects.items():
                if obj_info.get("object_id") == entity:
                    return f"{entity} is at '{loc}' instead of '{expected}'"
            return f"{entity} location unknown (expected '{expected}')"

        if property_name == "is_unlocked":
            is_unlocked = entity in state.unlocked_targets
            return f"{entity} is {'unlocked' if is_unlocked else 'locked'} (expected {'unlocked' if expected else 'locked'})"

        if property_name == "is_locked":
            is_locked = entity not in state.unlocked_targets
            return f"{entity} is {'locked' if is_locked else 'unlocked'} (expected {'locked' if expected else 'unlocked'})"

        # Check Habitat states
        habitat_value = state.object_states.get(entity, {}).get(property_name)
        if habitat_value is not None:
            return f"{entity}.{property_name} is {habitat_value} (expected {expected})"

        # Check custom properties
        custom_value = state.object_properties.get(entity, {}).get(property_name)
        if custom_value is not None:
            return f"{entity}.{property_name} is {custom_value} (expected {expected})"

        return f"{entity}.{property_name} not found (expected {expected})"


def evaluate_task(
    success_condition: Dict[str, Any],
    current_state: EMTOMGameState,
    state_history: Optional[List[EMTOMGameState]] = None,
) -> EvaluationResult:
    """
    Convenience function to evaluate a task.

    Args:
        success_condition: Task's success_condition dict
        current_state: Current game state
        state_history: Optional list of all states during execution

    Returns:
        EvaluationResult
    """
    evaluator = TaskEvaluator(success_condition)
    return evaluator.evaluate(current_state, state_history)
