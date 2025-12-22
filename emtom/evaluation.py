"""
EMTOM Evaluation System.

Uses PARTNR's SimBasedPredicates for ground-truth simulator state checks.
Extends with EMTOM-specific predicates for mechanics verification.

Based on: https://arxiv.org/abs/2411.00081
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from habitat_llm.sims.collaboration_sim import CollaborationSim
    from habitat_llm.world_model import Graph


class ConstraintType(Enum):
    """Types of temporal constraints."""
    BEFORE = "before"
    AFTER = "after"


@dataclass
class PropositionResult:
    """Result of checking a proposition."""
    is_satisfied: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Result of task evaluation.

    Attributes:
        percent_complete: Ratio of satisfied propositions (0.0 to 1.0)
        success: True if all propositions satisfied and constraints met
        failure_explanations: Human-readable list of why task failed
        proposition_status: Dict mapping prop_id -> satisfied (bool)
    """
    percent_complete: float
    success: bool
    failure_explanations: List[str]
    proposition_status: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "percent_complete": self.percent_complete,
            "success": self.success,
            "failure_explanations": self.failure_explanations,
            "proposition_status": self.proposition_status,
        }


class EMTOMPredicates:
    """
    EMTOM predicates that extend PARTNR's SimBasedPredicates.

    Includes all PARTNR predicates plus EMTOM-specific ones like is_open.
    """

    @classmethod
    def is_on_top(
        cls,
        sim: "CollaborationSim",
        object_handles: List[str],
        receptacle_handles: List[str],
    ) -> PropositionResult:
        """Check if object is on top of receptacle. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_on_top(sim, object_handles, receptacle_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_inside(
        cls,
        sim: "CollaborationSim",
        object_handles: List[str],
        receptacle_handles: List[str],
    ) -> PropositionResult:
        """Check if object is inside receptacle. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_inside(sim, object_handles, receptacle_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_in_room(
        cls,
        sim: "CollaborationSim",
        object_handles: List[str],
        room_ids: List[str],
    ) -> PropositionResult:
        """Check if object is in room. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_in_room(sim, object_handles, room_ids)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_on_floor(
        cls,
        sim: "CollaborationSim",
        object_handles: List[str],
    ) -> PropositionResult:
        """Check if object is on floor. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_on_floor(sim, object_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_next_to(
        cls,
        sim: "CollaborationSim",
        entity_handles_a: List[str],
        entity_handles_b: List[str],
        l2_threshold: float = 0.5,
    ) -> PropositionResult:
        """Check if entities are next to each other. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_next_to(
            sim, entity_handles_a, entity_handles_b, l2_threshold=l2_threshold
        )
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_clean(cls, sim: "CollaborationSim", object_handles: List[str]) -> PropositionResult:
        """Check if object is clean. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_clean(sim, object_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_dirty(cls, sim: "CollaborationSim", object_handles: List[str]) -> PropositionResult:
        """Check if object is dirty. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_dirty(sim, object_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_filled(cls, sim: "CollaborationSim", object_handles: List[str]) -> PropositionResult:
        """Check if object is filled. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_filled(sim, object_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_empty(cls, sim: "CollaborationSim", object_handles: List[str]) -> PropositionResult:
        """Check if object is empty. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_empty(sim, object_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_powered_on(cls, sim: "CollaborationSim", object_handles: List[str]) -> PropositionResult:
        """Check if object is powered on. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_powered_on(sim, object_handles)
        return PropositionResult(result.is_satisfied, result.info)

    @classmethod
    def is_powered_off(cls, sim: "CollaborationSim", object_handles: List[str]) -> PropositionResult:
        """Check if object is powered off. Delegates to PARTNR."""
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        result = SimBasedPredicates.is_powered_off(sim, object_handles)
        return PropositionResult(result.is_satisfied, result.info)

    # ========== EMTOM-specific predicates ==========

    @classmethod
    def is_open(
        cls,
        sim: "CollaborationSim",
        object_handles: List[str],
    ) -> PropositionResult:
        """
        Check if an articulated object (drawer, cabinet, etc.) is open.

        This is EMTOM-specific - PARTNR doesn't have this predicate.
        Queries the object state machine for is_open state.
        """
        object_states_dict = sim.object_state_machine.get_snapshot_dict(sim)

        for handle in object_handles:
            is_open_states = object_states_dict.get("is_open", {})
            if handle in is_open_states and is_open_states[handle]:
                return PropositionResult(True, {"object_handles": handle})

        return PropositionResult(False, {"object_handles": ""})

    @classmethod
    def is_closed(
        cls,
        sim: "CollaborationSim",
        object_handles: List[str],
    ) -> PropositionResult:
        """Check if an articulated object is closed (not open)."""
        result = cls.is_open(sim, object_handles)
        # Invert the result
        return PropositionResult(not result.is_satisfied, result.info)


class TaskEvaluator:
    """
    Evaluates task completion using PARTNR predicates.

    Usage:
        evaluator = TaskEvaluator(success_condition, sim, world_graph)
        result = evaluator.evaluate()
    """

    # Supported predicates - PARTNR predicates + EMTOM extensions
    # Use these names directly in task success_condition
    PREDICATES = {
        # PARTNR spatial predicates
        "is_on_top",
        "is_inside",
        "is_in_room",
        "is_on_floor",
        "is_next_to",
        # PARTNR object state predicates
        "is_clean",
        "is_dirty",
        "is_filled",
        "is_empty",
        "is_powered_on",
        "is_powered_off",
        # EMTOM extensions
        "is_open",
        "is_closed",
    }

    def __init__(
        self,
        success_condition: Dict[str, Any],
        sim: "CollaborationSim",
        world_graph: Optional["Graph"] = None,
    ):
        """
        Initialize evaluator.

        Args:
            success_condition: Task's success_condition dict with required_states
            sim: Habitat CollaborationSim instance for predicate checks
            world_graph: PARTNR world graph for name-to-handle resolution
        """
        self.success_condition = success_condition
        self.sim = sim
        self.world_graph = world_graph
        self.description = success_condition.get("description", "Complete the task")

        # Parse required states
        self.required_states = success_condition.get("required_states", [])

        # Parse temporal constraints
        self.temporal_constraints = success_condition.get("temporal_constraints", [])

    def _resolve_handle(self, name: str) -> str:
        """
        Resolve a simple name to a full simulator handle using PARTNR's world graph.

        Args:
            name: Simple name like "kettle_3" or "table_59"

        Returns:
            Full sim_handle from world graph, or original name if not found
        """
        if not self.world_graph:
            # No world graph available, return name as-is
            return name

        try:
            entity = self.world_graph.get_node_from_name(name)
            return entity.sim_handle
        except ValueError:
            # Entity not found in world graph
            return name

    def _check_proposition(self, prop: Dict[str, Any]) -> PropositionResult:
        """
        Check a single proposition against simulator state.

        Args:
            prop: Dict with entity, property, target (for relational) or value (for boolean)
                  Examples:
                    {"entity": "kettle_3", "property": "is_on_top", "target": "table_59"}
                    {"entity": "drawer_1", "property": "is_open", "value": True}

        Returns:
            PropositionResult with is_satisfied and info
        """
        entity = prop.get("entity")
        property_name = prop.get("property")
        target = prop.get("target")
        value = prop.get("value")

        if property_name not in self.PREDICATES:
            return PropositionResult(
                False,
                {"error": f"Unknown predicate: {property_name}. Supported: {self.PREDICATES}"}
            )

        predicate_fn = getattr(EMTOMPredicates, property_name)

        # Resolve simple names to full simulator handles
        entity_handle = self._resolve_handle(entity) if entity else None
        target_handle = self._resolve_handle(target) if target else None

        # Relational predicates (require target)
        if property_name in ("is_on_top", "is_inside"):
            if not target_handle:
                return PropositionResult(False, {"error": f"{property_name} requires 'target'"})
            return predicate_fn(self.sim, [entity_handle], [target_handle])

        elif property_name == "is_in_room":
            if not target:
                return PropositionResult(False, {"error": "is_in_room requires 'target' (room_id)"})
            # Room IDs are not object handles, use as-is
            return predicate_fn(self.sim, [entity_handle], [target])

        elif property_name == "is_next_to":
            if not target_handle:
                return PropositionResult(False, {"error": "is_next_to requires 'target'"})
            return predicate_fn(self.sim, [entity_handle], [target_handle])

        # Unary predicates (entity only)
        elif property_name in ("is_on_floor", "is_open", "is_closed",
                               "is_clean", "is_dirty", "is_filled",
                               "is_empty", "is_powered_on", "is_powered_off"):
            result = predicate_fn(self.sim, [entity_handle])
            # If value is explicitly False, invert the check
            if value is False:
                return PropositionResult(not result.is_satisfied, result.info)
            return result

        return PropositionResult(False, {"error": f"Unhandled predicate: {property_name}"})

    def evaluate(self) -> EvaluationResult:
        """
        Evaluate task completion.

        Returns:
            EvaluationResult with percent_complete, success, and failure_explanations
        """
        if not self.required_states:
            # No requirements = success
            return EvaluationResult(
                percent_complete=1.0,
                success=True,
                failure_explanations=[],
                proposition_status={},
            )

        proposition_status: Dict[str, bool] = {}
        failure_explanations: List[str] = []
        satisfied_count = 0

        for i, prop in enumerate(self.required_states):
            prop_id = prop.get("prop_id", f"prop_{i}")

            try:
                result = self._check_proposition(prop)
                proposition_status[prop_id] = result.is_satisfied

                if result.is_satisfied:
                    satisfied_count += 1
                else:
                    explanation = self._explain_failure(prop, result)
                    failure_explanations.append(explanation)
            except Exception as e:
                proposition_status[prop_id] = False
                failure_explanations.append(f"Error checking {prop_id}: {str(e)}")

        # Calculate percent complete
        percent_complete = satisfied_count / len(self.required_states)

        # Success requires all propositions satisfied
        success = percent_complete == 1.0

        return EvaluationResult(
            percent_complete=percent_complete,
            success=success,
            failure_explanations=failure_explanations,
            proposition_status=proposition_status,
        )

    def _explain_failure(self, prop: Dict[str, Any], result: PropositionResult) -> str:
        """Generate human-readable explanation for proposition failure."""
        entity = prop.get("entity")
        property_name = prop.get("property")
        target = prop.get("target")

        if property_name == "is_on_top":
            return f"{entity} is not on top of {target}"
        elif property_name == "is_inside":
            return f"{entity} is not inside {target}"
        elif property_name == "is_in_room":
            return f"{entity} is not in room {target}"
        elif property_name == "is_next_to":
            return f"{entity} is not next to {target}"
        elif property_name == "is_on_floor":
            return f"{entity} is not on the floor"
        elif property_name == "is_open":
            return f"{entity} is not open"
        elif property_name == "is_closed":
            return f"{entity} is not closed"
        elif property_name in ("is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off"):
            return f"{entity} is not {property_name.replace('is_', '')}"
        else:
            return f"{entity}.{property_name} failed"


def evaluate_task(
    success_condition: Dict[str, Any],
    sim: "CollaborationSim",
    world_graph: Optional["Graph"] = None,
) -> EvaluationResult:
    """
    Convenience function to evaluate a task.

    Args:
        success_condition: Task's success_condition dict
        sim: Habitat CollaborationSim instance
        world_graph: PARTNR world graph for name resolution (recommended)

    Returns:
        EvaluationResult
    """
    evaluator = TaskEvaluator(success_condition, sim, world_graph)
    return evaluator.evaluate()
