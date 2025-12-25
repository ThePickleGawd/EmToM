"""
EMTOM Evaluation System.

Uses PARTNR's SimBasedPredicates for ground-truth simulator state checks.
Extends with EMTOM-specific predicates (is_open, is_closed) not in PARTNR.

Based on: https://arxiv.org/abs/2411.00081
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.sims.collaboration_sim import CollaborationSim
    from habitat_llm.world_model import Graph


@dataclass
class PropositionResult:
    """Result of checking a proposition."""
    is_satisfied: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of task evaluation."""
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


# PARTNR predicates (from SimBasedPredicates)
PARTNR_PREDICATES = {
    "is_on_top", "is_inside", "is_in_room", "is_on_floor", "is_next_to",
    "is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off",
}

# EMTOM-specific predicates (not in PARTNR)
EMTOM_PREDICATES = {"is_open", "is_closed", "is_held_by"}


def is_open(
    sim: "CollaborationSim",
    object_handles: List[str],
    threshold: float = 0.1,
) -> PropositionResult:
    """Check if an articulated object is open using joint positions."""
    from habitat.sims.habitat_simulator.sim_utilities import (
        get_ao_default_link,
        link_is_open,
    )

    aom = sim.get_articulated_object_manager()

    for handle in object_handles:
        ao = aom.get_object_by_handle(handle)
        if ao is None:
            continue

        default_link = get_ao_default_link(ao, compute_if_not_found=True)
        if default_link is None:
            return PropositionResult(True, {"object_handles": handle})

        if link_is_open(ao, default_link, threshold=threshold):
            return PropositionResult(True, {"object_handles": handle})

    return PropositionResult(False, {"object_handles": ""})


def is_closed(
    sim: "CollaborationSim",
    object_handles: List[str],
) -> PropositionResult:
    """Check if an articulated object is closed."""
    result = is_open(sim, object_handles)
    return PropositionResult(not result.is_satisfied, result.info)


def is_held_by(
    sim: "CollaborationSim",
    object_handles: List[str],
    agent_ids: List[str],
) -> PropositionResult:
    """Check if an object is being held by an agent."""
    # Get agent's grasp manager to check what they're holding
    for agent_id in agent_ids:
        try:
            # Try to get the agent's grasp manager
            agent_idx = int(agent_id.split("_")[-1]) if "_" in agent_id else int(agent_id)
            agent = sim.agents_mgr[agent_idx]
            grasp_mgr = agent.grasp_mgr

            # Check if any of the object handles are currently grasped
            if grasp_mgr.is_grasped:
                grasped_obj = grasp_mgr.snap_idx
                rom = sim.get_rigid_object_manager()
                for handle in object_handles:
                    # Try to match by handle or object ID
                    obj = rom.get_object_by_handle(handle)
                    if obj and obj.object_id == grasped_obj:
                        return PropositionResult(True, {"agent": agent_id, "object": handle})
        except Exception:
            continue

    return PropositionResult(False, {"object_handles": object_handles, "agent_ids": agent_ids})


class TaskEvaluator:
    """Evaluates task completion using PARTNR + EMTOM predicates."""

    PREDICATES = PARTNR_PREDICATES | EMTOM_PREDICATES

    def __init__(
        self,
        success_condition: Dict[str, Any],
        sim: "CollaborationSim",
        world_graph: Optional["Graph"] = None,
    ):
        self.success_condition = success_condition
        self.sim = sim
        self.world_graph = world_graph
        self.required_states = success_condition.get("required_states", [])

    def _resolve_handle(self, name: str) -> str:
        """Resolve name to simulator handle via world graph."""
        if not self.world_graph:
            return name
        try:
            return self.world_graph.get_node_from_name(name).sim_handle
        except ValueError:
            return name

    def _get_predicate_fn(self, name: str):
        """Get predicate function by name."""
        if name in EMTOM_PREDICATES:
            return {"is_open": is_open, "is_closed": is_closed, "is_held_by": is_held_by}[name]

        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        return getattr(SimBasedPredicates, name)

    def _check_proposition(self, prop: Dict[str, Any]) -> PropositionResult:
        """Check a single proposition."""
        entity = prop.get("entity")
        property_name = prop.get("property")
        target = prop.get("target")
        value = prop.get("value")

        if property_name not in self.PREDICATES:
            return PropositionResult(False, {"error": f"Unknown predicate: {property_name}"})

        predicate_fn = self._get_predicate_fn(property_name)
        entity_handle = self._resolve_handle(entity) if entity else None
        target_handle = self._resolve_handle(target) if target else None

        # Relational predicates
        if property_name in ("is_on_top", "is_inside"):
            if not target_handle:
                return PropositionResult(False, {"error": f"{property_name} requires 'target'"})
            result = predicate_fn(self.sim, [entity_handle], [target_handle])

        elif property_name == "is_in_room":
            if not target:
                return PropositionResult(False, {"error": "is_in_room requires 'target'"})
            result = predicate_fn(self.sim, [entity_handle], [target])

        elif property_name == "is_next_to":
            if not target_handle:
                return PropositionResult(False, {"error": "is_next_to requires 'target'"})
            result = predicate_fn(self.sim, [entity_handle], [target_handle])

        elif property_name == "is_held_by":
            if not target:
                return PropositionResult(False, {"error": "is_held_by requires 'target' (agent_id)"})
            # target is the agent ID, entity is the object
            result = predicate_fn(self.sim, [entity_handle], [target])

        # Unary predicates
        else:
            result = predicate_fn(self.sim, [entity_handle])

        # Convert PARTNR result to our format if needed
        if not isinstance(result, PropositionResult):
            result = PropositionResult(result.is_satisfied, result.info)

        # Handle explicit False value
        if value is False:
            return PropositionResult(not result.is_satisfied, result.info)

        return result

    def evaluate(self) -> EvaluationResult:
        """Evaluate task completion."""
        if not self.required_states:
            return EvaluationResult(1.0, True, [], {})

        proposition_status = {}
        failure_explanations = []
        satisfied_count = 0

        for i, prop in enumerate(self.required_states):
            prop_id = prop.get("prop_id", f"prop_{i}")
            try:
                result = self._check_proposition(prop)
                proposition_status[prop_id] = result.is_satisfied
                if result.is_satisfied:
                    satisfied_count += 1
                else:
                    failure_explanations.append(self._explain_failure(prop))
            except Exception as e:
                proposition_status[prop_id] = False
                failure_explanations.append(f"Error checking {prop_id}: {e}")

        percent_complete = satisfied_count / len(self.required_states)
        return EvaluationResult(
            percent_complete=percent_complete,
            success=percent_complete == 1.0,
            failure_explanations=failure_explanations,
            proposition_status=proposition_status,
        )

    def _explain_failure(self, prop: Dict[str, Any]) -> str:
        """Generate failure explanation."""
        entity = prop.get("entity")
        prop_name = prop.get("property")
        target = prop.get("target")

        explanations = {
            "is_on_top": f"{entity} is not on top of {target}",
            "is_inside": f"{entity} is not inside {target}",
            "is_in_room": f"{entity} is not in room {target}",
            "is_next_to": f"{entity} is not next to {target}",
            "is_on_floor": f"{entity} is not on the floor",
            "is_open": f"{entity} is not open",
            "is_closed": f"{entity} is not closed",
            "is_held_by": f"{entity} is not held by {target}",
        }
        return explanations.get(prop_name, f"{entity} is not {prop_name.replace('is_', '')}")


def evaluate_task(
    success_condition: Dict[str, Any],
    sim: "CollaborationSim",
    world_graph: Optional["Graph"] = None,
) -> EvaluationResult:
    """Convenience function to evaluate a task."""
    return TaskEvaluator(success_condition, sim, world_graph).evaluate()
