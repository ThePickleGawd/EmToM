"""
EMTOM Evaluation System.

Uses PARTNR's SimBasedPredicates for ground-truth simulator state checks.
Extends with EMTOM-specific predicates (is_open, is_closed) not in PARTNR.

Supports three task categories:
- Cooperative: All agents work toward shared goals (required=True subtasks)
- Competitive: Teams compete for mutually exclusive goals (required="team_X" subtasks)
- Mixed: Shared main goal + agent-specific subgoals (required="agent_X" subtasks)

Based on: https://arxiv.org/abs/2411.00081
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.sims.collaboration_sim import CollaborationSim
    from habitat_llm.world_model import Graph
    from emtom.task_gen.task_generator import GeneratedTask
    from emtom.state.manager import GameStateManager


@dataclass
class PropositionResult:
    """Result of checking a proposition."""
    is_satisfied: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of task evaluation (cooperative tasks)."""
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


@dataclass
class CompetitiveResult:
    """Result of competitive task evaluation."""
    winner: Optional[str]  # "team_0", "team_1", or None (draw)
    team_status: Dict[str, bool]  # team_id -> completed all their subtasks
    team_progress: Dict[str, float]  # team_id -> percent of subtasks completed
    proposition_status: Dict[str, bool] = field(default_factory=dict)
    in_progress: bool = False  # True if episode hasn't terminated yet
    termination_reason: Optional[str] = None  # Why episode ended

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "team_status": self.team_status,
            "team_progress": self.team_progress,
            "proposition_status": self.proposition_status,
            "in_progress": self.in_progress,
            "termination_reason": self.termination_reason,
        }


@dataclass
class MixedResult:
    """Result of mixed task evaluation."""
    main_goal_success: bool  # Did required=True subtasks complete?
    main_goal_progress: float  # Percent of main goal subtasks completed
    agent_subgoal_status: Dict[str, bool]  # agent_id -> completed their subgoal
    proposition_status: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "main_goal_success": self.main_goal_success,
            "main_goal_progress": self.main_goal_progress,
            "agent_subgoal_status": self.agent_subgoal_status,
            "proposition_status": self.proposition_status,
        }


# PARTNR predicates (from SimBasedPredicates)
PARTNR_PREDICATES = {
    "is_on_top", "is_inside", "is_in_room", "is_on_floor", "is_next_to",
    "is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off",
}

# EMTOM-specific predicates (not in PARTNR)
EMTOM_PREDICATES = {"is_open", "is_closed", "is_held_by", "has_at_least", "has_most", "is_stunned"}

# Predicates that require GameStateManager instead of simulator
GAME_STATE_PREDICATES = {"has_at_least", "has_most", "is_stunned"}


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


def has_at_least(
    game_manager: "GameStateManager",
    entity: str,
    item_type: str,
    count: int,
) -> PropositionResult:
    """
    Check if an agent or team has at least N items of a given type.

    Use this for cooperative tasks where agents need to collect a certain
    number of items together.

    Args:
        game_manager: The game state manager
        entity: Agent ID (e.g., "agent_0") or team ID (e.g., "team_0")
        item_type: Base item type (e.g., "item_gold_coin")
        count: Minimum number required

    Returns:
        PropositionResult indicating if condition is met
    """
    is_team = entity.startswith("team_")

    if is_team:
        actual = game_manager.count_team_items_by_type(entity, item_type)
    else:
        actual = game_manager.count_items_by_type(entity, item_type)

    return PropositionResult(
        actual >= count,
        {
            "entity": entity,
            "item_type": item_type,
            "count": actual,
            "required": count,
            "satisfied": actual >= count,
        }
    )


def has_most(
    game_manager: "GameStateManager",
    entity: str,
    item_type: str,
) -> PropositionResult:
    """
    Check if an agent or team has more items of a type than all others.

    Use this for competitive tasks where the winner is whoever collects
    the most items. This predicate should only be evaluated at episode
    termination (handled by CategoryTaskEvaluator).

    Args:
        game_manager: The game state manager
        entity: Agent ID (e.g., "agent_0") or team ID (e.g., "team_0")
        item_type: Base item type (e.g., "item_gold_coin")

    Returns:
        PropositionResult indicating if entity has the most items.
        Returns False on ties (neither side wins).
    """
    is_team = entity.startswith("team_")

    if is_team:
        my_count = game_manager.count_team_items_by_type(entity, item_type)
        others = game_manager.get_all_team_ids()
    else:
        my_count = game_manager.count_items_by_type(entity, item_type)
        others = game_manager.get_all_agent_ids()

    # Compare against all others
    for other in others:
        if other == entity:
            continue

        if is_team:
            other_count = game_manager.count_team_items_by_type(other, item_type)
        else:
            other_count = game_manager.count_items_by_type(other, item_type)

        # Tie or beaten = not the winner
        if other_count >= my_count:
            status = "tied" if other_count == my_count else "lost"
            return PropositionResult(
                False,
                {
                    "entity": entity,
                    "item_type": item_type,
                    "my_count": my_count,
                    "status": status,
                    "beaten_by": other,
                    "their_count": other_count,
                }
            )

    return PropositionResult(
        True,
        {
            "entity": entity,
            "item_type": item_type,
            "count": my_count,
            "status": "won",
        }
    )


def is_stunned(
    game_manager: "GameStateManager",
    agent_id: str,
) -> PropositionResult:
    """
    Check if an agent is currently stunned.

    Args:
        game_manager: The game state manager
        agent_id: Agent ID to check (e.g., "agent_0")

    Returns:
        PropositionResult indicating if agent is stunned
    """
    stunned = game_manager.is_agent_stunned(agent_id)
    turns_remaining = game_manager.get_stun_turns_remaining(agent_id)

    return PropositionResult(
        stunned,
        {
            "agent": agent_id,
            "is_stunned": stunned,
            "turns_remaining": turns_remaining,
        }
    )


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


class CategoryTaskEvaluator:
    """
    Category-aware task evaluator.

    Evaluates tasks based on their category:
    - Cooperative: Check required=True subtasks (evaluated each step)
    - Competitive: Check required="team_X" subtasks, determine winner (evaluated at termination)
    - Mixed: Check required=True for main goal, required="agent_X" for subgoals
    """

    PREDICATES = PARTNR_PREDICATES | EMTOM_PREDICATES

    def __init__(
        self,
        task: "GeneratedTask",
        sim: "CollaborationSim",
        world_graph: Optional["Graph"] = None,
        game_manager: Optional["GameStateManager"] = None,
    ):
        self.task = task
        self.sim = sim
        self.world_graph = world_graph
        self.game_manager = game_manager

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
        """Check a single proposition (success_condition)."""
        entity = prop.get("entity")
        property_name = prop.get("property")
        target = prop.get("target")
        value = prop.get("value")

        if property_name not in self.PREDICATES:
            return PropositionResult(False, {"error": f"Unknown predicate: {property_name}"})

        # Handle game state predicates (require GameStateManager, not simulator)
        if property_name in GAME_STATE_PREDICATES:
            return self._check_game_state_predicate(prop)

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

    def _check_game_state_predicate(self, prop: Dict[str, Any]) -> PropositionResult:
        """Check a game state predicate (requires GameStateManager)."""
        if not self.game_manager:
            return PropositionResult(False, {"error": "GameStateManager required for game state predicates"})

        entity = prop.get("entity")
        property_name = prop.get("property")
        target = prop.get("target")  # item_type for has_most/has_at_least
        value = prop.get("value")  # count for has_at_least

        if property_name == "has_at_least":
            # entity: agent_id or team_id
            # target: item_type
            # value: count required
            if not target:
                return PropositionResult(False, {"error": "has_at_least requires 'target' (item_type)"})
            count = value if isinstance(value, int) else 1
            return has_at_least(self.game_manager, entity, target, count)

        elif property_name == "has_most":
            # entity: agent_id or team_id
            # target: item_type
            if not target:
                return PropositionResult(False, {"error": "has_most requires 'target' (item_type)"})
            return has_most(self.game_manager, entity, target)

        elif property_name == "is_stunned":
            # entity: agent_id
            result = is_stunned(self.game_manager, entity)
            # Handle value=False (checking if NOT stunned)
            if value is False:
                return PropositionResult(not result.is_satisfied, result.info)
            return result

        return PropositionResult(False, {"error": f"Unknown game state predicate: {property_name}"})

    def evaluate(self):
        """Evaluate based on task category."""
        category = self.task.category
        if category == "cooperative":
            return self._evaluate_cooperative()
        elif category == "competitive":
            return self._evaluate_competitive()
        elif category == "mixed":
            return self._evaluate_mixed()
        else:
            # Default to cooperative
            return self._evaluate_cooperative()

    def _evaluate_cooperative(self) -> EvaluationResult:
        """Evaluate cooperative task: all required=True subtasks must be satisfied."""
        required_subtasks = self.task.get_required_subtasks()

        if not required_subtasks:
            return EvaluationResult(1.0, True, [], {})

        proposition_status = {}
        failure_explanations = []
        satisfied_count = 0

        for subtask in required_subtasks:
            if not subtask.has_valid_condition():
                continue

            try:
                result = self._check_proposition(subtask.success_condition)
                proposition_status[subtask.id] = result.is_satisfied
                if result.is_satisfied:
                    satisfied_count += 1
                else:
                    failure_explanations.append(f"{subtask.id}: {subtask.description}")
            except Exception as e:
                proposition_status[subtask.id] = False
                failure_explanations.append(f"Error checking {subtask.id}: {e}")

        total = len([s for s in required_subtasks if s.has_valid_condition()])
        percent_complete = satisfied_count / total if total > 0 else 1.0

        return EvaluationResult(
            percent_complete=percent_complete,
            success=percent_complete == 1.0,
            failure_explanations=failure_explanations,
            proposition_status=proposition_status,
        )

    def _evaluate_competitive(self) -> CompetitiveResult:
        """
        Evaluate competitive task: determine winner based on team subtasks.

        For predicates like has_most, evaluation only happens at episode termination.
        If the episode hasn't terminated yet, returns in_progress=True.
        """
        teams = self.task.get_all_teams()

        if not teams:
            # No team subtasks, treat as draw
            return CompetitiveResult(
                winner=None,
                team_status={},
                team_progress={},
                proposition_status={},
            )

        # Check if episode has terminated (required for has_most predicates)
        is_terminated = False
        termination_reason = None
        if self.game_manager:
            is_terminated = self.game_manager.state.is_terminated
            termination_reason = self.game_manager.state.termination_reason

        # If not terminated, return in_progress result
        if not is_terminated:
            return CompetitiveResult(
                winner=None,
                team_status={team_id: False for team_id in teams},
                team_progress={team_id: 0.0 for team_id in teams},
                proposition_status={},
                in_progress=True,
                termination_reason=None,
            )

        proposition_status = {}
        team_status = {}
        team_progress = {}
        winner = None

        for team_id in teams:
            team_subtasks = self.task.get_team_subtasks(team_id)
            valid_subtasks = [s for s in team_subtasks if s.has_valid_condition()]

            if not valid_subtasks:
                team_status[team_id] = True
                team_progress[team_id] = 1.0
                if winner is None:
                    winner = team_id
                continue

            satisfied_count = 0
            for subtask in valid_subtasks:
                try:
                    result = self._check_proposition(subtask.success_condition)
                    proposition_status[subtask.id] = result.is_satisfied
                    if result.is_satisfied:
                        satisfied_count += 1
                except Exception:
                    proposition_status[subtask.id] = False

            progress = satisfied_count / len(valid_subtasks)
            team_progress[team_id] = progress
            team_status[team_id] = (progress == 1.0)

            # First team to complete all subtasks wins
            if team_status[team_id] and winner is None:
                winner = team_id

        return CompetitiveResult(
            winner=winner,
            team_status=team_status,
            team_progress=team_progress,
            proposition_status=proposition_status,
            in_progress=False,
            termination_reason=termination_reason,
        )

    def _evaluate_mixed(self) -> MixedResult:
        """Evaluate mixed task: main goal (required=True) + agent subgoals (required="agent_X")."""
        # Evaluate main goal (required=True subtasks)
        coop_result = self._evaluate_cooperative()

        # Evaluate agent subgoals
        agent_subgoal_status = {}
        proposition_status = dict(coop_result.proposition_status)

        for subtask in self.task.subtasks:
            owner = subtask.owner
            if owner and owner.startswith("agent_") and subtask.has_valid_condition():
                try:
                    result = self._check_proposition(subtask.success_condition)
                    proposition_status[subtask.id] = result.is_satisfied
                    agent_subgoal_status[owner] = result.is_satisfied
                except Exception:
                    proposition_status[subtask.id] = False
                    agent_subgoal_status[owner] = False

        return MixedResult(
            main_goal_success=coop_result.success,
            main_goal_progress=coop_result.percent_complete,
            agent_subgoal_status=agent_subgoal_status,
            proposition_status=proposition_status,
        )


def evaluate_category_task(
    task: "GeneratedTask",
    sim: "CollaborationSim",
    world_graph: Optional["Graph"] = None,
    game_manager: Optional["GameStateManager"] = None,
):
    """
    Evaluate a task based on its category.

    Args:
        task: The generated task to evaluate
        sim: Habitat simulator for physical state predicates
        world_graph: Optional world graph for handle resolution
        game_manager: GameStateManager for game state predicates (has_most, has_at_least)

    Returns:
    - EvaluationResult for cooperative tasks
    - CompetitiveResult for competitive tasks
    - MixedResult for mixed tasks
    """
    return CategoryTaskEvaluator(task, sim, world_graph, game_manager).evaluate()
