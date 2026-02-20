"""Tests for BeliefStateTracker and integrated epistemic goal evaluation."""

import pytest
from unittest.mock import MagicMock

from emtom.pddl.dsl import Literal, And, Knows, Believes, parse_goal_string
from emtom.pddl.belief_tracker import BeliefStateTracker
from emtom.pddl.goal_checker import PDDLGoalChecker
from emtom.pddl.epistemic import ObservabilityModel


class TestBeliefStateTracker:
    def _make_tracker(self, object_rooms=None, num_agents=2):
        return BeliefStateTracker.from_scene_and_observability(
            object_rooms=object_rooms or {
                "cabinet_27": "kitchen_1",
                "table_13": "bedroom_1",
                "bottle_4": "kitchen_1",
            },
            num_agents=num_agents,
        )

    def test_record_observation(self):
        tracker = self._make_tracker()
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))
        assert tracker.agent_knows("agent_0", "is_open", ("cabinet_27",))
        assert not tracker.agent_knows("agent_1", "is_open", ("cabinet_27",))

    def test_room_entry_grants_beliefs(self):
        tracker = self._make_tracker()

        def check_fn(pred, args):
            # Simulate: cabinet_27 is open, bottle_4 is on floor
            if pred == "is_open" and args == ("cabinet_27",):
                return True
            if pred == "is_on_floor" and args == ("bottle_4",):
                return True
            return False

        tracker.record_room_entry("agent_0", "kitchen_1", check_fn)
        assert tracker.agent_knows("agent_0", "is_open", ("cabinet_27",))
        assert tracker.agent_knows("agent_0", "is_on_floor", ("bottle_4",))
        # Agent 1 hasn't entered
        assert not tracker.agent_knows("agent_1", "is_open", ("cabinet_27",))

    def test_communication_transfers_beliefs(self):
        tracker = self._make_tracker()
        # Agent 0 knows something
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))
        tracker.record_observation("agent_0", "is_on_top", ("bottle_4", "table_13"))

        def check_fn(pred, args):
            return True

        # Agent 0 sends message mentioning cabinet_27
        tracker.record_communication(
            "agent_0", "agent_1",
            "The cabinet_27 is open.",
            check_fn,
        )
        # Agent 1 should know about cabinet_27 but not table_13 fact
        # (table_13 not mentioned in message)
        assert tracker.agent_knows("agent_1", "is_open", ("cabinet_27",))
        # bottle_4 wasn't mentioned, and table_13 wasn't mentioned
        assert not tracker.agent_knows("agent_1", "is_on_top", ("bottle_4", "table_13"))

    def test_communication_no_ids_transfers_all(self):
        """If no object IDs in message, transfer all sender beliefs."""
        tracker = self._make_tracker()
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))
        tracker.record_observation("agent_0", "is_clean", ("table_13",))

        tracker.record_communication(
            "agent_0", "agent_1",
            "I found some interesting stuff in the kitchen.",
            lambda p, a: True,
        )
        # No IDs in message → all beliefs transferred
        assert tracker.agent_knows("agent_1", "is_open", ("cabinet_27",))
        assert tracker.agent_knows("agent_1", "is_clean", ("table_13",))

    def test_state_change_observed_by_agents_in_room(self):
        tracker = self._make_tracker()
        tracker.agent_rooms["agent_0"] = "kitchen_1"
        tracker.agent_rooms["agent_1"] = "bedroom_1"

        tracker.record_state_change("is_open", ("cabinet_27",), "kitchen_1")
        assert tracker.agent_knows("agent_0", "is_open", ("cabinet_27",))
        assert not tracker.agent_knows("agent_1", "is_open", ("cabinet_27",))

    def test_agent_doesnt_know_unobserved(self):
        tracker = self._make_tracker()
        assert not tracker.agent_knows("agent_0", "is_open", ("cabinet_27",))

    def test_evaluate_epistemic_knows(self):
        tracker = self._make_tracker()
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))

        # K(agent_0, is_open(cabinet_27)) should be true
        formula = Knows("agent_0", Literal("is_open", ("cabinet_27",)))

        def world_check(pred, args):
            return pred == "is_open" and args == ("cabinet_27",)

        assert tracker.evaluate_epistemic(formula, world_check)

    def test_evaluate_epistemic_knows_false_world(self):
        """K(a, phi) requires phi to be true in the world."""
        tracker = self._make_tracker()
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))

        formula = Knows("agent_0", Literal("is_open", ("cabinet_27",)))

        # World says cabinet is NOT open anymore
        def world_check(pred, args):
            return False

        # Agent believes it but world disagrees → K should be false
        assert not tracker.evaluate_epistemic(formula, world_check)

    def test_evaluate_epistemic_believes(self):
        """B(a, phi) only requires agent belief, not world truth."""
        tracker = self._make_tracker()
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))

        formula = Believes("agent_0", Literal("is_open", ("cabinet_27",)))

        # B doesn't care about world state
        assert tracker.evaluate_epistemic(formula, lambda p, a: False)

    def test_evaluate_non_epistemic_literal(self):
        tracker = self._make_tracker()
        formula = Literal("is_open", ("cabinet_27",))

        assert tracker.evaluate_epistemic(formula, lambda p, a: True)
        assert not tracker.evaluate_epistemic(formula, lambda p, a: False)

    def test_reset(self):
        tracker = self._make_tracker()
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))
        tracker.agent_rooms["agent_0"] = "kitchen_1"
        tracker.reset()
        assert not tracker.agent_knows("agent_0", "is_open", ("cabinet_27",))
        assert "agent_0" not in tracker.agent_rooms


class TestGoalCheckerWithBeliefs:
    def test_k_goal_without_tracker_backward_compat(self):
        """Without belief tracker, K(a, phi) = phi (conservative)."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        checker = PDDLGoalChecker(goal=goal)  # No belief tracker

        result = checker.update(lambda pred, args: pred == "is_open" and args == ("cabinet_27",))
        assert result["all_complete"]

    def test_k_goal_with_tracker_requires_belief(self):
        """With belief tracker, K(a, phi) requires agent to know it."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        tracker = BeliefStateTracker.from_scene_and_observability(
            object_rooms={"cabinet_27": "kitchen_1"},
            num_agents=2,
        )
        checker = PDDLGoalChecker(goal=goal, belief_tracker=tracker)

        # World says cabinet is open, but agent doesn't know
        result = checker.update(lambda pred, args: pred == "is_open" and args == ("cabinet_27",))
        assert not result["all_complete"]

        # Now agent observes it
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))
        result = checker.update(lambda pred, args: pred == "is_open" and args == ("cabinet_27",))
        assert result["all_complete"]

    def test_mixed_epistemic_and_literal_goals(self):
        """Goals with both K() and plain literal conjuncts."""
        goal = And(operands=(
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        tracker = BeliefStateTracker.from_scene_and_observability(
            object_rooms={"cabinet_27": "kitchen_1", "bottle_4": "kitchen_1", "table_13": "bedroom_1"},
            num_agents=2,
        )
        checker = PDDLGoalChecker(goal=goal, belief_tracker=tracker)

        # Both true in world, but agent doesn't know about cabinet
        def world_check(pred, args):
            if pred == "is_open" and args == ("cabinet_27",):
                return True
            if pred == "is_on_top" and args == ("bottle_4", "table_13"):
                return True
            return False

        result = checker.update(world_check)
        # Literal goal should complete, K goal should not
        assert not result["all_complete"]
        assert len(result["completed"]) == 1

        # Agent learns about cabinet
        tracker.record_observation("agent_0", "is_open", ("cabinet_27",))
        result = checker.update(world_check)
        assert result["all_complete"]

    def test_from_task_data_with_tracker(self):
        """from_task_data passes belief tracker through."""
        task_data = {
            "pddl_goal": "(K agent_0 (is_open cabinet_27))",
            "pddl_ordering": [],
            "pddl_owners": {},
        }
        tracker = BeliefStateTracker.from_scene_and_observability(
            object_rooms={"cabinet_27": "kitchen_1"},
            num_agents=2,
        )
        checker = PDDLGoalChecker.from_task_data(task_data, belief_tracker=tracker)
        assert checker is not None
        assert checker._belief_tracker is tracker
