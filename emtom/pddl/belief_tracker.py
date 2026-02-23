"""
Runtime belief state tracker for epistemic goal evaluation.

Tracks what each agent has observed or been told, enabling proper
evaluation of K(agent, phi) and B(agent, phi) goals.

An agent knows a fact if:
1. They were in the same room when the state changed (observed), OR
2. They received a Communicate action with info about that fact, OR
3. The fact was true in a room they entered (initial observation on entry)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from emtom.pddl.dsl import (
    Formula, Literal, And, Or, Not, Knows, Believes, EpistemicFormula,
)
from emtom.pddl.epistemic import ObservabilityModel


# Fact = (predicate_name, args_tuple)
Fact = Tuple[str, Tuple[str, ...]]

# Predicates that are checkable for observation on room entry
_OBSERVABLE_PREDICATES = {
    "is_open", "is_closed", "is_clean", "is_dirty",
    "is_filled", "is_empty", "is_powered_on", "is_powered_off",
    "is_on_top", "is_inside", "is_in_room", "is_on_floor",
    "is_held_by", "is_unlocked",
}


@dataclass
class BeliefStateTracker:
    """
    Tracks per-agent beliefs for runtime K()/B() evaluation.

    Thread-safe for single-threaded game loop (not concurrent).
    """

    # agent -> set of believed facts
    beliefs: Dict[str, Set[Fact]] = field(default_factory=dict)

    # agent -> current room
    agent_rooms: Dict[str, str] = field(default_factory=dict)

    # object -> room mapping (from scene data, static)
    object_rooms: Dict[str, str] = field(default_factory=dict)

    # room -> list of objects in that room (reverse of object_rooms)
    _room_objects: Dict[str, List[str]] = field(default_factory=dict)

    # observability model
    observability: Optional[ObservabilityModel] = None

    def __post_init__(self):
        self._rebuild_room_objects()

    def _rebuild_room_objects(self) -> None:
        """Rebuild reverse mapping from object_rooms."""
        self._room_objects.clear()
        for obj, room in self.object_rooms.items():
            self._room_objects.setdefault(room, []).append(obj)

    @classmethod
    def from_scene_and_observability(
        cls,
        object_rooms: Dict[str, str],
        observability: Optional[ObservabilityModel] = None,
        num_agents: int = 2,
    ) -> "BeliefStateTracker":
        """Create a tracker from scene object-room mapping."""
        tracker = cls(
            object_rooms=dict(object_rooms),
            observability=observability,
        )
        tracker._rebuild_room_objects()
        # Initialize empty belief sets for all agents
        for i in range(num_agents):
            tracker.beliefs.setdefault(f"agent_{i}", set())
        return tracker

    def record_observation(
        self,
        agent: str,
        predicate: str,
        args: Tuple[str, ...],
    ) -> None:
        """Record that an agent has observed a fact."""
        fact: Fact = (predicate, args)
        self.beliefs.setdefault(agent, set()).add(fact)

    def record_room_entry(
        self,
        agent: str,
        room: str,
        check_fn: Callable[[str, Tuple[str, ...]], bool],
    ) -> None:
        """
        When agent enters a room, they observe all visible facts there.

        Uses check_fn(predicate, args) -> bool to query current state.
        """
        self.agent_rooms[agent] = room

        # Find all objects in this room
        objects_in_room = self._room_objects.get(room, [])

        agent_beliefs = self.beliefs.setdefault(agent, set())

        # Check unary predicates for each object/furniture in room
        _UNARY_PREDS = {"is_open", "is_closed", "is_clean", "is_dirty",
                        "is_filled", "is_empty", "is_powered_on", "is_powered_off",
                        "is_unlocked", "is_on_floor"}
        for obj in objects_in_room:
            for pred in _UNARY_PREDS:
                args = (obj,)
                if check_fn(pred, args):
                    agent_beliefs.add((pred, args))

        # Check binary predicates (is_on_top, is_inside) between all
        # objects/furniture in the room. This catches facts like
        # is_on_top(laptop_0, table_29) when the agent enters the room.
        _BINARY_PREDS = {"is_on_top", "is_inside", "is_in_room"}
        all_in_room = list(objects_in_room)
        for entity in all_in_room:
            for target in all_in_room:
                if entity != target:
                    for pred in _BINARY_PREDS:
                        args = (entity, target)
                        if check_fn(pred, args):
                            agent_beliefs.add((pred, args))

    def record_communication(
        self,
        sender: str,
        receiver: str,
        message: str,
        check_fn: Callable[[str, Tuple[str, ...]], bool],
    ) -> None:
        """
        When an agent sends a message, the receiver gains knowledge.

        Transfer sender's beliefs about entities mentioned (by ID) in
        the message to the receiver. Also check binary predicates between
        mentioned entities against the world state.
        """
        sender_beliefs = self.beliefs.setdefault(sender, set())
        receiver_beliefs = self.beliefs.setdefault(receiver, set())

        # Find object IDs mentioned in the message
        # Match patterns like: cabinet_27, bottle_4, item_key_1, etc.
        mentioned_ids = set(re.findall(r'\b([a-z][a-z_]*_\d+)\b', message))
        # Exclude agent IDs from mentioned objects
        mentioned_ids = {mid for mid in mentioned_ids if not mid.startswith("agent_")}

        # Transfer sender's beliefs about mentioned objects
        for fact in list(sender_beliefs):
            pred, args = fact
            if any(arg in mentioned_ids for arg in args):
                receiver_beliefs.add(fact)

        # Check binary predicates between mentioned ID pairs against world state.
        # When sender mentions e.g. "laptop_0 on table_29", verify the
        # relationship and record it as a belief for both sender and receiver.
        _BINARY_PREDS = ("is_on_top", "is_inside", "is_in_room")
        mentioned_list = sorted(mentioned_ids)
        for i, id1 in enumerate(mentioned_list):
            for id2 in mentioned_list[i + 1:]:
                for pred in _BINARY_PREDS:
                    for args in [(id1, id2), (id2, id1)]:
                        if check_fn(pred, args):
                            fact = (pred, args)
                            sender_beliefs.add(fact)
                            receiver_beliefs.add(fact)

        if not mentioned_ids:
            # If no IDs mentioned, transfer all sender beliefs
            receiver_beliefs.update(sender_beliefs)

    def record_state_change(
        self,
        predicate: str,
        args: Tuple[str, ...],
        change_room: Optional[str],
    ) -> None:
        """
        When state changes, all agents in the same room observe it.

        Args:
            predicate: The predicate that changed
            args: The args of the predicate
            change_room: The room where the change occurred
        """
        if not change_room:
            return

        fact: Fact = (predicate, args)
        for agent, room in self.agent_rooms.items():
            if room == change_room:
                self.beliefs.setdefault(agent, set()).add(fact)

    def agent_knows(
        self,
        agent: str,
        predicate: str,
        args: Tuple[str, ...],
    ) -> bool:
        """Check if agent knows (predicate, args) is true."""
        fact: Fact = (predicate, args)
        return fact in self.beliefs.get(agent, set())

    def evaluate_epistemic(
        self,
        formula: Formula,
        check_fn: Callable[[str, Tuple[str, ...]], bool],
    ) -> bool:
        """
        Evaluate a formula with epistemic awareness.

        - Literal: delegates to check_fn (world truth)
        - Knows(agent, phi): agent_knows AND phi is true in the world
        - Believes(agent, phi): agent has phi in beliefs (may not be true)
        - And/Or/Not: standard logic
        """
        if isinstance(formula, Knows):
            # K(a, phi) = phi is true in the world AND agent knows it
            world_true = self._eval_inner(formula.inner, check_fn)
            if not world_true:
                return False
            return self._agent_believes_formula(formula.agent, formula.inner)

        if isinstance(formula, Believes):
            # B(a, phi) = agent believes phi (may or may not be true)
            return self._agent_believes_formula(formula.agent, formula.inner)

        if isinstance(formula, Literal):
            return formula.evaluate(check_fn)

        if isinstance(formula, And):
            return all(
                self.evaluate_epistemic(op, check_fn) for op in formula.operands
            )

        if isinstance(formula, Or):
            return any(
                self.evaluate_epistemic(op, check_fn) for op in formula.operands
            )

        if isinstance(formula, Not):
            return not self.evaluate_epistemic(formula.operand, check_fn)

        # Fallback: delegate to formula's own evaluate
        return formula.evaluate(check_fn)

    def _eval_inner(
        self,
        formula: Formula,
        check_fn: Callable[[str, Tuple[str, ...]], bool],
    ) -> bool:
        """Evaluate inner formula against world state (non-epistemic)."""
        if isinstance(formula, EpistemicFormula):
            # For nested epistemic, evaluate recursively
            return self.evaluate_epistemic(formula, check_fn)
        return formula.evaluate(check_fn)

    def _agent_believes_formula(self, agent: str, formula: Formula) -> bool:
        """Check if agent has beliefs supporting the formula."""
        if isinstance(formula, Literal):
            return self.agent_knows(agent, formula.predicate, formula.args)

        if isinstance(formula, And):
            return all(
                self._agent_believes_formula(agent, op) for op in formula.operands
            )

        if isinstance(formula, Or):
            return any(
                self._agent_believes_formula(agent, op) for op in formula.operands
            )

        if isinstance(formula, Not):
            # Agent believes not-phi if they don't believe phi
            # (closed-world assumption on beliefs)
            return not self._agent_believes_formula(agent, formula.operand)

        if isinstance(formula, (Knows, Believes)):
            # Nested: K(a, K(b, phi)) — does agent_a believe that agent_b knows phi?
            # Simplified: agent_a knows phi AND agent_b knows phi
            inner_agent = formula.agent
            return (
                self._agent_believes_formula(agent, formula.inner)
                and self._agent_believes_formula(inner_agent, formula.inner)
            )

        return False

    def reset(self) -> None:
        """Reset all beliefs."""
        self.beliefs.clear()
        self.agent_rooms.clear()
