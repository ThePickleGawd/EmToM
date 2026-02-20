"""
Shared EmToM PDDL domain definition.

Encodes all EmToM actions, predicates, and types with conditional effects
for mechanics (inverse_state, state_mirroring, remote_control, etc.).
"""

from emtom.pddl.dsl import (
    Type,
    Predicate,
    Param,
    Action,
    Effect,
    Literal,
    And,
    Not,
    Domain,
)


# ---------------------------------------------------------------------------
# Type hierarchy
# ---------------------------------------------------------------------------

EMTOM_TYPES = [
    Type("agent"),
    Type("object"),
    Type("furniture", parent="object"),
    Type("room"),
    Type("item"),
]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

EMTOM_PREDICATES = [
    # Spatial / relational
    Predicate("is_on_top", (Param("x", "object"), Param("y", "furniture"))),
    Predicate("is_inside", (Param("x", "object"), Param("y", "furniture"))),
    Predicate("is_in_room", (Param("x", "object"), Param("r", "room"))),
    Predicate("is_on_floor", (Param("x", "object"),)),
    Predicate("is_next_to", (Param("x", "object"), Param("y", "object"))),

    # Unary state
    Predicate("is_open", (Param("f", "furniture"),)),
    Predicate("is_closed", (Param("f", "furniture"),)),
    Predicate("is_clean", (Param("x", "object"),)),
    Predicate("is_dirty", (Param("x", "object"),)),
    Predicate("is_filled", (Param("x", "object"),)),
    Predicate("is_empty", (Param("x", "object"),)),
    Predicate("is_powered_on", (Param("x", "object"),)),
    Predicate("is_powered_off", (Param("x", "object"),)),
    Predicate("is_unlocked", (Param("f", "furniture"),)),

    # Agent predicates
    Predicate("is_held_by", (Param("x", "object"), Param("a", "agent"))),
    Predicate("agent_in_room", (Param("a", "agent"), Param("r", "room"))),

    # Game state predicates
    Predicate("has_item", (Param("a", "agent"), Param("i", "item"))),
    Predicate("has_at_least", (Param("a", "agent"), Param("i", "item"))),
    Predicate("has_most", (Param("a", "agent"), Param("i", "item"))),

    # Mechanic predicates
    Predicate("is_inverse", (Param("f", "furniture"),)),
    Predicate("mirrors", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("is_restricted", (Param("a", "agent"), Param("r", "room"))),
    Predicate("is_locked_permanent", (Param("f", "furniture"),)),
    Predicate("requires_item", (Param("f", "furniture"), Param("i", "item"))),
]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

EMTOM_ACTIONS = [
    # Open: with conditional effects for mechanics
    Action(
        name="open",
        params=[Param("a", "agent"), Param("f", "furniture")],
        preconditions=And(operands=(
            Literal("is_closed", ("?f",)),
            Not(operand=Literal("is_locked_permanent", ("?f",))),
        )),
        effects=[
            Effect(Literal("is_open", ("?f",))),
            Effect(Literal("is_closed", ("?f",), negated=True)),
            # Conditional: inverse mechanic reverses the effect
            Effect(
                Literal("is_closed", ("?f",)),
                condition=Literal("is_inverse", ("?f",)),
            ),
            # Conditional: state mirroring propagates
            Effect(
                Literal("is_open", ("?g",)),
                condition=Literal("mirrors", ("?f", "?g")),
            ),
            # Conditional: remote control triggers unlock
            Effect(
                Literal("is_unlocked", ("?g",)),
                condition=Literal("controls", ("?f", "?g")),
            ),
        ],
        observability="full",
    ),

    # Close
    Action(
        name="close",
        params=[Param("a", "agent"), Param("f", "furniture")],
        preconditions=Literal("is_open", ("?f",)),
        effects=[
            Effect(Literal("is_closed", ("?f",))),
            Effect(Literal("is_open", ("?f",), negated=True)),
            Effect(
                Literal("is_open", ("?f",)),
                condition=Literal("is_inverse", ("?f",)),
            ),
            Effect(
                Literal("is_closed", ("?g",)),
                condition=Literal("mirrors", ("?f", "?g")),
            ),
        ],
        observability="full",
    ),

    # Navigate
    Action(
        name="navigate",
        params=[Param("a", "agent"), Param("r", "room")],
        preconditions=Not(operand=Literal("is_restricted", ("?a", "?r"))),
        effects=[
            Effect(Literal("agent_in_room", ("?a", "?r"))),
        ],
        observability="full",
    ),

    # Pick
    Action(
        name="pick",
        params=[Param("a", "agent"), Param("x", "object")],
        preconditions=None,
        effects=[
            Effect(Literal("is_held_by", ("?x", "?a"))),
        ],
        observability="full",
    ),

    # Place
    Action(
        name="place",
        params=[Param("a", "agent"), Param("x", "object"), Param("f", "furniture")],
        preconditions=Literal("is_held_by", ("?x", "?a")),
        effects=[
            Effect(Literal("is_held_by", ("?x", "?a"), negated=True)),
            Effect(Literal("is_on_top", ("?x", "?f"))),
        ],
        observability="full",
    ),

    # Communicate: epistemic effect — transfers knowledge
    Action(
        name="communicate",
        params=[Param("from", "agent"), Param("to", "agent")],
        preconditions=None,
        effects=[],  # Epistemic effects handled by the solver
        observability="full",
    ),

    # Wait: no-op
    Action(
        name="wait",
        params=[Param("a", "agent")],
        preconditions=None,
        effects=[],
        observability="full",
    ),

    # UseItem
    Action(
        name="use_item",
        params=[Param("a", "agent"), Param("i", "item"), Param("f", "furniture")],
        preconditions=And(operands=(
            Literal("has_item", ("?a", "?i")),
            Literal("requires_item", ("?f", "?i")),
        )),
        effects=[
            Effect(Literal("is_unlocked", ("?f",))),
        ],
        observability="full",
    ),
]


# ---------------------------------------------------------------------------
# Domain singleton
# ---------------------------------------------------------------------------

EMTOM_DOMAIN = Domain(
    name="emtom",
    types=EMTOM_TYPES,
    predicates=EMTOM_PREDICATES,
    actions=EMTOM_ACTIONS,
)
