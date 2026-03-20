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
    ForallEffect,
    Literal,
    And,
    Formula,
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
    Type("item", parent="object"),
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
    Predicate("is_locked", (Param("f", "furniture"),)),

    # Agent predicates
    Predicate("is_held_by", (Param("x", "object"), Param("a", "agent"))),
    Predicate("agent_in_room", (Param("a", "agent"), Param("r", "room"))),

    # Game state predicates
    Predicate("has_item", (Param("a", "agent"), Param("i", "item"))),
    Predicate("has_at_least", (Param("a", "agent"), Param("i", "item"))),
    Predicate("has_most", (Param("a", "agent"), Param("i", "item"))),
    Predicate("item_in_container", (Param("i", "item"), Param("f", "furniture"))),

    # Mechanic predicates
    Predicate("is_inverse", (Param("f", "furniture"),)),
    Predicate("mirrors", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("mirrors_closed", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls_unlocked", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls_closed", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls_locks", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("is_restricted", (Param("a", "agent"), Param("r", "room"))),
    Predicate("is_locked_permanent", (Param("f", "furniture"),)),
    Predicate("requires_item", (Param("f", "furniture"), Param("i", "item"))),
    Predicate("unlocks", (Param("x", "object"), Param("f", "furniture"))),
    Predicate("irreversible_enabled", (Param("x", "object"),)),
    Predicate("interaction_locked", (Param("x", "object"),)),
    Predicate("can_communicate", (Param("from", "agent"), Param("to", "agent"))),
]


# ---------------------------------------------------------------------------
# Predicate descriptions (for prompt generation)
# ---------------------------------------------------------------------------

# Maps predicate name to a one-line description for the LLM prompt.
# Grouped by comment headers in EMTOM_PREDICATES above.
_PREDICATE_DESCRIPTIONS = {
    "is_on_top": "object is on top of furniture",
    "is_inside": "object is inside furniture (container)",
    "is_in_room": "object is located in room",
    "is_on_floor": "object is on the floor",
    "is_next_to": "object is adjacent to another object",
    "is_open": "furniture is open",
    "is_closed": "furniture is closed",
    "is_clean": "object is clean",
    "is_dirty": "object is dirty",
    "is_filled": "object is filled with liquid",
    "is_empty": "object is empty",
    "is_powered_on": "object is powered on",
    "is_powered_off": "object is powered off",
    "is_unlocked": "furniture is unlocked",
    "is_locked": "furniture is locked",
    "is_held_by": "object is held by agent",
    "agent_in_room": "agent is in room",
    "has_item": "agent has item in inventory",
    "has_at_least": "agent has at least N of item",
    "has_most": "agent has the most of item among all agents",
    "item_in_container": "(planner) item is hidden inside furniture until opened",
    "is_inverse": "(mechanic) furniture has inverted open/close",
    "mirrors": "(mechanic) furniture1 state mirrors furniture2",
    "mirrors_closed": "(mechanic) furniture1 open/close toggles furniture2 closed/open state",
    "controls": "(mechanic) furniture1 remotely controls furniture2",
    "controls_unlocked": "(mechanic) furniture1 remotely controls furniture2 unlocked/locked state",
    "controls_closed": "(mechanic) furniture1 remotely controls furniture2 closed/open state",
    "controls_locks": "(mechanic) furniture1 remotely controls furniture2 locked/unlocked state",
    "is_restricted": "(mechanic) agent cannot enter room",
    "is_locked_permanent": "(mechanic) furniture is locked until key used",
    "requires_item": "(mechanic) furniture requires item to unlock",
    "unlocks": "(mechanic) interacting with furniture1 unlocks furniture2",
    "irreversible_enabled": "(mechanic) object becomes interaction-locked after one use",
    "interaction_locked": "(mechanic) object can no longer be targeted by interactions",
    "can_communicate": "(mechanic) agent can send messages to another agent",
}

_PREDICATE_GROUPS = [
    ("Spatial / Relational", ["is_on_top", "is_inside", "is_in_room", "is_on_floor", "is_next_to"]),
    ("Unary State", ["is_open", "is_closed", "is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off", "is_unlocked", "is_locked"]),
    ("Agent", ["is_held_by", "agent_in_room", "has_item", "has_at_least", "has_most", "item_in_container"]),
    ("Mechanic (init-only, do NOT use in pddl_goal)", ["is_inverse", "mirrors", "mirrors_closed", "controls", "controls_unlocked", "controls_closed", "controls_locks", "is_restricted", "is_locked_permanent", "requires_item", "unlocks", "irreversible_enabled", "interaction_locked", "can_communicate"]),
]

INIT_ONLY_PREDICATES = {
    "is_inverse",
    "mirrors",
    "mirrors_closed",
    "controls",
    "controls_unlocked",
    "controls_closed",
    "controls_locks",
    "is_restricted",
    "is_locked_permanent",
    "requires_item",
    "unlocks",
    "irreversible_enabled",
    "interaction_locked",
    "can_communicate",
    "item_in_container",
}


def validate_goal_formula_allowed(formula: Formula) -> list[str]:
    """Reject init-only mechanic predicates anywhere in goal space."""
    from emtom.pddl.dsl import And, Believes, Knows, Literal, Not, Or

    errors: list[str] = []

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            if node.predicate in INIT_ONLY_PREDICATES:
                errors.append(
                    f"Predicate '{node.predicate}' is init-only and cannot appear in pddl_goal: {node.to_pddl()}"
                )
            return
        if isinstance(node, (Knows, Believes)):
            _walk(node.inner)
            return
        if isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)
            return
        if isinstance(node, (And, Or)):
            for operand in node.operands:
                _walk(operand)

    _walk(formula)
    return errors


def get_predicates_for_prompt() -> str:
    """
    Generate predicate signatures for the LLM system prompt.

    Dynamically derived from EMTOM_PREDICATES — never hardcoded.
    """
    pred_map = {p.name: p for p in EMTOM_PREDICATES}
    lines = []
    for group_name, pred_names in _PREDICATE_GROUPS:
        lines.append(f"### {group_name}")
        for name in pred_names:
            pred = pred_map.get(name)
            if not pred:
                continue
            params_str = " ".join(f"{p.name}:{p.type}" for p in pred.params)
            desc = _PREDICATE_DESCRIPTIONS.get(name, "")
            lines.append(f"- `({name} {params_str})` — {desc}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

EMTOM_ACTIONS = [
    # Open: with conditional effects for mechanics
    Action(
        name="open",
        params=[Param("a", "agent"), Param("f", "furniture"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?f", "?r")),
            Literal("is_closed", ("?f",)),
            Not(operand=Literal("is_locked", ("?f",))),
            Not(operand=Literal("is_locked_permanent", ("?f",))),
            Not(operand=Literal("interaction_locked", ("?f",))),
        )),
        effects=[
            Effect(Literal("is_open", ("?f",))),
            Effect(Literal("is_closed", ("?f",), negated=True)),
            # Conditional: inverse mechanic — undo the open, leave it closed
            Effect(
                Literal("is_closed", ("?f",)),
                condition=Literal("is_inverse", ("?f",)),
            ),
            Effect(
                Literal("is_open", ("?f",), negated=True),
                condition=Literal("is_inverse", ("?f",)),
            ),
            # Conditional: state mirroring propagates (forall quantified)
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors_closed", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            # Conditional: remote control triggers unlock (forall quantified)
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_unlocked", ("?f", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_closed", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_locks", ("?f", "?g")),
                effect=Literal("is_locked", ("?g",)),
                negative_effect=Literal("is_unlocked", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("unlocks", ("?f", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            ForallEffect(
                variable=Param("i", "item"),
                condition=Literal("item_in_container", ("?i", "?f")),
                effect=Literal("has_item", ("?a", "?i")),
                negative_effect=Literal("item_in_container", ("?i", "?f")),
            ),
            Effect(
                Literal("interaction_locked", ("?f",)),
                condition=Literal("irreversible_enabled", ("?f",)),
            ),
        ],
        observability="full",
    ),

    # Close
    Action(
        name="close",
        params=[Param("a", "agent"), Param("f", "furniture"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?f", "?r")),
            Literal("is_open", ("?f",)),
            Not(operand=Literal("interaction_locked", ("?f",))),
        )),
        effects=[
            Effect(Literal("is_closed", ("?f",))),
            Effect(Literal("is_open", ("?f",), negated=True)),
            # Conditional: inverse mechanic — undo the close, leave it open
            Effect(
                Literal("is_open", ("?f",)),
                condition=Literal("is_inverse", ("?f",)),
            ),
            Effect(
                Literal("is_closed", ("?f",), negated=True),
                condition=Literal("is_inverse", ("?f",)),
            ),
            # Conditional: state mirroring propagates (forall quantified)
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors_closed", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_unlocked", ("?f", "?g")),
                effect=Literal("is_locked", ("?g",)),
                negative_effect=Literal("is_unlocked", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_closed", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_locks", ("?f", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("unlocks", ("?f", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            Effect(
                Literal("interaction_locked", ("?f",)),
                condition=Literal("irreversible_enabled", ("?f",)),
            ),
        ],
        observability="full",
    ),

    # Navigate: move agent to a room, remove from old room
    Action(
        name="navigate",
        params=[Param("a", "agent"), Param("r", "room")],
        preconditions=Not(operand=Literal("is_restricted", ("?a", "?r"))),
        effects=[
            Effect(Literal("agent_in_room", ("?a", "?r"))),
            # Remove agent from any previous room
            ForallEffect(
                variable=Param("old", "room"),
                condition=Literal("agent_in_room", ("?a", "?old")),
                effect=Literal("agent_in_room", ("?a", "?r")),
                negative_effect=Literal("agent_in_room", ("?a", "?old")),
            ),
        ],
        observability="full",
    ),

    # Pick
    Action(
        name="pick",
        params=[Param("a", "agent"), Param("x", "object"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?x", "?r")),
            Not(operand=Literal("interaction_locked", ("?x",))),
        )),
        effects=[
            Effect(Literal("is_held_by", ("?x", "?a"))),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("unlocks", ("?x", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            Effect(
                Literal("interaction_locked", ("?x",)),
                condition=Literal("irreversible_enabled", ("?x",)),
            ),
        ],
        observability="full",
    ),

    # Place
    Action(
        name="place",
        params=[Param("a", "agent"), Param("x", "object"), Param("f", "furniture"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("is_held_by", ("?x", "?a")),
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?f", "?r")),
            Not(operand=Literal("interaction_locked", ("?x",))),
        )),
        effects=[
            Effect(Literal("is_held_by", ("?x", "?a"), negated=True)),
            # Domain-level abstraction: runtime Place can realize either
            # on-top or within placement depending on relation argument.
            # We expose both so solvability checks remain aligned with
            # task-level goals authored in problem_pddl.
            Effect(Literal("is_on_top", ("?x", "?f"))),
            Effect(Literal("is_inside", ("?x", "?f"))),
            Effect(Literal("is_in_room", ("?x", "?r"))),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("unlocks", ("?x", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            Effect(
                Literal("interaction_locked", ("?x",)),
                condition=Literal("irreversible_enabled", ("?x",)),
            ),
        ],
        observability="full",
    ),

    # Communicate: epistemic effect — transfers knowledge
    Action(
        name="communicate",
        params=[Param("from", "agent"), Param("to", "agent")],
        preconditions=Literal("can_communicate", ("?from", "?to")),
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
        params=[Param("a", "agent"), Param("i", "item"), Param("f", "furniture"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("has_item", ("?a", "?i")),
            Literal("requires_item", ("?f", "?i")),
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?f", "?r")),
            Not(operand=Literal("interaction_locked", ("?i",))),
        )),
        effects=[
            Effect(Literal("is_unlocked", ("?f",))),
            Effect(Literal("is_locked", ("?f",), negated=True)),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("unlocks", ("?i", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            Effect(
                Literal("interaction_locked", ("?i",)),
                condition=Literal("irreversible_enabled", ("?i",)),
            ),
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
