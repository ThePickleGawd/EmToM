"""
Visualize a task's PDDL goal as a DAG.

Renders pddl_goal conjuncts as nodes, pddl_ordering as edges,
with coloring by owner (team/agent) and shape by goal type (physical vs epistemic).

Usage:
    python -m emtom.cli.visualize_task task.json [-o output.png] [--format svg]
    ./emtom/run_emtom.sh visualize task.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Palette for agents/teams
COLORS = {
    "agent_0": "#4A90D9",
    "agent_1": "#E07B39",
    "agent_2": "#5BA55B",
    "agent_3": "#C75050",
    "agent_4": "#8E6BB0",
    "team_0": "#4A90D9",
    "team_1": "#E07B39",
    "team_2": "#5BA55B",
    "shared": "#888888",
}


def parse_conjuncts(pddl_goal: str) -> List[str]:
    """Extract top-level conjuncts from a PDDL goal string."""
    goal = pddl_goal.strip()

    # Strip outer (and ...)
    if goal.startswith("(and "):
        inner = goal[5:-1].strip()
    else:
        return [goal]

    # Split by balanced parens
    conjuncts = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                conjuncts.append(inner[start : i + 1].strip())
    return conjuncts


def classify_goal(conjunct: str) -> Tuple[str, str, int]:
    """
    Classify a goal conjunct.

    Returns:
        (label, type, tom_depth)
        type: 'physical', 'epistemic', 'negated'
    """
    # Count K/B nesting
    tom_depth = 0
    inner = conjunct
    while True:
        m = re.match(r'^\((?:K|B)\s+\w+\s+(.+)\)$', inner)
        if m:
            tom_depth += 1
            inner = m.group(1).strip()
        else:
            break

    if conjunct.startswith("(not "):
        return conjunct, "negated", tom_depth
    if tom_depth > 0:
        return conjunct, "epistemic", tom_depth
    return conjunct, "physical", 0


def shorten_label(conjunct: str) -> str:
    """Make a readable short label for a graph node."""
    # Strip outer K/B wrappers for display, show as prefix
    prefix_parts = []
    inner = conjunct
    while True:
        m = re.match(r'^\((K|B)\s+(\w+)\s+(.+)\)$', inner)
        if m:
            op, agent, rest = m.group(1), m.group(2), m.group(3)
            prefix_parts.append(f"{op}({agent})")
            inner = rest.strip()
        else:
            break

    # Parse the predicate
    m = re.match(r'^\((\w+)\s+(.*)\)$', inner)
    if m:
        pred, args = m.group(1), m.group(2)
        # Shorten common predicates
        short_pred = pred.replace("is_on_top", "on").replace("is_inside", "in")
        label = f"{short_pred}({args})"
    else:
        label = inner

    if prefix_parts:
        # Nest: K(agent_0,\n  K(agent_2, fact))
        for i, part in enumerate(reversed(prefix_parts)):
            indent = "  " * (len(prefix_parts) - i - 1)
            if i == 0:
                label = f"{part[:-1]},\\n{indent}{label})"
            else:
                label = f"{part[:-1]},\\n{indent}{label})"

    return label


def build_dag(task_data: Dict[str, Any]) -> Optional[Any]:
    """Build a graphviz Digraph from task PDDL data."""
    import graphviz

    pddl_goal = task_data.get("pddl_goal", "")
    ordering = task_data.get("pddl_ordering", [])
    owners = task_data.get("pddl_owners", {})
    # Filter out _COMMENT keys
    owners = {k: v for k, v in owners.items() if not k.startswith("_")}
    category = task_data.get("category", "cooperative")
    title = task_data.get("title", task_data.get("task_id", "Task"))
    num_agents = task_data.get("num_agents", 2)

    if not pddl_goal:
        return None

    conjuncts = parse_conjuncts(pddl_goal)
    if not conjuncts:
        return None

    # Build graph
    dot = graphviz.Digraph(
        name=title,
        format="png",
        graph_attr={
            "rankdir": "TB",
            "label": f"{title}\\n({category}, {num_agents} agents)",
            "labelloc": "t",
            "fontsize": "16",
            "fontname": "Helvetica",
            "bgcolor": "#FAFAFA",
            "pad": "0.5",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "style": "filled",
            "margin": "0.15,0.08",
        },
        edge_attr={
            "color": "#555555",
            "arrowsize": "0.8",
        },
    )

    # Create node IDs and classify
    node_ids = {}
    for i, conj in enumerate(conjuncts):
        node_id = f"g{i}"
        node_ids[conj] = node_id

        label = shorten_label(conj)
        _, goal_type, tom_depth = classify_goal(conj)

        # Shape by type
        if goal_type == "epistemic":
            shape = "diamond"
        elif goal_type == "negated":
            shape = "octagon"
        else:
            shape = "box"

        # Color by owner
        owner = owners.get(conj, "")
        if owner:
            fillcolor = COLORS.get(owner, "#DDDDDD")
        elif goal_type == "epistemic":
            # Extract agent from K/B
            m = re.match(r'^\((K|B)\s+(\w+)', conj)
            if m:
                fillcolor = COLORS.get(m.group(2), "#D4E6F1")
            else:
                fillcolor = "#D4E6F1"
        else:
            fillcolor = "#E8E8E8"

        # Lighter fill for readability
        fontcolor = "#000000"

        dot.node(
            node_id,
            label=label,
            shape=shape,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            penwidth="1.5" if tom_depth > 0 else "1.0",
        )

    # Add ordering edges
    for rule in ordering:
        before = rule.get("before", "")
        after = rule.get("after", "")
        if before in node_ids and after in node_ids:
            dot.edge(node_ids[before], node_ids[after])

    # Add legend
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(
            label="Legend",
            style="dashed",
            color="#AAAAAA",
            fontsize="10",
        )
        legend.node("leg_phys", "Physical goal", shape="box", fillcolor="#E8E8E8")
        legend.node("leg_epist", "K() epistemic", shape="diamond", fillcolor="#D4E6F1")
        legend.node("leg_neg", "Negated", shape="octagon", fillcolor="#E8E8E8")
        legend.edge("leg_phys", "leg_epist", style="invis")
        legend.edge("leg_epist", "leg_neg", style="invis")

    # Add mechanics annotation
    mechanics = task_data.get("active_mechanics", []) or task_data.get("mechanic_bindings", [])
    if mechanics:
        mech_names = []
        for m in mechanics:
            if isinstance(m, dict):
                mech_names.append(m.get("mechanic_type", "?"))
            elif isinstance(m, str):
                mech_names.append(m)
        if mech_names:
            dot.attr(
                label=f"{title}\\n({category}, {num_agents} agents)\\nmechanics: {', '.join(set(mech_names))}",
            )

    return dot


def main():
    parser = argparse.ArgumentParser(description="Visualize task PDDL goal as DAG")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output file path (without extension)")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"], help="Output format")
    parser.add_argument("--view", action="store_true", help="Open in viewer after rendering")
    args = parser.parse_args()

    with open(args.task_file) as f:
        task_data = json.load(f)

    dot = build_dag(task_data)
    if dot is None:
        print("No PDDL goal found in task.", file=sys.stderr)
        sys.exit(1)

    dot.format = args.format

    if args.output:
        output_path = args.output
    else:
        stem = Path(args.task_file).stem
        output_path = f"/tmp/emtom_dag_{stem}"

    rendered = dot.render(output_path, view=args.view, cleanup=True)
    print(rendered)


if __name__ == "__main__":
    main()
