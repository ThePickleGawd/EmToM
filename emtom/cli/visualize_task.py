"""
Visualize a task's PDDL goal as a DAG.

Reads the ``goals`` array (GoalEntry format with index-based ``after``
dependencies) when present, falling back to legacy ``pddl_goal`` /
``pddl_ordering`` / ``pddl_owners`` fields.  Nodes are colored by owner
and shaped by goal type (physical / epistemic / negated).

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
    """Build a graphviz Digraph from task goal data."""
    import graphviz

    category = task_data.get("category", "cooperative")
    title = task_data.get("title", task_data.get("task_id", "Task"))
    num_agents = task_data.get("num_agents", 2)

    # Try goals array first, fall back to legacy
    goals_array = task_data.get("goals")
    if goals_array:
        # New format: goals array with index-based ordering
        entries = []
        for g in goals_array:
            entries.append({
                "id": g["id"],
                "pddl": g["pddl"],
                "after": g.get("after", []),
                "owner": g.get("owner"),
            })
    else:
        # Legacy format: parse pddl_goal + pddl_ordering + pddl_owners
        pddl_goal = task_data.get("pddl_goal", "")
        ordering = task_data.get("pddl_ordering", [])
        owners = task_data.get("pddl_owners", {})
        owners = {k: v for k, v in owners.items() if not k.startswith("_")}

        if not pddl_goal:
            return None

        conjuncts = parse_conjuncts(pddl_goal)
        if not conjuncts:
            return None

        # Build index-based entries from legacy format
        pddl_to_idx = {c: i for i, c in enumerate(conjuncts)}
        after_map: Dict[int, List[int]] = {i: [] for i in range(len(conjuncts))}
        for rule in ordering:
            before_str = rule.get("before", "").strip()
            after_str = rule.get("after", "").strip()
            before_idx = pddl_to_idx.get(before_str)
            after_idx = pddl_to_idx.get(after_str)
            if before_idx is not None and after_idx is not None:
                after_map[after_idx].append(before_idx)

        entries = []
        for i, conj in enumerate(conjuncts):
            entries.append({
                "id": i,
                "pddl": conj,
                "after": after_map[i],
                "owner": owners.get(conj),
            })

    if not entries:
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

    # Create nodes
    id_to_node = {}
    for entry in entries:
        gid = entry["id"]
        pddl = entry["pddl"]
        owner = entry.get("owner")
        node_id = f"g{gid}"
        id_to_node[gid] = node_id

        label = shorten_label(pddl)
        _, goal_type, tom_depth = classify_goal(pddl)

        # Shape by type
        if goal_type == "epistemic":
            shape = "diamond"
        elif goal_type == "negated":
            shape = "octagon"
        else:
            shape = "box"

        # Color by owner
        if owner:
            fillcolor = COLORS.get(owner, "#DDDDDD")
        elif goal_type == "epistemic":
            m = re.match(r'^\((K|B)\s+(\w+)', pddl)
            if m:
                fillcolor = COLORS.get(m.group(2), "#D4E6F1")
            else:
                fillcolor = "#D4E6F1"
        else:
            fillcolor = "#E8E8E8"

        dot.node(
            node_id,
            label=label,
            shape=shape,
            fillcolor=fillcolor,
            fontcolor="#000000",
            penwidth="1.5" if tom_depth > 0 else "1.0",
        )

    # Add ordering edges from 'after' references
    for entry in entries:
        gid = entry["id"]
        for dep_id in entry.get("after", []):
            if dep_id in id_to_node and gid in id_to_node:
                dot.edge(id_to_node[dep_id], id_to_node[gid])

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
