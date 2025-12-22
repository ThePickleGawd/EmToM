"""
DAG Visualization utilities for EMTOM subtask graphs.

Provides text-based and Graphviz DOT visualizations of task DAGs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .task_generator import Subtask, GeneratedTask

from .dag import (
    find_root_nodes,
    find_terminal_nodes,
    get_subtask_id,
    topological_sort,
    validate_dag,
)


def visualize_dag(
    subtasks: List["Subtask"],
    title: Optional[str] = None,
    show_conditions: bool = True,
    show_agents: bool = True,
) -> str:
    """
    Generate ASCII visualization of a subtask DAG.

    Args:
        subtasks: List of Subtask objects
        title: Optional title to display
        show_conditions: Whether to show success_condition details
        show_agents: Whether to show assigned agents

    Returns:
        Multi-line string representation of the DAG
    """
    if not subtasks:
        return "Empty DAG (no subtasks)"

    lines = []

    # Header
    if title:
        lines.append(f"╔{'═' * (len(title) + 2)}╗")
        lines.append(f"║ {title} ║")
        lines.append(f"╚{'═' * (len(title) + 2)}╝")
        lines.append("")

    # Validate DAG
    is_valid, errors = validate_dag(subtasks)
    if not is_valid:
        lines.append("⚠️  DAG VALIDATION ERRORS:")
        for err in errors:
            lines.append(f"   - {err}")
        lines.append("")

    # Build adjacency info
    id_to_subtask = {get_subtask_id(s): s for s in subtasks}
    roots = find_root_nodes(subtasks)
    terminals = find_terminal_nodes(subtasks)
    root_ids = {get_subtask_id(s) for s in roots}
    terminal_ids = {get_subtask_id(s) for s in terminals}

    # Stats
    lines.append(f"📊 Stats: {len(subtasks)} subtasks | {len(roots)} root(s) | {len(terminals)} terminal(s)")
    lines.append("")

    # Try to sort topologically for display order
    try:
        sorted_subtasks = topological_sort(subtasks)
    except ValueError:
        sorted_subtasks = subtasks  # Fall back if cycle

    # Build reverse adjacency (who depends on this node)
    dependents: Dict[str, List[str]] = {get_subtask_id(s): [] for s in subtasks}
    for s in subtasks:
        for dep in s.depends_on:
            if dep in dependents:
                dependents[dep].append(get_subtask_id(s))

    # Display each node
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│                          DAG Structure                          │")
    lines.append("└─────────────────────────────────────────────────────────────────┘")
    lines.append("")

    for subtask in sorted_subtasks:
        sid = get_subtask_id(subtask)

        # Node type indicators
        indicators = []
        if sid in root_ids:
            indicators.append("🌱 ROOT")
        if sid in terminal_ids:
            indicators.append("🎯 TERMINAL")
        indicator_str = f" ({', '.join(indicators)})" if indicators else ""

        # Node header
        lines.append(f"┌─ [{sid}]{indicator_str}")

        # Description
        if subtask.description:
            desc = subtask.description[:60] + "..." if len(subtask.description) > 60 else subtask.description
            lines.append(f"│  📝 {desc}")

        # Assigned agent
        if show_agents and subtask.assigned_agent:
            lines.append(f"│  👤 Assigned: {subtask.assigned_agent}")

        # Success condition
        if show_conditions and subtask.success_condition:
            cond = subtask.success_condition
            entity = cond.get("entity", "?")
            prop = cond.get("property", "?")
            val = cond.get("value", cond.get("target", "?"))
            lines.append(f"│  ✓  Condition: {entity}.{prop} = {val}")

        # Dependencies (incoming edges)
        if subtask.depends_on:
            deps_str = ", ".join(subtask.depends_on)
            lines.append(f"│  ← Depends on: {deps_str}")

        # Dependents (outgoing edges)
        if dependents.get(sid):
            deps_str = ", ".join(dependents[sid])
            lines.append(f"│  → Required by: {deps_str}")

        lines.append("└")
        lines.append("")

    # Summary
    lines.append("─" * 60)
    lines.append("Legend: 🌱 = root (no dependencies) | 🎯 = terminal (final outcome)")

    return "\n".join(lines)


def visualize_task_dag(task: "GeneratedTask") -> str:
    """
    Visualize the DAG from a GeneratedTask.

    Args:
        task: GeneratedTask object

    Returns:
        Multi-line string representation
    """
    return visualize_dag(
        task.subtasks,
        title=f"Task: {task.title}",
        show_conditions=True,
        show_agents=True,
    )


def to_dot(
    subtasks: List["Subtask"],
    graph_name: str = "task_dag",
    title: Optional[str] = None,
) -> str:
    """
    Export DAG to Graphviz DOT format.

    Args:
        subtasks: List of Subtask objects
        graph_name: Name for the graph
        title: Optional title/label for the graph

    Returns:
        DOT format string (can be rendered with graphviz)
    """
    if not subtasks:
        return f"digraph {graph_name} {{ }}"

    lines = [f"digraph {graph_name} {{"]
    lines.append("    rankdir=TB;")  # Top to bottom
    lines.append("    node [shape=box, style=rounded];")

    if title:
        lines.append(f'    label="{title}";')
        lines.append("    labelloc=t;")
    lines.append("")

    # Identify root and terminal nodes
    roots = find_root_nodes(subtasks)
    terminals = find_terminal_nodes(subtasks)
    root_ids = {str(get_subtask_id(s)) for s in roots}
    terminal_ids = {str(get_subtask_id(s)) for s in terminals}

    # Add nodes
    for subtask in subtasks:
        sid = str(get_subtask_id(subtask))

        # Build label
        label_parts = [sid]
        if subtask.description:
            desc = subtask.description[:40]
            if len(subtask.description) > 40:
                desc += "..."
            label_parts.append(desc)
        if subtask.assigned_agent:
            label_parts.append(f"[{subtask.assigned_agent}]")

        label = "\\n".join(label_parts)

        # Style based on node type
        style = []
        if sid in root_ids:
            style.append("fillcolor=lightgreen")
            style.append("style=\"rounded,filled\"")
        elif sid in terminal_ids:
            style.append("fillcolor=lightblue")
            style.append("style=\"rounded,filled\"")

        style_str = ", ".join(style) if style else ""
        if style_str:
            lines.append(f'    "{sid}" [label="{label}", {style_str}];')
        else:
            lines.append(f'    "{sid}" [label="{label}"];')

    lines.append("")

    # Add edges (dependencies)
    for subtask in subtasks:
        sid = str(get_subtask_id(subtask))
        for dep in subtask.depends_on:
            lines.append(f'    "{dep}" -> "{sid}";')

    lines.append("}")
    return "\n".join(lines)


def to_mermaid(
    subtasks: List["Subtask"],
    title: Optional[str] = None,
) -> str:
    """
    Export DAG to Mermaid flowchart format.

    Args:
        subtasks: List of Subtask objects
        title: Optional title for the diagram

    Returns:
        Mermaid format string (can be rendered in markdown)
    """
    if not subtasks:
        return "```mermaid\nflowchart TD\n    empty[No subtasks]\n```"

    lines = ["```mermaid", "flowchart TD"]

    if title:
        lines.append(f"    %% {title}")

    # Identify root and terminal nodes
    roots = find_root_nodes(subtasks)
    terminals = find_terminal_nodes(subtasks)
    root_ids = {str(get_subtask_id(s)) for s in roots}
    terminal_ids = {str(get_subtask_id(s)) for s in terminals}

    # Add nodes with styling
    for subtask in subtasks:
        sid = str(get_subtask_id(subtask))
        safe_sid = sid.replace("-", "_")  # Mermaid-safe ID

        # Build label
        desc = subtask.description[:30] + "..." if len(subtask.description) > 30 else subtask.description
        label = f"{sid}: {desc}" if desc else sid

        # Node shape based on type
        if sid in root_ids:
            lines.append(f"    {safe_sid}([{label}])")  # Stadium shape for roots
        elif sid in terminal_ids:
            lines.append(f"    {safe_sid}[/{label}/]")  # Parallelogram for terminals
        else:
            lines.append(f"    {safe_sid}[{label}]")  # Rectangle for others

    lines.append("")

    # Add edges
    for subtask in subtasks:
        sid = str(get_subtask_id(subtask))
        safe_sid = sid.replace("-", "_")
        for dep in subtask.depends_on:
            safe_dep = str(dep).replace("-", "_")
            lines.append(f"    {safe_dep} --> {safe_sid}")

    # Styling
    lines.append("")
    if root_ids:
        root_list = " ".join(str(s).replace("-", "_") for s in root_ids)
        lines.append(f"    style {root_list} fill:#90EE90")
    if terminal_ids:
        term_list = " ".join(str(s).replace("-", "_") for s in terminal_ids)
        lines.append(f"    style {term_list} fill:#ADD8E6")

    lines.append("```")
    return "\n".join(lines)


def print_dag(subtasks: List["Subtask"], **kwargs) -> None:
    """Print DAG visualization to stdout."""
    print(visualize_dag(subtasks, **kwargs))


def print_task_dag(task: "GeneratedTask") -> None:
    """Print task DAG visualization to stdout."""
    print(visualize_task_dag(task))


# CLI support
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Visualize EMTOM task DAGs")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--format", choices=["ascii", "dot", "mermaid"], default="ascii",
                        help="Output format (default: ascii)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Load task
    with open(args.task_file) as f:
        task_data = json.load(f)

    # Import here to avoid circular imports
    from .task_generator import GeneratedTask
    task = GeneratedTask.from_dict(task_data)

    # Generate visualization
    if args.format == "ascii":
        output = visualize_task_dag(task)
    elif args.format == "dot":
        output = to_dot(task.subtasks, title=task.title)
    elif args.format == "mermaid":
        output = to_mermaid(task.subtasks, title=task.title)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)
