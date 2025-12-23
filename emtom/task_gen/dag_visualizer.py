"""
DAG Visualization for EMTOM subtask graphs.

Usage:
    python -m emtom.task_gen.dag_visualizer path/to/task.json
    python -m emtom.task_gen.dag_visualizer path/to/task.json -o output.png

    # Or in code:
    from emtom.task_gen.dag_visualizer import view_task_dag
    view_task_dag(task)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from graphviz import Digraph

if TYPE_CHECKING:
    from .task_generator import GeneratedTask, Subtask

from .dag import find_root_nodes, find_terminal_nodes, get_subtask_id


def _build_dag_graph(
    subtasks: List["Subtask"],
    title: Optional[str] = None,
) -> Digraph:
    """Build a Digraph object from subtasks."""
    dot = Digraph(comment=title or "Task DAG")
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("node", shape="box", style="rounded")

    if title:
        dot.attr(label=title, labelloc="t", fontsize="16")

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

        label = "\n".join(label_parts)

        # Style based on node type
        if sid in root_ids:
            dot.node(sid, label, fillcolor="lightgreen", style="rounded,filled")
        elif sid in terminal_ids:
            dot.node(sid, label, fillcolor="lightblue", style="rounded,filled")
        else:
            dot.node(sid, label)

    # Add edges (dependencies)
    for subtask in subtasks:
        sid = str(get_subtask_id(subtask))
        for dep in subtask.depends_on:
            dot.edge(str(dep), sid)

    return dot


def view_dag(
    subtasks: List["Subtask"],
    title: Optional[str] = None,
    output: Optional[str] = None,
) -> str:
    """
    Render and display a subtask DAG.

    Args:
        subtasks: List of Subtask objects
        title: Optional title for the graph
        output: Optional output path (e.g., "dag.png"). If None, opens in viewer.

    Returns:
        Path to the rendered file
    """
    if not subtasks:
        print("Empty DAG (no subtasks)")
        return ""

    dot = _build_dag_graph(subtasks, title)

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        import tempfile
        out_path = Path(tempfile.gettempdir()) / "task_dag.png"

    # Render to file
    fmt = out_path.suffix.lstrip(".") or "png"
    parent = str(out_path.parent) if out_path.parent != Path(".") else "."
    dot.render(out_path.stem, directory=parent, format=fmt, cleanup=True)
    result_path = str(out_path.with_suffix(f".{fmt}"))
    print(f"DAG saved to: {result_path}")
    return result_path


def view_task_dag(task: "GeneratedTask", output: Optional[str] = None) -> str:
    """
    Render and display a GeneratedTask's DAG.

    Args:
        task: GeneratedTask object
        output: Optional output path (e.g., "dag.png"). If None, opens in viewer.

    Returns:
        Path to the rendered file
    """
    return view_dag(task.subtasks, title=task.title, output=output)


# CLI
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Visualize EMTOM task DAG")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("-o", "--output", help="Output file path (e.g., dag.png)")
    args = parser.parse_args()

    with open(args.task_file) as f:
        task_data = json.load(f)

    from .task_generator import GeneratedTask

    task = GeneratedTask.from_dict(task_data)
    view_task_dag(task, output=args.output)
