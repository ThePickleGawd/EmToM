#!/usr/bin/env python3
"""Analyze task data and generate charts."""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

TASKS_DIR = Path("data/emtom/tasks")

def load_task_data():
    """Load all task JSON files and extract relevant data."""
    tasks = []
    for json_file in TASKS_DIR.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            # Extract calibration results (array format, each entry is a benchmark run)
            calibration = data.get("calibration", [])
            if isinstance(calibration, dict):
                # Legacy dict format — convert on the fly
                calibration = [
                    {**v, "_legacy_key": k} for k, v in calibration.items() if isinstance(v, dict)
                ]
            for entry in calibration:
                # Derive a model label from agent_models (e.g. "gpt-5.2" or "gpt-5.2_vs_sonnet")
                agent_models = entry.get("agent_models", {})
                unique_models = sorted(set(agent_models.values())) if agent_models else []
                if len(unique_models) == 1:
                    model_label = unique_models[0]
                elif len(unique_models) > 1:
                    model_label = "_vs_".join(unique_models)
                else:
                    model_label = entry.get("_legacy_key", "unknown")

                tasks.append({
                    "task_id": data.get("task_id"),
                    "num_agents": data.get("num_agents"),
                    "num_subtasks": len(data.get("subtasks", [])),
                    "model": model_label,
                    "passed": entry.get("passed", False),
                    "percent_complete": entry.get("percent_complete", 0),
                })
    return tasks

def plot_success_vs_subtasks(tasks):
    """Chart 1: Success rate vs number of subtasks (nodes)."""
    subtask_counts = defaultdict(lambda: {"passed": 0, "total": 0})
    for task in tasks:
        n = task["num_subtasks"]
        subtask_counts[n]["total"] += 1
        if task["passed"]:
            subtask_counts[n]["passed"] += 1

    sorted_keys = sorted(subtask_counts.keys())
    x = sorted_keys
    success_rates = [subtask_counts[k]["passed"] / subtask_counts[k]["total"] * 100
                    for k in sorted_keys]
    totals = [subtask_counts[k]["total"] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, success_rates, color='steelblue', edgecolor='black')

    # Add count labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.annotate(f'n={total}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Number of Subtasks', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate vs Number of Subtasks', fontsize=14)
    ax.set_xticks(x)
    ax.set_ylim(0, max(success_rates) * 1.3 if max(success_rates) > 0 else 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('chart_success_vs_subtasks.png', dpi=150)
    plt.close()
    print(f"Saved: chart_success_vs_subtasks.png")

def plot_success_vs_agents(tasks):
    """Chart 2: Success rate vs number of agents."""
    agent_counts = defaultdict(lambda: {"passed": 0, "total": 0})
    for task in tasks:
        n = task["num_agents"]
        agent_counts[n]["total"] += 1
        if task["passed"]:
            agent_counts[n]["passed"] += 1

    sorted_keys = sorted(agent_counts.keys())
    x = sorted_keys
    success_rates = [agent_counts[k]["passed"] / agent_counts[k]["total"] * 100
                    for k in sorted_keys]
    totals = [agent_counts[k]["total"] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, success_rates, color='coral', edgecolor='black')

    # Add count labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.annotate(f'n={total}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate vs Number of Agents', fontsize=14)
    ax.set_xticks(x)
    ax.set_ylim(0, max(success_rates) * 1.3 if max(success_rates) > 0 else 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('chart_success_vs_agents.png', dpi=150)
    plt.close()
    print(f"Saved: chart_success_vs_agents.png")

def plot_subtasks_histogram(tasks):
    """Chart 3: Histogram of number of subtasks distribution."""
    # Use unique tasks (not per-model)
    unique_tasks = {}
    for task in tasks:
        tid = task["task_id"]
        if tid not in unique_tasks:
            unique_tasks[tid] = task["num_subtasks"]

    subtask_counts = list(unique_tasks.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = range(min(subtask_counts), max(subtask_counts) + 2)
    counts, bins, patches = ax.hist(subtask_counts, bins=bins, color='seagreen',
                                     edgecolor='black', align='left')

    # Add count labels on bars
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.annotate(f'{int(count)}',
                        xy=(patch.get_x() + patch.get_width() / 2, count),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Number of Subtasks', fontsize=12)
    ax.set_ylabel('Count (number of tasks)', fontsize=12)
    ax.set_title('Distribution of Number of Subtasks per Task', fontsize=14)
    ax.set_xticks(range(min(subtask_counts), max(subtask_counts) + 1))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('chart_subtasks_histogram.png', dpi=150)
    plt.close()
    print(f"Saved: chart_subtasks_histogram.png")

def main():
    print("Loading task data...")
    tasks = load_task_data()
    print(f"Loaded {len(tasks)} task calibration records")

    # Summary stats
    passed = sum(1 for t in tasks if t["passed"])
    print(f"Overall: {passed}/{len(tasks)} passed ({passed/len(tasks)*100:.1f}%)")

    print("\nGenerating charts...")
    plot_success_vs_subtasks(tasks)
    plot_success_vs_agents(tasks)
    plot_subtasks_histogram(tasks)

    print(f"\nCharts saved to: project root")

if __name__ == "__main__":
    main()
