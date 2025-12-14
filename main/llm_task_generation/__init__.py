"""
LLM Task Generation module.

This module provides tools to generate robot tasks based on exploration findings
using large language models (GPT).

The task generator reads exploration data (JSON files from GPT-guided exploration)
and uses GPT to generate actionable tasks inspired by surprising findings.

Usage:
    # Generate tasks for all mechanics directories:
    python -m main.llm_task_generation.task_generator --all_mechanics

    # Generate tasks for a specific mechanics directory:
    python -m main.llm_task_generation.task_generator \
        --mechanics_dir ./main/mechanics/height_difference

    # Use a specific exploration file:
    python -m main.llm_task_generation.task_generator \
        --exploration_file ./exploration_results/my_exploration.json
"""

from main.llm_task_generation.task_generator import (
    TaskGenerator,
    get_most_recent_exploration_file,
    discover_mechanics_directories,
    ensure_tasks_directory,
    generate_tasks_for_mechanics,
)

__all__ = [
    "TaskGenerator",
    "get_most_recent_exploration_file",
    "discover_mechanics_directories",
    "ensure_tasks_directory",
    "generate_tasks_for_mechanics",
]
