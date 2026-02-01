#!/usr/bin/env python3
"""
Categorize tasks from the emtom task dataset.

Reads all tasks from data/emtom/tasks, groups them by category (cooperative,
competitive, mixed), and uses GPT-5.2 to generate descriptive subcategories
for each task type.

Usage:
    python emtom/scripts/categorize_tasks.py [--output OUTPUT_PATH]
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

# Load .env file if it exists
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_tasks(tasks_dir: Path) -> dict[str, list[dict]]:
    """Load all tasks and group by category."""
    tasks_by_category = defaultdict(list)

    for task_file in sorted(tasks_dir.glob("*.json")):
        with open(task_file) as f:
            task_data = json.load(f)

        category = task_data.get("category", "unknown")
        tasks_by_category[category].append({
            "filename": task_file.name,
            "task_id": task_data.get("task_id", ""),
            "title": task_data.get("title", ""),
            "task": task_data.get("task", ""),
            "num_agents": task_data.get("num_agents", 0),
        })

    return dict(tasks_by_category)


def categorize_with_gpt(client: OpenAI, category: str, tasks: list[dict]) -> dict:
    """Use GPT-5.2 to categorize tasks into subcategories."""

    # Build task list for the prompt
    task_descriptions = []
    for i, task in enumerate(tasks, 1):
        task_descriptions.append(f"{i}. {task['title']}\n   Task: {task['task']}")

    tasks_text = "\n\n".join(task_descriptions)

    prompt = f"""Analyze these {category.upper()} tasks and create meaningful subcategories.

TASKS:
{tasks_text}

Create 3-5 subcategories that group these tasks by their TASK DESCRIPTION.
For each subcategory:
1. Give it a short, descriptive name (someting small from 1-3 words)
2. Write a brief description (1-2 sentences) explaining what tasks in this category have in common

Respond in this exact JSON format:
{{
    "subcategories": [
        {{
            "name": "Subcategory Name",
            "description": "Brief description of what makes these tasks similar."
        }}
    ]
}}

Only output valid JSON, nothing else."""

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are an expert at analyzing game tasks and identifying patterns. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    response_text = response.choices[0].message.content.strip()

    # Parse JSON from response (handle potential markdown code blocks)
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    return json.loads(response_text)


def build_output(tasks_by_category: dict[str, list[dict]], client: OpenAI) -> dict:
    """Build the final categorized output structure."""
    output = {}

    for category, tasks in tasks_by_category.items():
        print(f"Processing {category} ({len(tasks)} tasks)...")

        if len(tasks) == 0:
            continue

        # Get subcategories from GPT
        gpt_result = categorize_with_gpt(client, category, tasks)

        # Build category output
        category_output = {}
        for subcat in gpt_result.get("subcategories", []):
            name = subcat["name"]
            description = subcat["description"]
            task_indices = subcat.get("task_indices", [])

            # Get actual task IDs for this subcategory
            task_ids = [tasks[i-1]["task_id"] for i in task_indices if 0 < i <= len(tasks)]

            category_output[name] = {
                "description": description,
                "task_count": len(task_ids),
                "task_ids": task_ids
            }

        output[category.capitalize()] = category_output

    return output


def main():
    parser = argparse.ArgumentParser(description="Categorize emtom tasks using GPT-5.2")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "emtom" / "tasks",
        help="Directory containing task JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "emtom" / "task_categories.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # Validate tasks directory
    if not args.tasks_dir.exists():
        print(f"Error: Tasks directory not found: {args.tasks_dir}")
        return 1

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1

    client = OpenAI(api_key=api_key)

    # Load tasks
    print(f"Loading tasks from {args.tasks_dir}...")
    tasks_by_category = load_tasks(args.tasks_dir)

    total_tasks = sum(len(tasks) for tasks in tasks_by_category.values())
    print(f"Found {total_tasks} tasks in {len(tasks_by_category)} categories")

    for category, tasks in tasks_by_category.items():
        print(f"  - {category}: {len(tasks)} tasks")

    # Build categorized output
    print("\nAnalyzing tasks with GPT-5.2...")
    output = build_output(tasks_by_category, client)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved categorized tasks to {args.output}")

    # Print summary
    print("\n=== Category Summary ===")
    for category, subcats in output.items():
        print(f"\n{category}:")
        for name, info in subcats.items():
            print(f"  - {name}: {info['description']}")

    return 0


if __name__ == "__main__":
    exit(main())
