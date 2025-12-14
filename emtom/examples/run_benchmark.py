#!/usr/bin/env python3
"""
Example script for running the EMTOM benchmark.

Demonstrates:
1. Loading tasks from JSON
2. Creating agents (scripted or LLM-based)
3. Running tasks with the TaskRunner
4. Evaluating results with BenchmarkEvaluator
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from emtom.benchmark import (
    TaskRunner,
    TaskResult,
    RunConfig,
    ScriptedAgent,
    BenchmarkEvaluator,
    BenchmarkResults,
)
from emtom.task_gen import GeneratedTask

def load_tasks(task_file: str) -> list:
    """Load tasks from JSON file."""
    with open(task_file) as f:
        data = json.load(f)

    tasks = []
    for task_data in data.get("tasks", []):
        task = GeneratedTask.from_dict(task_data)
        tasks.append(task)

    return tasks


def create_scripted_agents_for_task(task: GeneratedTask) -> dict:
    """Create scripted agents with predefined actions for testing."""
    agents = {}

    # Generate simple scripts based on task type
    for agent_id in task.agent_roles.keys():
        knowledge = task.agent_knowledge.get(agent_id, [])

        # Create a basic script based on what the agent knows
        script = []

        if "inverse" in str(task.required_mechanics).lower():
            # For inverse mechanics, try close to open
            if "Doors in this house work backwards" in knowledge:
                # This agent knows the trick
                script = [
                    {"action": "close", "target": "door_1"},
                    {"action": "communicate", "target": "agent_1" if agent_id == "agent_0" else "agent_0",
                     "message": "The doors work backwards! Use 'close' to open them."},
                    {"action": "close", "target": "door_3"},
                ]
            else:
                # This agent needs to learn
                script = [
                    {"action": "open", "target": "door_2"},  # Will fail
                    {"action": "wait"},  # Wait for message
                    {"action": "close", "target": "door_2"},  # Now try correct way
                ]

        elif "counting" in str(task.required_mechanics).lower():
            # For counting trigger, alternate pressing
            if agent_id == "agent_0":
                script = [
                    {"action": "move", "target": "vault_room"},
                    {"action": "press", "target": "security_button"},
                    {"action": "move", "target": "waiting_room"},
                    {"action": "communicate", "target": "agent_1", "message": "Your turn to press!"},
                ]
            else:
                script = [
                    {"action": "wait"},
                    {"action": "wait"},
                    {"action": "move", "target": "vault_room"},
                    {"action": "press", "target": "security_button"},
                    {"action": "press", "target": "security_button"},
                ]
        else:
            # Default script
            script = [
                {"action": "look"},
                {"action": "wait"},
                {"action": "wait"},
            ]

        agents[agent_id] = ScriptedAgent(agent_id, script)

    return agents


def run_single_task(task: GeneratedTask, config: RunConfig, verbose: bool = True) -> TaskResult:
    """Run a single task and return results."""
    if verbose:
        print(f"\nSetting up task: {task.title}")
        print(f"  Category: {task.category.value}")
        print(f"  Difficulty: {task.difficulty}")
        print(f"  Required mechanics: {task.required_mechanics}")

    # Create agents
    agents = create_scripted_agents_for_task(task)

    # Create runner and setup
    runner = TaskRunner(config)
    runner.setup_task(task, agents)

    # Run the task
    result = runner.run()

    return result


def run_benchmark(
    task_file: str,
    num_runs: int = 1,
    verbose: bool = True,
    output_file: str = None,
) -> BenchmarkResults:
    """
    Run the full EMTOM benchmark.

    Args:
        task_file: Path to JSON file containing tasks
        num_runs: Number of times to run each task
        verbose: Whether to print progress
        output_file: Optional path to save results

    Returns:
        BenchmarkResults with aggregated metrics
    """
    # Load tasks
    tasks = load_tasks(task_file)
    print(f"Loaded {len(tasks)} tasks from {task_file}")

    # Create evaluator
    evaluator = BenchmarkEvaluator(agent_model="scripted-test")

    # Register task info
    for task in tasks:
        evaluator.add_task_info(
            task_id=task.task_id,
            title=task.title,
            difficulty=task.difficulty,
            category=task.category.value,
            theory_of_mind_required=task.theory_of_mind_required,
            communication_required=task.communication_required,
            num_subtasks=len(task.subtasks),
        )

    # Run config
    config = RunConfig(
        max_steps=50,
        verbose=verbose,
        allow_communication=True,
    )

    # Run each task
    for run_idx in range(num_runs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"BENCHMARK RUN {run_idx + 1}/{num_runs}")
            print(f"{'='*60}")

        for task in tasks:
            result = run_single_task(task, config, verbose=verbose)
            evaluator.add_result(result)

            if verbose:
                status_str = result.status.value.upper()
                if result.status.value == "success":
                    status_str = f"\033[92m{status_str}\033[0m"  # Green
                elif result.status.value == "failure":
                    status_str = f"\033[91m{status_str}\033[0m"  # Red
                print(f"\nResult: {status_str}")
                if result.failure_reason:
                    print(f"Reason: {result.failure_reason}")

    # Evaluate results
    results = evaluator.evaluate()

    # Print summary
    evaluator.print_summary(results)

    # Save if requested
    if output_file:
        results.save(output_file)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run EMTOM benchmark")
    parser.add_argument(
        "--tasks",
        default="data/tasks/emtom_challenges_20251212_191827.json",
        help="Path to tasks JSON file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per task",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    # Check if task file exists
    task_path = Path(args.tasks)
    if not task_path.exists():
        # Try relative to project root
        task_path = project_root / args.tasks
        if not task_path.exists():
            print(f"Error: Task file not found: {args.tasks}")
            sys.exit(1)

    # Default output file
    output = args.output
    if output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"data/benchmark_results/emtom_results_{timestamp}.json"

    run_benchmark(
        task_file=str(task_path),
        num_runs=args.runs,
        verbose=not args.quiet,
        output_file=output,
    )


if __name__ == "__main__":
    main()
