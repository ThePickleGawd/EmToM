#!/usr/bin/env python3
"""
Generate collaborative challenge tasks from EMTOM Habitat exploration trajectories.

This script:
1. Loads trajectory files from Habitat exploration
2. Analyzes the trajectory for mechanic patterns and surprises
3. Generates collaborative challenge tasks with success/failure conditions

Usage:
    python generate_tasks.py --trajectory-dir data/emtom/trajectories
    python generate_tasks.py --trajectory-file data/emtom/trajectories/trajectory_xyz.json
"""

import argparse
import json
import glob
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from emtom.task_gen import (
    TrajectoryAnalyzer,
    TaskGenerator,
)
from habitat_llm.llm import instantiate_llm


def load_trajectory(trajectory_path: str) -> Dict[str, Any]:
    """Load a trajectory file from Habitat exploration."""
    with open(trajectory_path, 'r') as f:
        data = json.load(f)
    return data


def find_trajectories(trajectory_dir: str) -> List[str]:
    """Find all trajectory files in a directory."""
    patterns = [
        f"{trajectory_dir}/**/trajectory_*.json",
        f"{trajectory_dir}/trajectory_*.json",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    return sorted(set(files))


def print_task(task, index: int):
    """Pretty print a generated task."""
    print(f"\n{'='*60}")
    print(f"TASK {index}: {task.title}")
    print(f"{'='*60}")
    print(f"Category: {task.category.value}")
    print(f"Difficulty: {task.difficulty}/5")
    print(f"Estimated steps: {task.estimated_steps}")
    print(f"Theory of Mind required: {task.theory_of_mind_required}")
    print(f"Communication required: {task.communication_required}")

    print(f"\nDESCRIPTION:")
    print(f"  {task.description}")

    print(f"\nAGENT ROLES ({task.num_agents} agents):")
    for agent_id, role in task.agent_roles.items():
        print(f"  {agent_id}: {role}")

    print(f"\nAGENT STARTING KNOWLEDGE:")
    for agent_id, knowledge in task.agent_knowledge.items():
        if knowledge:
            print(f"  {agent_id}:")
            for k in knowledge:
                print(f"    - {k}")
        else:
            print(f"  {agent_id}: (no special knowledge)")

    print(f"\nSUBTASKS ({len(task.subtasks)}):")
    for subtask in task.subtasks:
        deps = f" [depends on: {', '.join(subtask.depends_on)}]" if subtask.depends_on else ""
        assigned = f" [assigned to: {subtask.assigned_agent}]" if subtask.assigned_agent else ""
        print(f"  - {subtask.subtask_id}: {subtask.description}{deps}{assigned}")
        if subtask.hints:
            for hint in subtask.hints:
                print(f"      Hint: {hint}")

    print(f"\nSUCCESS CONDITION:")
    print(f"  {task.success_condition.description}")
    if task.success_condition.time_limit:
        print(f"  Time limit: {task.success_condition.time_limit} steps")
    print(f"  Required states:")
    for state in task.success_condition.required_states:
        print(f"    - {state}")

    print(f"\nFAILURE CONDITIONS:")
    for fc in task.failure_conditions:
        print(f"  - {fc.description}")
        if fc.max_failed_attempts:
            print(f"    Max failed attempts: {fc.max_failed_attempts}")

    print(f"\nREQUIRED MECHANICS: {task.required_mechanics}")


def generate_tasks_from_trajectory(
    trajectory: Dict[str, Any],
    output_dir: str,
    llm_client: Any,
    num_agents: int = 2,
    max_tasks: int = 5,
    task_type: int = 1,
) -> List[Any]:
    """Generate tasks from a single trajectory using LLM.

    Args:
        task_type: Type of task to generate (1 = Theory of Mind, 2 = Regular)
    """
    print(f"\n[Analyzing] Trajectory: {trajectory.get('episode_id', 'unknown')}")
    print(f"  Scene: {trajectory.get('metadata', {}).get('scene_id', 'unknown')}")
    print(f"  Steps: {trajectory.get('statistics', {}).get('total_steps', 'N/A')}")
    print(f"  Surprises: {trajectory.get('statistics', {}).get('total_surprises', 0)}")

    # Check for surprises
    surprises = trajectory.get("surprise_summary", [])
    if not surprises:
        for step in trajectory.get("steps", []):
            surprises.extend(step.get("surprises", []))

    if not surprises:
        print("  ERROR: No surprises found in trajectory - cannot generate tasks")
        return []

    print(f"  Found {len(surprises)} surprises:")
    for s in surprises:
        print(f"    - {s.get('action', '?')} on {s.get('target', '?')}: {s.get('explanation', '')[:50]}...")

    # Analyze trajectory
    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.analyze(trajectory)

    # Generate tasks using LLM
    task_type_name = "Theory of Mind" if task_type == 1 else "Regular"
    print(f"\n  Generating {task_type_name} tasks with LLM...")
    generator = TaskGenerator(llm_client=llm_client)
    tasks = generator.generate_tasks(
        trajectory=trajectory,
        analysis=analysis,
        num_agents=num_agents,
        max_tasks=max_tasks,
        task_type=task_type,
    )

    print(f"  Generated {len(tasks)} collaborative tasks")

    # Save tasks
    episode_id = trajectory.get('episode_id', 'unknown')
    output_file = Path(output_dir) / f"emtom_challenges_{episode_id}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tasks_data = {
        "source_trajectory": episode_id,
        "scene_id": trajectory.get('scene_id', 'unknown'),
        "discovered_mechanics": [m.mechanic_type for m in analysis.discovered_mechanics],
        "tasks": [task.to_dict() for task in tasks],
    }

    with open(output_file, "w") as f:
        json.dump(tasks_data, f, indent=2)

    print(f"  Saved to: {output_file}")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate EMTOM collaborative challenge tasks from Habitat trajectories"
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        help="Path to a specific trajectory JSON file",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default="data/emtom/trajectories",
        help="Directory to search for trajectory files (default: data/emtom/trajectories)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/emtom/tasks",
        help="Output directory for generated tasks (default: data/emtom/tasks)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents for collaborative tasks (default: 2)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=5,
        help="Maximum tasks to generate per trajectory (default: 5)",
    )
    parser.add_argument(
        "--print-tasks",
        action="store_true",
        help="Print generated tasks to console",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EMTOM Task Generation from Habitat Trajectories")
    print("=" * 60)

    # Initialize LLM client
    print("\nInitializing LLM client...")
    llm_client = instantiate_llm("openai_chat")
    print(f"  Using model: {llm_client.generation_params.model}")

    # Find trajectories
    if args.trajectory_file:
        trajectory_files = [args.trajectory_file]
    else:
        trajectory_files = find_trajectories(args.trajectory_dir)

    if not trajectory_files:
        print(f"\nNo trajectory files found in: {args.trajectory_dir}")
        print("Run exploration first: ./emtom/run_emtom.sh exploration")
        sys.exit(1)

    print(f"\nFound {len(trajectory_files)} trajectory file(s)")
    print(f"\nTask Type: Theory of Mind (ToM tasks only)")

    # Process each trajectory
    all_tasks = []
    for traj_file in trajectory_files:
        try:
            trajectory = load_trajectory(traj_file)
            tasks = generate_tasks_from_trajectory(
                trajectory=trajectory,
                output_dir=args.output_dir,
                llm_client=llm_client,
                num_agents=args.num_agents,
                max_tasks=args.max_tasks,
                task_type=1,  # Always generate Theory of Mind tasks
            )
            all_tasks.extend(tasks)

            if args.print_tasks:
                for i, task in enumerate(tasks, 1):
                    print_task(task, i)

        except Exception as e:
            print(f"\n[Error] Failed to process {traj_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Task Generation Complete")
    print(f"{'='*60}")
    print(f"Total trajectories processed: {len(trajectory_files)}")
    print(f"Total tasks generated: {len(all_tasks)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
