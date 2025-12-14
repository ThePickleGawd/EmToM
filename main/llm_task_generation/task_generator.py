#!/usr/bin/env python3
"""
LLM-based Task Generator Module.

This module uses OpenAI's GPT to generate tasks based on exploration findings,
particularly focusing on surprising observations discovered during scene exploration.

Usage:
    # Generate tasks for a specific mechanics directory:
    python -m main.llm_task_generation.task_generator \
        --exploration_results_dir ./exploration_results \
        --mechanics_dir ./main/mechanics/height_difference \
        --gpt_model gpt-4o-mini

    # Generate tasks for ALL mechanics directories:
    python -m main.llm_task_generation.task_generator \
        --exploration_results_dir ./exploration_results \
        --all_mechanics \
        --gpt_model gpt-4o-mini
"""

import argparse
import json
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Default mechanics base directory
DEFAULT_MECHANICS_BASE_DIR = "./main/mechanics"


class TaskGenerator:
    """
    GPT-based task generator that creates tasks from exploration findings.

    Uses the surprising findings and observations from GPT-guided exploration
    to generate meaningful tasks for robot agents.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the task generator.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

        self.system_prompt = """You are an AI assistant that generates robot tasks based on exploration findings.

Given observations from a robot exploring a household environment, you will create specific,
actionable tasks that the robot could perform. Your PRIMARY focus should be on SURPRISING FINDINGS -
these are unexpected observations, failed interactions, or unusual situations discovered during exploration.

CRITICAL: Tasks MUST be directly inspired by and address the surprising findings. Each task should:
1. Directly relate to a specific surprising finding from the exploration
2. Challenge the robot to overcome or investigate the surprising situation
3. Test the robot's ability to handle unexpected scenarios

Secondary considerations:
- Leverage the objects and furniture discovered
- Include meaningful interactions (pick, place, open, close, navigate)
- Create tasks of varying difficulty

Each task should have:
- A clear, specific goal tied to a surprising finding
- Required objects/furniture from the exploration
- Expected actions to complete the task
- A difficulty level (easy, medium, hard)
- A category (navigation, manipulation, multi-step, problem-solving)
- Clear reference to which surprising finding inspired the task

Format your response as a JSON array of task objects."""

    def _chat(self, user_message: str) -> str:
        """Send a message to GPT and get a response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content

    def generate_tasks_from_exploration(
        self,
        exploration_data: Dict[str, Any],
        num_tasks: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate tasks based on exploration data.

        Args:
            exploration_data: Parsed JSON from exploration results
            num_tasks: Number of tasks to generate

        Returns:
            List of generated task dictionaries
        """
        # Extract relevant information from exploration data
        rooms_info = []
        all_surprising_findings = []
        all_objects = []
        all_furniture = []

        for room in exploration_data.get("rooms_visited", []):
            room_name = room.get("room_name", "unknown")
            observations = room.get("gpt_observations", "")
            hypotheses = room.get("gpt_hypotheses", [])
            surprising = room.get("surprising_findings", [])
            objects = room.get("objects_observed", [])
            furniture = room.get("furniture_observed", [])
            interactions = room.get("interactions", [])

            rooms_info.append({
                "room": room_name,
                "observations": observations,
                "hypotheses": hypotheses,
                "objects": objects,
                "furniture": furniture,
                "interactions": interactions
            })

            all_surprising_findings.extend(surprising)
            all_objects.extend(objects)
            all_furniture.extend(furniture)

        # Build the prompt
        prompt = f"""Based on the following exploration of a household environment, generate {num_tasks} specific tasks for a robot to perform.

=== EXPLORATION SUMMARY ===
Total rooms explored: {len(rooms_info)}
Total surprising findings: {len(all_surprising_findings)}

=== ROOMS EXPLORED ===
"""
        for room_info in rooms_info:
            prompt += f"""
Room: {room_info['room']}
- Observations: {room_info['observations']}
- Objects found: {room_info['objects']}
- Furniture: {room_info['furniture']}
- Hypotheses: {room_info['hypotheses']}
"""

        if all_surprising_findings:
            prompt += f"""
=== SURPRISING FINDINGS ===
These are unexpected observations that may inspire interesting tasks:
"""
            for i, finding in enumerate(all_surprising_findings, 1):
                prompt += f"{i}. {finding}\n"

        prompt += f"""
=== INTERACTIONS PERFORMED ===
"""
        for room_info in rooms_info:
            for interaction in room_info['interactions']:
                prompt += f"- {interaction['action']} on {interaction['target']}: {interaction['result']}\n"
                if interaction.get('surprising'):
                    prompt += f"  (SURPRISING: {interaction.get('finding', 'N/A')})\n"

        prompt += """
=== TASK GENERATION REQUEST ===
Generate tasks that:
1. Address the surprising findings (objects out of reach, unexpected placements, etc.)
2. Involve the objects and furniture discovered
3. Include a mix of difficulties and categories
4. Are specific and actionable

Respond with a JSON array in this exact format:
[
  {
    "task_id": "task_001",
    "name": "Short descriptive name",
    "description": "Detailed description of what the robot should do",
    "goal": "The specific goal to achieve",
    "required_objects": ["object1", "object2"],
    "required_furniture": ["furniture1"],
    "actions": ["Navigate to X", "Pick Y", "Place Y on Z"],
    "difficulty": "easy|medium|hard",
    "category": "navigation|manipulation|multi-step|problem-solving",
    "inspired_by": "What exploration finding inspired this task",
    "room": "Which room this task takes place in"
  }
]
"""

        response = self._chat(prompt)

        # Parse the JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            tasks = json.loads(json_str.strip())
            return tasks
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse GPT response as JSON: {e}")
            print(f"Raw response:\n{response}")
            return []


def get_most_recent_exploration_file(exploration_dir: str) -> Optional[str]:
    """
    Find the most recently created exploration JSON file.

    Args:
        exploration_dir: Directory containing exploration result JSON files

    Returns:
        Path to the most recent file, or None if no files found
    """
    pattern = os.path.join(exploration_dir, "*.json")
    files = glob(pattern)

    if not files:
        return None

    # Sort by modification time, most recent first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def discover_mechanics_directories(base_dir: str) -> List[str]:
    """
    Discover all mechanics subdirectories.

    Args:
        base_dir: Base directory containing mechanics subdirectories
                  (e.g., ./main/mechanics)

    Returns:
        List of paths to mechanics directories
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Mechanics base directory does not exist: {base_dir}")
        return []

    mechanics_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('__'):
            mechanics_dirs.append(str(item))

    return sorted(mechanics_dirs)


def ensure_tasks_directory(mechanics_dir: str) -> str:
    """
    Ensure the 'tasks' subdirectory exists within a mechanics directory.

    Args:
        mechanics_dir: Path to the mechanics directory

    Returns:
        Path to the tasks directory
    """
    tasks_dir = os.path.join(mechanics_dir, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir


def generate_tasks_for_mechanics(
    generator: TaskGenerator,
    exploration_data: Dict[str, Any],
    mechanics_dir: str,
    exploration_file: str,
    num_tasks: int = 5
) -> Tuple[str, int]:
    """
    Generate tasks for a specific mechanics directory.

    Args:
        generator: TaskGenerator instance
        exploration_data: Parsed exploration JSON data
        mechanics_dir: Path to the mechanics directory
        exploration_file: Path to source exploration file
        num_tasks: Number of tasks to generate

    Returns:
        Tuple of (output_file_path, num_tasks_generated)
    """
    # Get mechanics name for context
    mechanics_name = os.path.basename(mechanics_dir)
    print(f"\n{'='*60}")
    print(f"Generating tasks for: {mechanics_name}")
    print(f"{'='*60}")

    # Ensure tasks directory exists
    tasks_dir = ensure_tasks_directory(mechanics_dir)
    print(f"Tasks directory: {tasks_dir}")

    # Generate tasks
    tasks = generator.generate_tasks_from_exploration(
        exploration_data,
        num_tasks=num_tasks
    )

    if not tasks:
        print(f"Warning: No tasks generated for {mechanics_name}")
        return "", 0

    print(f"Generated {len(tasks)} tasks")

    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(tasks_dir, f"generated_tasks_{timestamp}.json")

    # Build output structure
    output_data = {
        "generation_time": datetime.now().isoformat(),
        "source_exploration_file": exploration_file,
        "mechanics_directory": mechanics_dir,
        "mechanics_name": mechanics_name,
        "gpt_model": generator.model,
        "num_tasks_requested": num_tasks,
        "num_tasks_generated": len(tasks),
        "tasks": tasks
    }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Tasks saved to: {output_file}")

    # Print summary of generated tasks
    print(f"\n--- Generated Tasks for {mechanics_name} ---")
    for task in tasks:
        print(f"  {task.get('task_id', 'N/A')}: {task.get('name', 'Unnamed')}")
        print(f"    Difficulty: {task.get('difficulty', 'N/A')} | Category: {task.get('category', 'N/A')}")
        print(f"    Inspired by: {task.get('inspired_by', 'N/A')[:80]}...")

    return output_file, len(tasks)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tasks from exploration findings using GPT"
    )
    parser.add_argument(
        "--exploration_results_dir",
        type=str,
        default="./exploration_results",
        help="Directory containing exploration result JSON files"
    )
    parser.add_argument(
        "--mechanics_dir",
        type=str,
        default=None,
        help="Specific mechanics directory to generate tasks for (e.g., ./main/mechanics/height_difference)"
    )
    parser.add_argument(
        "--mechanics_base_dir",
        type=str,
        default=DEFAULT_MECHANICS_BASE_DIR,
        help=f"Base directory containing mechanics subdirectories (default: {DEFAULT_MECHANICS_BASE_DIR})"
    )
    parser.add_argument(
        "--all_mechanics",
        action="store_true",
        help="Generate tasks for ALL mechanics directories found in mechanics_base_dir"
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-4o-mini",
        help="GPT model to use for task generation (e.g., gpt-4o-mini, gpt-5, gpt-5-mini)"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=5,
        help="Number of tasks to generate per mechanics directory"
    )
    parser.add_argument(
        "--exploration_file",
        type=str,
        default=None,
        help="Specific exploration file to use (overrides auto-detection of most recent)"
    )

    args = parser.parse_args()

    # Find the exploration file (most recent by default)
    if args.exploration_file:
        exploration_file = args.exploration_file
    else:
        exploration_file = get_most_recent_exploration_file(args.exploration_results_dir)

    if not exploration_file:
        print(f"Error: No exploration JSON files found in {args.exploration_results_dir}")
        return 1

    print(f"Using exploration file: {exploration_file}")

    # Load exploration data
    with open(exploration_file, 'r') as f:
        exploration_data = json.load(f)

    # Print exploration summary
    print(f"\n{'='*60}")
    print("EXPLORATION DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Rooms visited: {len(exploration_data.get('rooms_visited', []))}")
    print(f"  Total interactions: {exploration_data.get('total_interactions', 0)}")
    print(f"  Surprising findings: {exploration_data.get('total_surprising_findings', 0)}")

    # Print surprising findings for context
    all_surprises = []
    for room in exploration_data.get("rooms_visited", []):
        for finding in room.get("surprising_findings", []):
            all_surprises.append(f"[{room.get('room_name', 'unknown')}] {finding}")
        for interaction in room.get("interactions", []):
            if interaction.get("surprising"):
                all_surprises.append(
                    f"[{room.get('room_name', 'unknown')}] "
                    f"{interaction.get('action', 'Unknown')} on {interaction.get('target', 'unknown')}: "
                    f"{interaction.get('finding', 'No details')}"
                )

    if all_surprises:
        print(f"\n  Surprising findings found:")
        for i, surprise in enumerate(all_surprises, 1):
            print(f"    {i}. {surprise[:100]}...")
    else:
        print("\n  Warning: No surprising findings found in exploration data!")
        print("  Tasks will be generated based on general observations instead.")

    # Determine which mechanics directories to process
    mechanics_dirs = []

    if args.all_mechanics:
        # Discover all mechanics directories
        mechanics_dirs = discover_mechanics_directories(args.mechanics_base_dir)
        if not mechanics_dirs:
            print(f"Error: No mechanics directories found in {args.mechanics_base_dir}")
            return 1
        print(f"\nDiscovered {len(mechanics_dirs)} mechanics directories:")
        for d in mechanics_dirs:
            print(f"  - {d}")
    elif args.mechanics_dir:
        # Use specified mechanics directory
        if not os.path.isdir(args.mechanics_dir):
            print(f"Error: Mechanics directory does not exist: {args.mechanics_dir}")
            return 1
        mechanics_dirs = [args.mechanics_dir]
    else:
        # Default: use all mechanics directories
        mechanics_dirs = discover_mechanics_directories(args.mechanics_base_dir)
        if not mechanics_dirs:
            print(f"Error: No mechanics directories found in {args.mechanics_base_dir}")
            print("Use --mechanics_dir to specify a specific directory")
            return 1
        print(f"\nUsing all {len(mechanics_dirs)} discovered mechanics directories:")
        for d in mechanics_dirs:
            print(f"  - {d}")

    # Initialize task generator
    print(f"\nInitializing task generator with model: {args.gpt_model}")
    generator = TaskGenerator(model=args.gpt_model)

    # Generate tasks for each mechanics directory
    total_tasks = 0
    results = []

    for mechanics_dir in mechanics_dirs:
        try:
            output_file, num_generated = generate_tasks_for_mechanics(
                generator=generator,
                exploration_data=exploration_data,
                mechanics_dir=mechanics_dir,
                exploration_file=exploration_file,
                num_tasks=args.num_tasks
            )
            if output_file:
                results.append({
                    "mechanics_dir": mechanics_dir,
                    "output_file": output_file,
                    "tasks_generated": num_generated
                })
                total_tasks += num_generated
        except Exception as e:
            print(f"Error generating tasks for {mechanics_dir}: {e}")
            continue

    # Print final summary
    print(f"\n{'='*60}")
    print("TASK GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total mechanics directories processed: {len(results)}")
    print(f"Total tasks generated: {total_tasks}")
    print("\nOutput files:")
    for result in results:
        print(f"  - {result['output_file']} ({result['tasks_generated']} tasks)")

    return 0 if results else 1


if __name__ == "__main__":
    exit(main())
