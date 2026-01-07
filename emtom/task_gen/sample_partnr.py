#!/usr/bin/env python3
"""
Sample PARTNR tasks to inspire EMTOM task generation.

Extracts diverse tasks from PARTNR dataset and saves them
in a readable format for inspiration.

Usage:
    python emtom/task_gen/sample_partnr.py [--num 50] [--output /tmp/partnr_samples]
"""

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_partnr_episodes(data_path: Path) -> List[Dict[str, Any]]:
    """Load episodes from gzipped PARTNR dataset."""
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('episodes', [])


def extract_task_summary(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant task info for inspiration."""
    # Get evaluation propositions in readable format
    eval_props = []
    for prop in episode.get('evaluation_propositions', []):
        func = prop.get('function_name', 'unknown')
        args = prop.get('args', {})
        # Simplify object handles to just the name
        obj_handles = args.get('object_handles', [])
        obj_names = [h.split('_:')[0].split('/')[-1] for h in obj_handles]
        recep_handles = args.get('receptacle_handles', [])
        recep_names = [h.split('_:')[0].split('/')[-1] for h in recep_handles]

        eval_props.append({
            'function': func,
            'objects': obj_names,
            'receptacles': recep_names,
            'number': args.get('number', 1),
        })

    # Get initial state info
    initial_state = episode.get('info', {}).get('initial_state', [])
    initial_objects = []
    for item in initial_state:
        if 'object_classes' in item:
            initial_objects.append({
                'classes': item.get('object_classes', []),
                'furniture': item.get('furniture_names', []),
                'regions': item.get('allowed_regions', []),
                'count': item.get('number', 1),
            })

    # Get object states if any
    object_states = episode.get('object_states', {})

    # Get articulated object states (doors, drawers)
    ao_states = episode.get('ao_states', {})

    return {
        'episode_id': episode.get('episode_id'),
        'scene_id': episode.get('scene_id'),
        'instruction': episode.get('instruction'),
        'task_type': episode.get('info', {}).get('task_gen', 'unknown'),
        'evaluation_propositions': eval_props,
        'initial_objects': initial_objects,
        'object_states': object_states if object_states else None,
        'ao_states': ao_states if ao_states else None,
        'num_rigid_objects': len(episode.get('rigid_objs', [])),
    }


def categorize_task(instruction: str) -> str:
    """Categorize task by type based on instruction."""
    instruction_lower = instruction.lower()

    if 'clean' in instruction_lower or 'tidy' in instruction_lower:
        return 'cleaning'
    elif 'set up' in instruction_lower or 'prepare' in instruction_lower:
        return 'setup'
    elif 'move' in instruction_lower and 'to' in instruction_lower:
        return 'rearrange'
    elif 'bring' in instruction_lower or 'take' in instruction_lower:
        return 'transport'
    elif 'put' in instruction_lower or 'place' in instruction_lower:
        return 'placement'
    elif 'swap' in instruction_lower:
        return 'swap'
    elif 'organize' in instruction_lower:
        return 'organize'
    else:
        return 'other'


def sample_diverse_tasks(
    episodes: List[Dict[str, Any]],
    num_samples: int = 50,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Sample diverse tasks across categories."""
    if seed is not None:
        random.seed(seed)

    # Categorize all tasks
    by_category: Dict[str, List[dict]] = {}
    for ep in episodes:
        instruction = ep.get('instruction', '')
        category = categorize_task(instruction)
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(ep)


    # Sample evenly from categories
    samples = []
    per_category = max(1, num_samples // len(by_category))

    for category, tasks in by_category.items():
        n = min(per_category, len(tasks))
        selected = random.sample(tasks, n)
        for ep in selected:
            summary = extract_task_summary(ep)
            summary['category'] = category
            samples.append(summary)

    # If we need more, sample randomly from remainder
    while len(samples) < num_samples:
        ep = random.choice(episodes)
        summary = extract_task_summary(ep)
        summary['category'] = categorize_task(ep.get('instruction', ''))
        if summary not in samples:
            samples.append(summary)

    return samples[:num_samples]


def sample_planning_tasks_to_directory(
    output_dir: Path,
    num_samples: int = 10,
    seed: Optional[int] = None,
    dataset: str = 'train',
    verbose: bool = True,
) -> bool:
    """
    Sample simple planning tasks for task generator inspiration.

    These are single-agent rearrangement tasks (no Theory of Mind).
    Useful for learning task structure patterns, success conditions,
    and how to phrase instructions.

    Args:
        output_dir: Directory to save sampled tasks
        num_samples: Number of tasks to sample
        seed: Random seed for reproducibility
        dataset: Which dataset to sample from ('train', 'val', etc.)
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    # Find dataset
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'versioned_data' / 'partnr_episodes' / 'v0_0' / f'{dataset}.json.gz'

    if not data_path.exists():
        if verbose:
            print(f"[sample_tasks] Dataset not found: {data_path}")
        return False

    try:
        if verbose:
            print(f"[sample_tasks] Loading planning task dataset...")
        episodes = load_partnr_episodes(data_path)

        if verbose:
            print(f"[sample_tasks] Sampling {num_samples} tasks...")
        samples = sample_diverse_tasks(episodes, num_samples, seed)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save readable text version (main file agents read)
        text_file = output_dir / 'planning_examples.txt'
        with open(text_file, 'w') as f:
            f.write("Simple Planning Task Examples\n")
            f.write("=" * 60 + "\n")
            f.write("These are single-agent rearrangement tasks (NO Theory of Mind).\n")
            f.write("Use for inspiration on: task phrasing, success conditions, goal types.\n")
            f.write("Your EMTOM tasks should ADD: information asymmetry, agent coordination.\n\n")

            for i, sample in enumerate(samples):
                f.write(f"--- Example {i+1}: {sample['category'].upper()} ---\n")
                f.write(f"Task: {sample['instruction']}\n")

                if sample['evaluation_propositions']:
                    # Group by predicate type and count
                    predicates = {}
                    for prop in sample['evaluation_propositions']:
                        func = prop['function']
                        predicates[func] = predicates.get(func, 0) + 1
                    pred_summary = ', '.join(f"{func}({count})" for func, count in predicates.items())
                    f.write(f"Goal predicates: {pred_summary}\n")

                if sample.get('object_states'):
                    state_types = list(sample['object_states'].keys())
                    f.write(f"Uses initial_states: {', '.join(state_types)}\n")

                f.write("\n")

        if verbose:
            print(f"[sample_tasks] Saved {num_samples} examples to: {output_dir}")

        return True

    except Exception as e:
        if verbose:
            print(f"[sample_tasks] Error: {e}")
        return False


# Backwards compatibility alias
sample_partnr_to_directory = sample_planning_tasks_to_directory


def main():
    parser = argparse.ArgumentParser(description="Sample PARTNR tasks for inspiration")
    parser.add_argument('--num', type=int, default=50, help='Number of tasks to sample')
    parser.add_argument('--output', type=str, default='/tmp/partnr_samples', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='train',
                       choices=['train', 'val', 'val_mini', 'train_mini'],
                       help='Which dataset to sample from')
    args = parser.parse_args()

    # Find dataset
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'versioned_data' / 'partnr_episodes' / 'v0_0' / f'{args.dataset}.json.gz'

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return 1

    print(f"Loading PARTNR dataset: {data_path}")
    episodes = load_partnr_episodes(data_path)
    print(f"Loaded {len(episodes)} episodes")

    # Sample diverse tasks
    print(f"\nSampling {args.num} diverse tasks...")
    samples = sample_diverse_tasks(episodes, args.num, args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual task files for easy reading
    for i, sample in enumerate(samples):
        task_file = output_dir / f"task_{i:03d}_{sample['category']}.json"
        with open(task_file, 'w') as f:
            json.dump(sample, f, indent=2)

    # Save summary file
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'num_samples': len(samples),
            'source': str(data_path),
            'seed': args.seed,
            'samples': samples,
        }, f, indent=2)

    # Save readable text version
    text_file = output_dir / 'tasks.txt'
    with open(text_file, 'w') as f:
        f.write("PARTNR Task Samples for EMTOM Inspiration\n")
        f.write("=" * 60 + "\n\n")

        for i, sample in enumerate(samples):
            f.write(f"--- Task {i+1} [{sample['category']}] ---\n")
            f.write(f"Episode: {sample['episode_id']} | Scene: {sample['scene_id']}\n")
            f.write(f"Instruction: {sample['instruction']}\n")
            f.write(f"Objects in scene: {sample['num_rigid_objects']}\n")

            if sample['evaluation_propositions']:
                f.write("Success conditions:\n")
                for prop in sample['evaluation_propositions']:
                    f.write(f"  - {prop['function']}({', '.join(prop['objects'])}) -> {', '.join(prop['receptacles'])}\n")

            if sample.get('object_states'):
                f.write(f"Object states: {sample['object_states']}\n")

            if sample.get('ao_states'):
                f.write(f"Articulated states: {sample['ao_states']}\n")

            f.write("\n")

    print(f"\nSaved {len(samples)} task samples to: {output_dir}")
    print(f"  - Individual JSON files: task_XXX_<category>.json")
    print(f"  - Summary: summary.json")
    print(f"  - Readable: tasks.txt")

    return 0


if __name__ == '__main__':
    exit(main())
