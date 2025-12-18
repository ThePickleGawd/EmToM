#!/usr/bin/env python3
"""
Entry point for the agentic task generator.

Usage:
    python -m emtom.task_gen.runner --num-tasks 10 --model gpt-5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic task generator for EMTOM benchmark"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of quality tasks to generate (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="LLM model to use for the agent (default: gpt-5)"
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default="data/emtom/trajectories",
        help="Directory containing trajectory JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/emtom/tasks/curated",
        help="Directory for curated output tasks"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum ReAct iterations before stopping (default: 100)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="examples/emtom_two_robots",
        help="Hydra config name for benchmark runner"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print agent thoughts and actions"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    return parser.parse_args()


def load_config(config_name: str) -> DictConfig:
    """Load Hydra configuration."""
    # Initialize Hydra
    with hydra.initialize(
        version_base=None,
        config_path="../../../habitat_llm/conf",
    ):
        config = hydra.compose(config_name=config_name)
    return config


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("EMTOM Agentic Task Generator")
    print("=" * 60)
    print(f"Target tasks: {args.num_tasks}")
    print(f"Model: {args.model}")
    print(f"Trajectories: {args.trajectory_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    print()

    # Check trajectory directory
    if not Path(args.trajectory_dir).exists():
        print(f"Error: Trajectory directory not found: {args.trajectory_dir}")
        print("Run exploration first: ./emtom/run_emtom.sh explore")
        sys.exit(1)

    # Check for trajectories
    trajectories = list(Path(args.trajectory_dir).glob("*.json"))
    if not trajectories:
        print(f"Error: No trajectory files found in {args.trajectory_dir}")
        print("Run exploration first: ./emtom/run_emtom.sh explore")
        sys.exit(1)

    print(f"Found {len(trajectories)} trajectory files")
    print()

    # Load config
    print("Loading Hydra configuration...")
    try:
        config = load_config(args.config_name)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using minimal config for testing")
        config = OmegaConf.create({})

    # Initialize LLM
    print("Initializing LLM client...")
    try:
        from habitat_llm.llm import instantiate_llm
        llm_client = instantiate_llm("openai_chat", model=args.model)
        print(f"Using model: {llm_client.generation_params.model}")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    # Create and run agent
    from emtom.task_gen.agent import TaskGeneratorAgent

    agent = TaskGeneratorAgent(
        llm_client=llm_client,
        config=config,
        working_dir="data/emtom/tasks",
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )

    # Run agent
    submitted_tasks = agent.run(num_tasks_target=args.num_tasks)

    # Summary
    print()
    print("=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Tasks generated: {len(submitted_tasks)}")
    for task_path in submitted_tasks:
        print(f"  - {task_path}")
    print()


if __name__ == "__main__":
    main()
