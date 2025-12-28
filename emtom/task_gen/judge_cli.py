#!/usr/bin/env python3
"""
Standalone CLI for Theory of Mind task evaluation.

Usage:
    python -m emtom.task_gen.judge_cli --task <path_to_task.json> [--model gpt-5]
"""

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from emtom.task_gen.tom_judge import ToMJudge


def create_llm_client(model: str):
    """Create an LLM client for the specified model."""
    from habitat_llm.llm.openai_chat import OpenAIChat

    # Build config for OpenAI client
    config = OmegaConf.create({
        "llm": {
            "model": model,
            "generation_params": {
                "temperature": 0.0,  # Deterministic for evaluation
                "max_tokens": 2000,
            }
        }
    })

    return OpenAIChat(config.llm)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a task for Theory of Mind requirements"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Path to task JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="LLM model to use for evaluation (default: gpt-5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Overall score threshold for passing (default: 0.7)"
    )
    parser.add_argument(
        "--min-criterion",
        type=float,
        default=0.5,
        help="Minimum score for any criterion (default: 0.5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Load task
    task_path = Path(args.task)
    if not task_path.exists():
        print(f"Error: Task file not found: {task_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in task file: {e}", file=sys.stderr)
        sys.exit(1)

    # Create LLM client and judge
    if args.verbose:
        print(f"Using model: {args.model}")
        print(f"Evaluating task: {task_path}")

    llm_client = create_llm_client(args.model)
    judge = ToMJudge(
        llm_client,
        overall_threshold=args.threshold,
        min_criterion_threshold=args.min_criterion,
        verbose=args.verbose
    )

    # Run evaluation
    judgment = judge.evaluate(task_data)

    # Output results
    if args.json:
        print(judgment.to_json())
    else:
        print(judge.format_result(judgment))

    # Exit code based on pass/fail
    sys.exit(0 if judgment.is_valid_tom else 1)


if __name__ == "__main__":
    main()
