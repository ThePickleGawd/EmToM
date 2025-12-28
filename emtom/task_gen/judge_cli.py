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

    # Build config for OpenAI client (must match OpenAIChat expectations)
    config = OmegaConf.create({
        "llm": {
            "verbose": False,
            "keep_message_history": False,
            "system_message": "You are an expert evaluator for Theory of Mind tasks.",
            "generation_params": {
                "model": model,
                "temperature": 0.0,  # Deterministic for evaluation
                "max_tokens": 2000,
                "stream": False,
                "stop": [],
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

    # Get JSON output
    json_output = judgment.to_json()

    # Save to file in same directory as input task
    output_dir = task_path.parent

    # Find next available verification number
    existing = list(output_dir.glob("ToM_verification_*.json"))
    if existing:
        # Extract numbers and find max
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split("_")[-1])
                numbers.append(num)
            except ValueError:
                pass
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1

    output_file = output_dir / f"ToM_verification_{next_num}.json"
    with open(output_file, "w") as f:
        f.write(json_output)

    # Print JSON to stdout
    print(json_output)
    print(f"\nSaved to: {output_file}", file=sys.stderr)

    # Exit code based on pass/fail
    sys.exit(0 if judgment.is_valid_tom else 1)


if __name__ == "__main__":
    main()
