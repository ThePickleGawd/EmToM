#!/usr/bin/env python3
"""
Standalone CLI for Theory of Mind task evaluation.

Usage:
    python -m emtom.task_gen.judge_cli --task <path> --llm openai_chat --model gpt-5
    python -m emtom.task_gen.judge_cli --task <path> --llm bedrock_claude --model sonnet
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from emtom.task_gen.tom_judge import ToMJudge


# ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# Default max retry attempts
DEFAULT_MAX_RETRIES = 5


def create_llm_client(model: str, llm_provider: str):
    """Create an LLM client for the specified model and provider."""
    from habitat_llm.llm import instantiate_llm

    return instantiate_llm(
        llm_provider,
        generation_params={
            "model": model,
            "temperature": 0.0,  # Deterministic for evaluation
            "max_tokens": 2000,
        }
    )


def find_latest_generated_task(tasks_dir: Path) -> Optional[Path]:
    """Find the most recently modified task file in the tasks directory."""
    task_files = list(tasks_dir.glob("*.json"))
    if not task_files:
        return None
    # Return the most recently modified file
    return max(task_files, key=lambda p: p.stat().st_mtime)


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
        "--llm",
        type=str,
        required=True,
        help="LLM provider: openai_chat, bedrock_claude"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model name (e.g., gpt-5, sonnet, opus)"
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
        "--no-auto-retry",
        action="store_true",
        help="Disable automatic retry with generator on failure"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts before giving up (default: {DEFAULT_MAX_RETRIES})"
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
        print(f"Using LLM: {args.llm} ({args.model})")
        print(f"Evaluating task: {task_path}")

    llm_client = create_llm_client(args.model, args.llm)
    judge = ToMJudge(
        llm_client,
        overall_threshold=args.threshold,
        min_criterion_threshold=args.min_criterion,
        verbose=args.verbose
    )

    # Find project root and tasks directory
    project_root = Path(__file__).resolve().parent.parent.parent
    tasks_dir = project_root / "data" / "emtom" / "tasks"

    # Retry loop: keep generating and judging until task passes or max retries reached
    attempt = 0
    current_task_path = task_path
    current_task_data = task_data

    while True:
        attempt += 1
        print(f"\n{Colors.BOLD}{Colors.CYAN}=== ToM Verification Attempt {attempt}/{args.max_retries} ==={Colors.RESET}", file=sys.stderr)

        # Run evaluation
        judgment = judge.evaluate(current_task_data)

        # Get JSON output
        json_output = judgment.to_json()

        # Save to timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = project_root / "outputs" / "emtom" / f"{timestamp}-judge"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use task name in output filename
        task_name = current_task_path.stem
        output_file = output_dir / f"ToM_verification_{task_name}.json"
        with open(output_file, "w") as f:
            f.write(json_output)

        # Print JSON to stdout
        print(json_output)

        # Print colored output path
        print(f"\nSaved to: {Colors.CYAN}{output_file}{Colors.RESET}", file=sys.stderr)

        # Check if task passed
        if judgment.is_valid_tom:
            print(f"\n{Colors.BOLD}{Colors.GREEN}PASSED ToM Task Verification (attempt {attempt}){Colors.RESET}", file=sys.stderr)
            print(f"{Colors.GREEN}Valid task: {current_task_path}{Colors.RESET}\n", file=sys.stderr)
            sys.exit(0)

        # Task failed
        print(f"\n{Colors.BOLD}{Colors.RED}FAILED ToM Task Verification (attempt {attempt}){Colors.RESET}", file=sys.stderr)
        print(f"\n{Colors.YELLOW}Suggestions for improvement:{Colors.RESET}", file=sys.stderr)
        for i, suggestion in enumerate(judgment.suggestions, 1):
            print(f"  {i}. {suggestion}", file=sys.stderr)

        # Check if we should retry
        if args.no_auto_retry:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}Run the generator to create a new task based on these suggestions:{Colors.RESET}", file=sys.stderr)
            print(f"  ./emtom/run_emtom.sh generate --retry-verification {output_file}\n", file=sys.stderr)
            sys.exit(1)

        if attempt >= args.max_retries:
            print(f"\n{Colors.BOLD}{Colors.RED}Max retries ({args.max_retries}) reached. Giving up.{Colors.RESET}", file=sys.stderr)
            print(f"Last failed task: {current_task_path}", file=sys.stderr)
            print(f"\n{Colors.YELLOW}You can manually retry with:{Colors.RESET}", file=sys.stderr)
            print(f"  ./emtom/run_emtom.sh generate --retry-verification {output_file}\n", file=sys.stderr)
            sys.exit(1)

        # Auto-retry: generate a new task based on suggestions
        print(f"\n{Colors.BOLD}{Colors.CYAN}Generating new task based on suggestions (attempt {attempt + 1})...{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.CYAN}Passing suggestions from: {output_file}{Colors.RESET}", file=sys.stderr)
        print(f"{Colors.CYAN}Number of suggestions: {len(judgment.suggestions)}{Colors.RESET}\n", file=sys.stderr)

        # Record the latest task modification time before generation
        latest_before = find_latest_generated_task(tasks_dir)
        latest_mtime_before = latest_before.stat().st_mtime if latest_before else 0

        # Build the generate command
        cmd = [
            str(project_root / "emtom" / "run_emtom.sh"),
            "generate",
            "--retry-verification", str(output_file),
            "--model", args.model,
            "--llm", args.llm,
        ]

        # Run the generator
        try:
            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode != 0:
                print(f"{Colors.RED}Generator failed with code {result.returncode}{Colors.RESET}", file=sys.stderr)
                sys.exit(result.returncode)
        except Exception as e:
            print(f"{Colors.RED}Error running generator: {e}{Colors.RESET}", file=sys.stderr)
            sys.exit(1)

        # Find the newly generated task
        latest_after = find_latest_generated_task(tasks_dir)
        if latest_after is None:
            print(f"{Colors.RED}No task files found in {tasks_dir}{Colors.RESET}", file=sys.stderr)
            sys.exit(1)

        if latest_after.stat().st_mtime <= latest_mtime_before:
            print(f"{Colors.RED}Generator did not create a new task file{Colors.RESET}", file=sys.stderr)
            sys.exit(1)

        # Load the new task for the next iteration
        current_task_path = latest_after
        try:
            with open(current_task_path) as f:
                current_task_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Invalid JSON in generated task: {e}{Colors.RESET}", file=sys.stderr)
            sys.exit(1)

        print(f"\n{Colors.CYAN}New task generated: {current_task_path}{Colors.RESET}", file=sys.stderr)


if __name__ == "__main__":
    main()
