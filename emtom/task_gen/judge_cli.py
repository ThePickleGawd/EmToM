#!/usr/bin/env python3
"""
Standalone CLI for Theory of Mind task evaluation.

Usage:
    python -m emtom.task_gen.judge_cli --task <path> --llm openai_chat --model gpt-5
    python -m emtom.task_gen.judge_cli --task <path> --llm bedrock_claude --model sonnet
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from emtom.task_gen.tom_judge import ToMJudge


# ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


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

    # Run evaluation
    judgment = judge.evaluate(task_data)

    # Get JSON output
    json_output = judgment.to_json()

    # Save to timestamped output directory (like other commands)
    project_root = Path(__file__).resolve().parent.parent.parent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = project_root / "outputs" / "emtom" / f"{timestamp}-judge"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use task name in output filename
    task_name = task_path.stem
    output_file = output_dir / f"ToM_verification_{task_name}.json"
    with open(output_file, "w") as f:
        f.write(json_output)

    # Print JSON to stdout
    print(json_output)

    # Print colored output path
    print(f"\nSaved to: {Colors.CYAN}{output_file}{Colors.RESET}", file=sys.stderr)

    # Print pass/fail status with colors
    if judgment.is_valid_tom:
        print(f"\n{Colors.BOLD}{Colors.GREEN}PASSED ToM Task Verification{Colors.RESET}\n", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}FAILED ToM Task Verification{Colors.RESET}", file=sys.stderr)
        print(f"\n{Colors.YELLOW}Suggestions for improvement:{Colors.RESET}", file=sys.stderr)
        for i, suggestion in enumerate(judgment.suggestions, 1):
            print(f"  {i}. {suggestion}", file=sys.stderr)

        if args.no_auto_retry:
            # Manual retry mode - just print the command
            print(f"\n{Colors.BOLD}{Colors.YELLOW}Run the generator to create a new task based on these suggestions:{Colors.RESET}", file=sys.stderr)
            print(f"  ./emtom/run_emtom.sh generate --retry-verification {output_file}\n", file=sys.stderr)
            sys.exit(1)
        else:
            # Auto-retry mode - spawn the generator
            print(f"\n{Colors.BOLD}{Colors.CYAN}Automatically generating a new task based on suggestions...{Colors.RESET}\n", file=sys.stderr)

            # Find project root (where run_emtom.sh lives)
            project_root = Path(__file__).resolve().parent.parent.parent

            # Build the command
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
                sys.exit(result.returncode)
            except Exception as e:
                print(f"{Colors.RED}Error running generator: {e}{Colors.RESET}", file=sys.stderr)
                print(f"You can manually retry with:", file=sys.stderr)
                print(f"  ./emtom/run_emtom.sh generate --retry-verification {output_file}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()
