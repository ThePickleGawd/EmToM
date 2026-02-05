#!/usr/bin/env python3
"""
Test recursive task generation by having an LLM solve planning problems.

1. Randomly selects N rows from the plan generation dataset
2. Strips each query to: action descriptions + restrictions + [STATEMENT] with
   initial conditions and goal (no plan provided)
3. Sends stripped queries to ChatGPT for plan generation
4. Saves LLM responses alongside ground truth for comparison
"""

import json
import os
import re
import argparse
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI

# Load .env if it exists (mirrors pattern from habitat_llm/llm/openai_chat.py)
_env_file = Path(__file__).resolve().parent.parent / ".env"
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
                    os.environ.setdefault(
                        key.strip(), value.strip().strip('"').strip("'")
                    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "recursive_taskgen_data"
DEFAULT_PARQUET = (
    DATA_DIR
    / "task_1_plan_generation"
    / "train-00000-of-00001-f765a1b29ae17c5a.parquet"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

SYSTEM_PROMPT = "You are an expert planner. You solve planning problems by outputting action sequences. Output ONLY the plan steps with no additional text."


def strip_query(query: str) -> str:
    """
    Strip the query to include (one-shot format):
      - Action descriptions and restrictions
      - First [STATEMENT] with its full plan (the one-shot example)
      - Last [STATEMENT] with initial conditions + goal only (problem to solve)
    The plan for the last statement is removed so the LLM generates it.
    """
    # Split on [STATEMENT] markers
    parts = query.split("[STATEMENT]")

    # Part 0: action descriptions and restrictions
    preamble = parts[0].rstrip()

    # Part 1: one-shot example (keep fully intact including its plan)
    oneshot_example = parts[1].rstrip()

    # Last part: the test problem (strip the empty plan section)
    last_statement = parts[-1]
    goal_match = re.search(
        r"(.*?My goal is to have that[^\n]*\.)", last_statement, re.DOTALL
    )
    if goal_match:
        test_problem = goal_match.group(1).strip()
    else:
        test_problem = last_statement.split("My plan is as follows")[0].strip()

    return (
        f"{preamble}\n\n[STATEMENT]\n{oneshot_example}\n\n"
        f"[STATEMENT]\n{test_problem}"
    )


def build_prompt(stripped_query: str) -> str:
    """Build the user prompt for the LLM."""
    return (
        f"{stripped_query}\n\n"
        "Provide the plan to achieve the goal. "
        "Output ONLY the plan steps, one per line, in the following format:\n"
        "(action_name oN oN ...)\n\n"
        "Rules:\n"
        "- Use lowercase action names\n"
        "- Use shorthand 'oN' instead of 'object_N' (e.g., object_3 -> o3)\n"
        "- One action per line, wrapped in parentheses\n"
        "- No explanations, commentary, or extra text"
    )


def call_chatgpt(client: OpenAI, prompt: str, model: str) -> str:
    """Send a prompt to ChatGPT and return the response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def select_rows(df: pd.DataFrame, rows: str = None, n_samples: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Select rows from the dataframe.
    --rows START,END  → inclusive range [START, END]
    --n_samples N     → randomly sample N rows
    """
    if rows:
        parts = rows.split(",")
        start, end = int(parts[0]), int(parts[1])
        if start < 0 or end >= len(df) or start > end:
            raise ValueError(
                f"Row range {start},{end} is out of bounds (dataset has {len(df)} rows, 0-{len(df)-1})"
            )
        return df.iloc[start : end + 1]
    else:
        n = min(n_samples, len(df))
        return df.sample(n=n, random_state=seed)


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM plan solving on planning domain problems"
    )
    parser.add_argument(
        "--rows", type=str, default=None,
        help="Inclusive row range, e.g. '0,9' selects rows 0-9 (10 rows)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of random rows to sample (ignored if --rows is set)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5-mini", help="ChatGPT model to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--peek", type=int, default=None, metavar="ROW",
        help="Print stripped text for a specific row index (relative to selected rows, 0-based) without calling the LLM"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to parquet file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory"
    )
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else DEFAULT_PARQUET
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} rows")

    # Select rows
    sampled = select_rows(df, rows=args.rows, n_samples=args.n_samples, seed=args.seed)
    n = len(sampled)
    if args.rows:
        log.info(f"Selected rows {args.rows} ({n} rows)")
    else:
        log.info(f"Randomly sampled {n} rows (seed={args.seed})")

    # Peek mode: print stripped text for a single row and exit
    if args.peek is not None:
        if args.peek < 0 or args.peek >= n:
            raise ValueError(
                f"--peek {args.peek} is out of bounds for selected {n} rows (use 0-{n-1})"
            )
        row = sampled.iloc[args.peek]
        row_idx = sampled.index[args.peek]
        stripped = strip_query(row["query"])
        print(f"{'='*80}")
        print(f"ROW {row_idx} | domain={row['domain']} | instance_id={row['instance_id']}")
        print(f"{'='*80}")
        print(stripped)
        print(f"\n--- GROUND TRUTH ---")
        print(row["ground_truth_plan"])
        return

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Export it or add it to .env in the project root."
        )
    client = OpenAI(api_key=api_key)

    results = []
    for idx, (row_idx, row) in enumerate(sampled.iterrows()):
        log.info(
            f"[{idx + 1}/{n}] Row {row_idx} | domain={row['domain']} | "
            f"instance_id={row['instance_id']}"
        )

        stripped = strip_query(row["query"])
        prompt = build_prompt(stripped)

        try:
            llm_response = call_chatgpt(client, prompt, model=args.model)
        except Exception as e:
            log.error(f"  API error: {e}")
            llm_response = f"ERROR: {e}"

        results.append(
            {
                "original_index": int(row_idx),
                "domain": row["domain"],
                "instance_id": int(row["instance_id"]),
                "stripped_query": stripped,
                "ground_truth_plan": row["ground_truth_plan"],
                "llm_response": llm_response,
            }
        )

        gt_preview = row["ground_truth_plan"].replace("\n", " | ")[:80]
        resp_preview = llm_response.replace("\n", " | ")[:80]
        log.info(f"  GT:  {gt_preview}")
        log.info(f"  LLM: {resp_preview}")

    # Save results
    row_tag = args.rows.replace(",", "-") if args.rows else f"seed{args.seed}"
    output_file = output_dir / f"plan_results_{args.model}_{row_tag}_n{n}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_file}")

    # Quick accuracy summary
    exact_matches = sum(
        1
        for r in results
        if r["llm_response"].strip() == r["ground_truth_plan"].strip()
        and not r["llm_response"].startswith("ERROR")
    )
    errors = sum(1 for r in results if r["llm_response"].startswith("ERROR"))
    log.info(
        f"Exact matches: {exact_matches}/{n} ({exact_matches / n * 100:.1f}%) | "
        f"Errors: {errors}/{n}"
    )


if __name__ == "__main__":
    main()
