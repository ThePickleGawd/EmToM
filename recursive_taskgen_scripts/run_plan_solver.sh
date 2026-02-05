#!/usr/bin/env bash
# Wrapper script for plan_solver.py
#
# Usage:
#   ./run_plan_solver.sh --rows 0,9                    # rows 0-9 inclusive (10 rows)
#   ./run_plan_solver.sh --rows 0,99                   # rows 0-99 inclusive (100 rows)
#   ./run_plan_solver.sh --rows 5,5                    # single row 5
#   ./run_plan_solver.sh --n_samples 50 --seed 123     # 50 random rows
#   ./run_plan_solver.sh --rows 0,4 --peek             # preview stripped text, no LLM call
#   ./run_plan_solver.sh --rows 0,99 --model gpt-5     # use a specific model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="habitat-llm"

conda run -n "$CONDA_ENV" python3 "$SCRIPT_DIR/plan_solver.py" "$@"
