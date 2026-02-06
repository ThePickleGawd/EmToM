#!/bin/bash
# Evolutionary Difficulty Task Generation
# Usage: ./emtom/run_evolve.sh [options]
#
# Options:
#   --model-ladder "ministral-3-8b,haiku,gpt-5-mini,sonnet,gpt-5.1,gpt-5.2"
#   --generator-model gpt-5.2
#   --tasks-per-round 20
#   --seed-pool-size 30
#   --resume <output_dir>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

python -m emtom.evolve.orchestrator "$@"
