#!/bin/bash
# EMTOM Benchmark Pipeline
# Usage: ./emtom/run_emtom.sh [exploration|generate|benchmark|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default values
MAX_SIM_STEPS=20000
MAX_LLM_CALLS=20
EXPLORATION_STEPS=50

print_usage() {
    echo "EMTOM Benchmark Pipeline"
    echo ""
    echo "Usage: ./emtom/run_emtom.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  exploration    Run LLM-guided exploration in Habitat"
    echo "  generate       Generate tasks from exploration trajectories"
    echo "  benchmark      Run benchmark with generated tasks"
    echo "  all            Run full pipeline: exploration -> generate -> benchmark"
    echo ""
    echo "Exploration Options:"
    echo "  --steps N            Number of exploration steps (default: $EXPLORATION_STEPS)"
    echo ""
    echo "Task Generation:"
    echo "  (Interactive prompt will ask for task type: 1=Theory of Mind, 2=Regular)"
    echo ""
    echo "Benchmark Options:"
    echo "  --max-sim-steps N    Max simulation steps before timeout (default: $MAX_SIM_STEPS)"
    echo "  --max-llm-calls N    Max LLM calls per agent (default: $MAX_LLM_CALLS)"
    echo ""
    echo "Examples:"
    echo "  ./emtom/run_emtom.sh exploration --steps 30"
    echo "  ./emtom/run_emtom.sh generate"
    echo "  ./emtom/run_emtom.sh benchmark --max-sim-steps 1000 --max-llm-calls 15"
    echo "  ./emtom/run_emtom.sh all --steps 50 --max-sim-steps 2000"
}

run_exploration() {
    echo "=============================================="
    echo "Running EMTOM Exploration (Habitat Backend)"
    echo "=============================================="
    echo "Mode: LLM-guided"
    echo "Steps: $EXPLORATION_STEPS"
    echo "=============================================="
    echo ""

    # Use Hydra config system - pass parameters as config overrides
    # Override output dir to include "exploration" in the name
    python emtom/examples/run_habitat_exploration.py \
        --config-name examples/planner_multi_agent_demo_config \
        +exploration_steps=$EXPLORATION_STEPS \
        evaluation.save_video=true \
        "hydra.run.dir=./outputs/emtom/\${now:%Y-%m-%d_%H-%M-%S}-exploration"
}

run_generate() {
    echo "=============================================="
    echo "Running EMTOM Task Generation"
    echo "=============================================="
    echo ""
    echo "Select the type of task to generate:"
    echo "  1) Theory of Mind tasks"
    echo "  2) Regular tasks"
    echo ""
    read -p "Enter your choice (1 or 2): " TASK_TYPE

    # Validate input
    while [ "$TASK_TYPE" != "1" ] && [ "$TASK_TYPE" != "2" ]; do
        echo "Invalid choice. Please enter 1 or 2."
        read -p "Enter your choice (1 or 2): " TASK_TYPE
    done

    if [ "$TASK_TYPE" -eq 1 ]; then
        TASK_TYPE_NAME="Theory of Mind"
    else
        TASK_TYPE_NAME="Regular"
    fi

    echo ""
    echo "Task Type: $TASK_TYPE_NAME (option $TASK_TYPE)"
    echo "This generates tasks from exploration trajectories."
    echo ""
    python emtom/examples/generate_tasks.py \
        --trajectory-dir data/emtom/trajectories \
        --output-dir data/emtom/tasks \
        --task-type $TASK_TYPE
}

run_benchmark() {
    echo "=============================================="
    echo "Running EMTOM Habitat Benchmark"
    echo "=============================================="
    echo "Max simulation steps: $MAX_SIM_STEPS"
    echo "Max LLM calls per agent: $MAX_LLM_CALLS"
    echo "=============================================="

    # Override output dir to include "benchmark" in the name
    python emtom/examples/run_habitat_benchmark.py \
        --config-name examples/emtom_two_robots \
        habitat.environment.max_episode_steps=$MAX_SIM_STEPS \
        evaluation.agents.agent_0.planner.plan_config.replanning_threshold=$MAX_LLM_CALLS \
        evaluation.agents.agent_1.planner.plan_config.replanning_threshold=$MAX_LLM_CALLS \
        "hydra.run.dir=./outputs/emtom/\${now:%Y-%m-%d_%H-%M-%S}-benchmark"
}

run_all() {
    echo "=============================================="
    echo "Running Full EMTOM Pipeline"
    echo "=============================================="
    run_exploration
    run_generate
    run_benchmark
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        exploration|generate|benchmark|all)
            COMMAND=$1
            shift
            ;;
        --max-sim-steps)
            MAX_SIM_STEPS=$2
            shift 2
            ;;
        --max-llm-calls)
            MAX_LLM_CALLS=$2
            shift 2
            ;;
        --steps)
            EXPLORATION_STEPS=$2
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    print_usage
    exit 1
fi

case $COMMAND in
    exploration)
        run_exploration
        ;;
    generate)
        run_generate
        ;;
    benchmark)
        run_benchmark
        ;;
    all)
        run_all
        ;;
esac

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
