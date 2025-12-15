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
EXPLORATION_STEPS=3
TASK_TYPE=1
MECHANICS=""
TASK_FILE=""
LLM_AGENTS=""

print_usage() {
    echo "EMTOM Benchmark Pipeline"
    echo ""
    echo "Usage: ./emtom/run_emtom.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  exploration    Run LLM-guided exploration in Habitat"
    echo "  generate       Generate tasks from exploration trajectories"
    echo "  benchmark      Run benchmark with generated tasks"
    echo "  test           Human-in-the-loop testing mode (manual command input)"
    echo "  all            Run full pipeline: exploration -> generate -> benchmark"
    echo ""
    echo "Exploration Options:"
    echo "  --steps N            Number of exploration steps (default: $EXPLORATION_STEPS)"
    echo ""
    echo "Task Generation Options:"
    echo "  --task-type N        1=Theory of Mind (default), 2=Regular tasks"
    echo ""
    echo "Benchmark Options:"
    echo "  --max-sim-steps N    Max simulation steps before timeout (default: $MAX_SIM_STEPS)"
    echo "  --max-llm-calls N    Max LLM calls per agent (default: $MAX_LLM_CALLS)"
    echo ""
    echo "Test Options:"
    echo "  --mechanics M1 M2    Mechanics to enable (e.g., inverse_state remote_control)"
    echo "  --task FILE          Task file to load (uses task's mechanic bindings automatically)"
    echo "  --llm-agents A1 A2   Agents to make LLM-controlled (e.g., agent_1)"
    echo ""
    echo "Examples:"
    echo "  ./emtom/run_emtom.sh exploration --steps 30"
    echo "  ./emtom/run_emtom.sh generate"
    echo "  ./emtom/run_emtom.sh benchmark --max-sim-steps 1000 --max-llm-calls 15"
    echo "  ./emtom/run_emtom.sh test --mechanics inverse_state remote_control"
    echo "  ./emtom/run_emtom.sh test --task data/emtom/tasks/emtom_tom_test.json"
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
    if [ "$TASK_TYPE" -eq 1 ]; then
        TASK_TYPE_NAME="Theory of Mind"
    else
        TASK_TYPE_NAME="Regular"
    fi

    echo "=============================================="
    echo "Running EMTOM Task Generation"
    echo "=============================================="
    echo "Task Type: $TASK_TYPE_NAME"
    echo "(use --task-type 2 for regular tasks)"
    echo "=============================================="
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

run_test() {
    echo "=============================================="
    echo "Running EMTOM Human Test Mode"
    echo "=============================================="
    echo "Mechanics: ${MECHANICS:-from task file or default}"
    echo "Task file: ${TASK_FILE:-none}"
    echo "LLM agents: ${LLM_AGENTS:-none (all human-controlled)}"
    echo "=============================================="
    echo ""
    echo "Commands: Open[target], Close[target], Navigate[target], Pick[target]"
    echo "          Hide[target], Inspect[target], WriteMessage[target]"
    echo "          llm, status, mechanics, history, help, quit"
    echo ""

    # Build command arguments
    CMD_ARGS=""
    if [ -n "$MECHANICS" ]; then
        CMD_ARGS="$CMD_ARGS --mechanics $MECHANICS"
    fi
    if [ -n "$TASK_FILE" ]; then
        CMD_ARGS="$CMD_ARGS --task $TASK_FILE"
        # Use task mechanics by default when task is provided
        CMD_ARGS="$CMD_ARGS --use-task-mechanics"
    fi
    if [ -n "$LLM_AGENTS" ]; then
        CMD_ARGS="$CMD_ARGS --llm-agents $LLM_AGENTS"
    fi

    python emtom/examples/run_human_test.py \
        --config-name examples/planner_multi_agent_demo_config \
        $CMD_ARGS \
        "hydra.run.dir=./outputs/emtom/\${now:%Y-%m-%d_%H-%M-%S}-human_test"
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
        exploration|generate|benchmark|test|all)
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
        --task-type)
            TASK_TYPE=$2
            shift 2
            ;;
        --mechanics)
            # Collect all mechanics until next flag or end
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MECHANICS="$MECHANICS $1"
                shift
            done
            MECHANICS=$(echo "$MECHANICS" | xargs)  # trim whitespace
            ;;
        --task)
            TASK_FILE=$2
            shift 2
            ;;
        --llm-agents)
            # Collect all agents until next flag or end
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LLM_AGENTS="$LLM_AGENTS $1"
                shift
            done
            LLM_AGENTS=$(echo "$LLM_AGENTS" | xargs)  # trim whitespace
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
    test)
        run_test
        ;;
    all)
        run_all
        ;;
esac

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
