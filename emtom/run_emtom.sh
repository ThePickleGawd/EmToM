#!/bin/bash
# EMTOM Benchmark Pipeline
# Usage: ./emtom/run_emtom.sh <command> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Expand short model names to full model IDs
expand_model_name() {
    local model=$1
    case $model in
        kimi-k2-thinking)       echo "moonshot.kimi-k2-thinking" ;;
        ministral-3-8b)         echo "mistral.ministral-3-8b-instruct" ;;
        ministral-3-14b)        echo "mistral.ministral-3-14b-instruct" ;;
        mistral-large-3)        echo "mistral.mistral-large-3-675b-instruct" ;;
        qwen3-next-80b)         echo "qwen.qwen3-next-80b-a3b" ;;
        qwen3-vl-235b)          echo "qwen.qwen3-vl-235b-a22b" ;;
        *)                      echo "$model" ;;  # Return as-is if no mapping
    esac
}

# Auto-detect LLM provider from model name
detect_llm_provider() {
    local model=$1
    case $model in
        gpt-5|gpt-5-mini|gpt-5.1|gpt-5.2)
            echo "openai_chat" ;;
        sonnet|haiku|opus)
            echo "bedrock_claude" ;;
        kimi-k2-thinking|moonshot.kimi-k2-thinking)
            echo "bedrock_kimi" ;;
        ministral-3-8b|ministral-3-14b|mistral-large-3|mistral.ministral-3-8b-instruct|mistral.ministral-3-14b-instruct|mistral.mistral-large-3-675b-instruct)
            echo "bedrock_mistral" ;;
        qwen3-next-80b|qwen3-vl-235b|qwen.qwen3-next-80b-a3b|qwen.qwen3-vl-235b-a22b)
            echo "bedrock_qwen" ;;
        *)
            echo "" ;;  # Unknown model
    esac
}

print_llm_options() {
    echo ""
    echo -e "${RED}Error: LLM model must be specified${NC}"
    echo ""
    echo -e "${BOLD}Available Models:${NC}"
    echo -e "┌───────────────────────────┬────────────────────┐"
    echo -e "│ ${BOLD}Model Name${NC}                │ ${BOLD}--model${NC}            │"
    echo -e "├───────────────────────────┼────────────────────┤"
    echo -e "│ GPT-5                     │ ${GREEN}gpt-5${NC}              │"
    echo -e "│ GPT-5 Mini                │ ${GREEN}gpt-5-mini${NC}         │"
    echo -e "│ GPT-5.1                   │ ${GREEN}gpt-5.1${NC}            │"
    echo -e "│ GPT-5.2 (default)         │ ${GREEN}gpt-5.2${NC}            │"
    echo -e "├───────────────────────────┼────────────────────┤"
    echo -e "│ Claude Sonnet             │ ${GREEN}sonnet${NC}             │"
    echo -e "│ Claude Haiku              │ ${GREEN}haiku${NC}              │"
    echo -e "│ Claude Opus               │ ${GREEN}opus${NC}               │"
    echo -e "├───────────────────────────┼────────────────────┤"
    echo -e "│ Kimi K2 Thinking          │ ${GREEN}kimi-k2-thinking${NC}   │"
    echo -e "├───────────────────────────┼────────────────────┤"
    echo -e "│ Ministral 3 8B            │ ${GREEN}ministral-3-8b${NC}     │"
    echo -e "│ Ministral 3 14B           │ ${GREEN}ministral-3-14b${NC}    │"
    echo -e "│ Mistral Large 3           │ ${GREEN}mistral-large-3${NC}    │"
    echo -e "├───────────────────────────┼────────────────────┤"
    echo -e "│ Qwen3 Next 80B            │ ${GREEN}qwen3-next-80b${NC}     │"
    echo -e "│ Qwen3 VL 235B             │ ${GREEN}qwen3-vl-235b${NC}      │"
    echo -e "└───────────────────────────┴────────────────────┘"
    echo ""
    echo -e "${YELLOW}Example usage:${NC}"
    echo -e "  ./emtom/run_emtom.sh generate ${GREEN}--model gpt-5${NC}"
    echo -e "  ./emtom/run_emtom.sh judge --task task.json ${GREEN}--model mistral-large-3${NC}"
    echo ""
}

# Default values
MAX_SIM_STEPS=200000
MAX_LLM_CALLS=""  # Empty = use 3x golden trajectory length
EXPLORATION_STEPS=20
MECHANICS=""
TASK_FILE=""
LLM_AGENTS=""
NUM_TASKS=1
MODEL="gpt-5.2"
LLM_PROVIDER=""  # LLM provider: auto-detected from model
SUBTASKS_MIN=3
SUBTASKS_MAX=20
ITERATIONS_PER_TASK=200
AGENTS_MIN=2
AGENTS_MAX=10
AGENT_TYPE="robot"  # robot or human
QUERY=""  # seed query for task generation
THRESHOLD=0.7  # ToM judge threshold
RETRY_VERIFICATION=""  # Path to failed ToM verification file
NO_AUTO_RETRY=false  # Disable automatic retry on judge failure
CATEGORY=""  # Task category: cooperative, competitive, or mixed

print_usage() {
    echo -e "${BOLD}EMTOM Benchmark Pipeline${NC}"
    echo ""
    echo -e "Usage: ./emtom/run_emtom.sh ${YELLOW}<command>${NC} [options]"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo "  explore     Run LLM-guided exploration in Habitat"
    echo "  generate    Generate tasks iteratively with testing loop"
    echo "  benchmark   Run benchmark with generated tasks"
    echo "  test        Human-in-the-loop testing mode (manual command input)"
    echo "  judge       Evaluate task quality + ToM with multi-model council (Claude Opus + GPT-5)"
    echo "  all         Run full pipeline: explore -> generate -> benchmark"
    echo ""
    echo -e "${BOLD}Agent Options:${NC}"
    echo "  --agents N           Exact number of agents (sets both min and max, 2-10 for robots, 2-5 for humans)"
    echo "  --agents-min N       Minimum agents for task generation (default: $AGENTS_MIN)"
    echo "  --agents-max N       Maximum agents for task generation (default: $AGENTS_MAX)"
    echo "  --agent-type TYPE    Agent type: human or robot (default: $AGENT_TYPE)"
    echo ""
    echo -e "${BOLD}Exploration Options:${NC}"
    echo "  --steps N            Number of exploration steps (default: $EXPLORATION_STEPS)"
    echo "  --model MODEL        LLM model name (default: gpt-5.2, provider auto-detected)"
    echo "                       (Episodes are always randomly selected for diversity)"
    echo ""
    echo -e "${BOLD}Generation Options:${NC}"
    echo "  --num-tasks N        Number of tasks to generate (default: 1)"
    echo "  --model MODEL        LLM model name (default: gpt-5.2, provider auto-detected)"
    echo ""
    echo -e "  ${BOLD}Available Models:${NC}"
    echo -e "  ┌───────────────────────────┬────────────────────┐"
    echo -e "  │ ${BOLD}Model Name${NC}                │ ${BOLD}--model${NC}            │"
    echo -e "  ├───────────────────────────┼────────────────────┤"
    echo -e "  │ GPT-5                     │ ${GREEN}gpt-5${NC}              │"
    echo -e "  │ GPT-5 Mini                │ ${GREEN}gpt-5-mini${NC}         │"
    echo -e "  │ GPT-5.1                   │ ${GREEN}gpt-5.1${NC}            │"
    echo -e "  │ GPT-5.2 (default)         │ ${GREEN}gpt-5.2${NC}            │"
    echo -e "  ├───────────────────────────┼────────────────────┤"
    echo -e "  │ Claude Sonnet             │ ${GREEN}sonnet${NC}             │"
    echo -e "  │ Claude Haiku              │ ${GREEN}haiku${NC}              │"
    echo -e "  │ Claude Opus               │ ${GREEN}opus${NC}               │"
    echo -e "  ├───────────────────────────┼────────────────────┤"
    echo -e "  │ Kimi K2 Thinking          │ ${GREEN}kimi-k2-thinking${NC}   │"
    echo -e "  ├───────────────────────────┼────────────────────┤"
    echo -e "  │ Ministral 3 8B            │ ${GREEN}ministral-3-8b${NC}     │"
    echo -e "  │ Ministral 3 14B           │ ${GREEN}ministral-3-14b${NC}    │"
    echo -e "  │ Mistral Large 3           │ ${GREEN}mistral-large-3${NC}    │"
    echo -e "  ├───────────────────────────┼────────────────────┤"
    echo -e "  │ Qwen3 Next 80B            │ ${GREEN}qwen3-next-80b${NC}     │"
    echo -e "  │ Qwen3 VL 235B             │ ${GREEN}qwen3-vl-235b${NC}      │"
    echo -e "  └───────────────────────────┴────────────────────┘"
    echo ""
    echo "  --subtasks N         Exact number of subtasks per task (sets both min and max)"
    echo "  --subtasks-min N     Minimum subtasks per task (default: $SUBTASKS_MIN)"
    echo "  --subtasks-max N     Maximum subtasks per task (default: $SUBTASKS_MAX)"
    echo "  --iterations-per-task N   Max iterations per task (default: 100)"
    echo "  --query \"TEXT\"       Seed query to guide task generation (e.g., \"A task using the radio\")"
    echo "  --retry-verification FILE  Retry generation using suggestions from failed ToM verification"
    echo "  --category TYPE      Task category: cooperative, competitive, or mixed (default: random)"
    echo ""
    echo -e "${BOLD}Benchmark Options:${NC}"
    echo "  --max-sim-steps N    Max simulation steps before timeout (default: $MAX_SIM_STEPS)"
    echo "  --max-llm-calls N    Max LLM calls per agent (default: 3x golden trajectory)"
    echo ""
    echo -e "${BOLD}Test Options:${NC}"
    echo "  --mechanics M1 M2    Mechanics to enable (e.g., inverse_state remote_control)"
    echo "  --task FILE          Task file to load (uses task's mechanic bindings automatically)"
    echo "  --llm-agents A1 A2   Agents to make LLM-controlled (e.g., agent_1)"
    echo ""
    echo -e "${BOLD}Judge Options:${NC}"
    echo "  --task FILE          Task file to evaluate (required)"
    echo "  --model MODEL        LLM model name (default: gpt-5.2, provider auto-detected)"
    echo "  --threshold N        Overall score threshold for passing (default: 0.7)"
    echo "  --no-auto-retry      Disable automatic retry on failure (just show suggestions)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ./emtom/run_emtom.sh explore --steps 30 ${GREEN}--model gpt-5${NC}"
    echo -e "  ./emtom/run_emtom.sh explore --agents 3 ${GREEN}--model sonnet${NC}"
    echo -e "  ./emtom/run_emtom.sh generate ${GREEN}--model gpt-5${NC}"
    echo -e "  ./emtom/run_emtom.sh generate ${GREEN}--model sonnet${NC} --num-tasks 5"
    echo -e "  ./emtom/run_emtom.sh generate --agents-min 2 --agents-max 4 ${GREEN}--model gpt-5${NC}"
    echo -e "  ./emtom/run_emtom.sh generate ${GREEN}--model mistral-large-3${NC} --query \"A task using the radio\""
    echo "  ./emtom/run_emtom.sh all"
    echo "  ./emtom/run_emtom.sh benchmark --max-sim-steps 1000"
    echo "  ./emtom/run_emtom.sh test --mechanics inverse_state remote_control"
    echo -e "  ./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json"
}

# Get config name based on number of agents and type
get_agent_config() {
    local num_agents=$1
    local agent_type=$2

    if [ "$agent_type" = "human" ]; then
        case $num_agents in
            2) echo "examples/emtom_2_humans" ;;
            3) echo "examples/emtom_3_humans" ;;
            4) echo "examples/emtom_4_humans" ;;
            5) echo "examples/emtom_5_humans" ;;
            *)
                echo "Error: --agents must be 2-5 for humanoid agents (human configs only go up to 5)" >&2
                exit 1
                ;;
        esac
    else
        # Robot configs (support 2-10 agents)
        case $num_agents in
            2) echo "examples/emtom_2_robots" ;;
            3) echo "examples/emtom_3_robots" ;;
            4) echo "examples/emtom_4_robots" ;;
            5) echo "examples/emtom_5_robots" ;;
            6) echo "examples/emtom_6_robots" ;;
            7) echo "examples/emtom_7_robots" ;;
            8) echo "examples/emtom_8_robots" ;;
            9) echo "examples/emtom_9_robots" ;;
            10) echo "examples/emtom_10_robots" ;;
            *)
                echo "Error: --agents must be 2-10 for robot agents" >&2
                exit 1
                ;;
        esac
    fi
}

# Note: Headless configs removed - spawn positions are now cached in task.json
# First scene load calculates spawns (slow), subsequent loads reuse cached spawns (fast)

run_exploration() {
    # Auto-detect LLM provider if not specified
    if [ -z "$LLM_PROVIDER" ]; then
        LLM_PROVIDER=$(detect_llm_provider "$MODEL")
        if [ -z "$LLM_PROVIDER" ]; then
            echo -e "${RED}Error: Could not auto-detect provider for model '$MODEL'${NC}"
            print_llm_options
            exit 1
        fi
    fi

    # Expand short model names to full IDs
    MODEL=$(expand_model_name "$MODEL")

    echo "=============================================="
    echo "Running EMTOM Exploration (Habitat Backend)"
    echo "=============================================="
    echo "Mode: LLM-guided"
    echo "LLM: $LLM_PROVIDER ($MODEL)"
    echo "Steps: $EXPLORATION_STEPS"
    echo "Agents: $AGENTS_MAX ($AGENT_TYPE)"
    echo "(Episodes randomly selected for diversity)"
    echo "=============================================="
    echo ""

    # Get the appropriate config for the number of agents (use max for exploration)
    CONFIG_NAME=$(get_agent_config $AGENTS_MAX $AGENT_TYPE)

    # Use Hydra config system - pass parameters as config overrides
    python emtom/examples/run_habitat_exploration.py \
        --config-name $CONFIG_NAME \
        +exploration_steps=$EXPLORATION_STEPS \
        +model=$MODEL \
        +llm_provider=$LLM_PROVIDER \
        evaluation.save_video=true \
        "hydra.run.dir=./outputs/emtom/\${now:%Y-%m-%d_%H-%M-%S}-exploration"
}

run_generate() {
    # Auto-detect LLM provider if not specified
    if [ -z "$LLM_PROVIDER" ]; then
        LLM_PROVIDER=$(detect_llm_provider "$MODEL")
        if [ -z "$LLM_PROVIDER" ]; then
            echo -e "${RED}Error: Could not auto-detect provider for model '$MODEL'${NC}"
            print_llm_options
            exit 1
        fi
    fi

    # Expand short model names to full IDs
    MODEL=$(expand_model_name "$MODEL")

    # Get the appropriate config for the max number of agents
    # Note: Spawn positions are cached in task.json for fast subsequent loads
    CONFIG_NAME=$(get_agent_config $AGENTS_MAX $AGENT_TYPE)

    echo "=============================================="
    echo "Running EMTOM Task Generation (Live Scene Mode)"
    echo "=============================================="
    echo "Target tasks: $NUM_TASKS"
    echo "Agents: $AGENTS_MIN - $AGENTS_MAX ($AGENT_TYPE)"
    echo "LLM: $LLM_PROVIDER ($MODEL)"
    echo "Category: ${CATEGORY:-random}"
    echo "Subtasks: $SUBTASKS_MIN - $SUBTASKS_MAX"
    echo "Iterations per task: $ITERATIONS_PER_TASK"
    if [ -n "$QUERY" ]; then
        echo "Query: $QUERY"
    fi
    if [ -n "$RETRY_VERIFICATION" ]; then
        echo "Retry from: $RETRY_VERIFICATION"
    fi
    echo "(Loads random scene from PARTNR dataset)"
    echo "=============================================="
    echo ""

    # Build extra args array (handles spaces/quotes properly without eval)
    EXTRA_ARGS=()
    if [ -n "$QUERY" ]; then
        EXTRA_ARGS+=(--query "$QUERY")
    fi
    if [ -n "$RETRY_VERIFICATION" ]; then
        EXTRA_ARGS+=(--retry-verification "$RETRY_VERIFICATION")
    fi
    if [ -n "$CATEGORY" ]; then
        EXTRA_ARGS+=(--category "$CATEGORY")
    fi

    # Use Hydra config system with custom overrides
    # Scene is loaded live from PARTNR dataset - no trajectories needed
    python emtom/task_gen/runner.py \
        "${EXTRA_ARGS[@]}" \
        --config-name $CONFIG_NAME \
        +num_tasks=$NUM_TASKS \
        +agents_min=$AGENTS_MIN \
        +agents_max=$AGENTS_MAX \
        +model=$MODEL \
        +llm_provider=$LLM_PROVIDER \
        +subtasks_min=$SUBTASKS_MIN \
        +subtasks_max=$SUBTASKS_MAX \
        +iterations_per_task=$ITERATIONS_PER_TASK \
        +output_dir=data/emtom/tasks \
        "hydra.run.dir=./outputs/emtom/\${now:%Y-%m-%d_%H-%M-%S}-generate"
}

run_benchmark() {
    # Determine task file (specified or most recent)
    ACTUAL_TASK_FILE="$TASK_FILE"
    if [ -z "$ACTUAL_TASK_FILE" ]; then
        # Find most recent task file
        ACTUAL_TASK_FILE=$(ls -t data/emtom/tasks/*.json 2>/dev/null | head -1)
        if [ -z "$ACTUAL_TASK_FILE" ]; then
            echo -e "${RED}ERROR: No task files found in data/emtom/tasks/${NC}"
            echo "Run task generation first: ./emtom/run_emtom.sh generate"
            exit 1
        fi
    fi

    # Auto-detect num_agents from task file
    TASK_NUM_AGENTS=$(python3 -c "import json; print(json.load(open('$ACTUAL_TASK_FILE')).get('num_agents', 2))" 2>/dev/null)
    if [ -z "$TASK_NUM_AGENTS" ]; then
        TASK_NUM_AGENTS=2
    fi

    echo "=============================================="
    echo "Running EMTOM Habitat Benchmark"
    echo "=============================================="
    echo "Max simulation steps: $MAX_SIM_STEPS"
    if [ -n "$MAX_LLM_CALLS" ]; then
        echo "Max LLM calls per agent: $MAX_LLM_CALLS"
    else
        echo "Max LLM calls per agent: 3x golden trajectory"
    fi
    echo "Task file: $ACTUAL_TASK_FILE"
    echo "Agents: $TASK_NUM_AGENTS (from task file)"
    echo "=============================================="

    # Get the appropriate config for the number of agents (from task)
    CONFIG_NAME=$(get_agent_config $TASK_NUM_AGENTS $AGENT_TYPE)

    # Build optional overrides (only if MAX_LLM_CALLS is explicitly set)
    MAX_TURNS_OVERRIDE=""
    REPLANNING_OVERRIDES=""
    if [ -n "$MAX_LLM_CALLS" ]; then
        MAX_TURNS_OVERRIDE="+max_turns=$MAX_LLM_CALLS"
        for ((i=0; i<TASK_NUM_AGENTS; i++)); do
            REPLANNING_OVERRIDES="$REPLANNING_OVERRIDES ++evaluation.agents.agent_${i}.planner.plan_config.replanning_threshold=$MAX_LLM_CALLS"
        done
    fi

    python emtom/examples/run_habitat_benchmark.py \
        --config-name $CONFIG_NAME \
        habitat.environment.max_episode_steps=$MAX_SIM_STEPS \
        $MAX_TURNS_OVERRIDE \
        $REPLANNING_OVERRIDES \
        +task=$ACTUAL_TASK_FILE \
        "hydra.run.dir=./outputs/emtom/\${now:%Y-%m-%d_%H-%M-%S}-benchmark"
}

run_test() {
    # Auto-detect num_agents from task file if provided, otherwise use AGENTS_MAX
    TEST_NUM_AGENTS=$AGENTS_MAX
    if [ -n "$TASK_FILE" ] && [ -f "$TASK_FILE" ]; then
        TASK_NUM_AGENTS=$(python3 -c "import json; print(json.load(open('$TASK_FILE')).get('num_agents', 2))" 2>/dev/null)
        if [ -n "$TASK_NUM_AGENTS" ]; then
            TEST_NUM_AGENTS=$TASK_NUM_AGENTS
        fi
    fi

    echo "=============================================="
    echo "Running EMTOM Human Test Mode"
    echo "=============================================="
    echo "Mechanics: ${MECHANICS:-from task file or default}"
    echo "Task file: ${TASK_FILE:-none}"
    echo "LLM agents: ${LLM_AGENTS:-none (all human-controlled)}"
    echo "Agents: $TEST_NUM_AGENTS ($AGENT_TYPE)${TASK_FILE:+ (from task file)}"
    echo "=============================================="
    echo ""
    echo "Actions: Navigate[target], Open[target], Close[target], Pick[target], Place[target]"
    echo "         Search[container], UseItem[item, target], Communicate[message]"
    echo "Commands: status, mechanics, history, skip, quit, help"
    echo ""

    # Get the appropriate config for the number of agents
    CONFIG_NAME=$(get_agent_config $TEST_NUM_AGENTS $AGENT_TYPE)

    # Build command arguments
    CMD_ARGS=""
    if [ -n "$MECHANICS" ]; then
        CMD_ARGS="$CMD_ARGS --mechanics $MECHANICS"
    fi
    if [ -n "$TASK_FILE" ]; then
        CMD_ARGS="$CMD_ARGS --task $TASK_FILE"
    fi
    if [ -n "$LLM_AGENTS" ]; then
        CMD_ARGS="$CMD_ARGS --llm-agents $LLM_AGENTS"
    fi

    python emtom/examples/run_human_test.py \
        --config-name $CONFIG_NAME \
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

run_judge() {
    if [ -z "$TASK_FILE" ]; then
        echo "Error: --task is required for judge command"
        echo "Usage: ./emtom/run_emtom.sh judge --task <path_to_task.json>"
        exit 1
    fi

    echo "=============================================="
    echo "Running EMTOM Task Judge (Council)"
    echo "=============================================="
    echo "Task: $TASK_FILE"
    echo "Models: opus, gpt-5 (multi-model council)"
    echo "Threshold: $THRESHOLD"
    echo "=============================================="
    echo ""

    # Build command arguments
    CMD_ARGS="--task $TASK_FILE --models opus,gpt-5 --threshold $THRESHOLD"
    if [ "$VERBOSE" = true ]; then
        CMD_ARGS="$CMD_ARGS --verbose"
    fi

    python -m emtom.task_gen.judge $CMD_ARGS
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        explore|generate|benchmark|test|judge|all)
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
        --num-tasks|--tasks)
            NUM_TASKS=$2
            shift 2
            ;;
        --model)
            MODEL=$2
            shift 2
            ;;
        --llm)
            LLM_PROVIDER=$2
            shift 2
            ;;
        --subtasks)
            SUBTASKS_MIN=$2
            SUBTASKS_MAX=$2
            shift 2
            ;;
        --subtasks-min)
            SUBTASKS_MIN=$2
            shift 2
            ;;
        --subtasks-max)
            SUBTASKS_MAX=$2
            shift 2
            ;;
        --iterations-per-task)
            ITERATIONS_PER_TASK=$2
            shift 2
            ;;
        --query)
            QUERY=$2
            shift 2
            ;;
        --retry-verification)
            RETRY_VERIFICATION=$2
            shift 2
            ;;
        --agents)
            AGENTS_MIN=$2
            AGENTS_MAX=$2
            shift 2
            ;;
        --agents-min)
            AGENTS_MIN=$2
            shift 2
            ;;
        --agents-max)
            AGENTS_MAX=$2
            shift 2
            ;;
        --num-agents)
            # Backwards compatibility: --num-agents sets both min and max
            AGENTS_MIN=$2
            AGENTS_MAX=$2
            shift 2
            ;;
        --agent-type)
            AGENT_TYPE=$2
            if [[ "$AGENT_TYPE" != "human" && "$AGENT_TYPE" != "robot" ]]; then
                echo "Error: --agent-type must be 'human' or 'robot'"
                exit 1
            fi
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
        --threshold)
            THRESHOLD=$2
            shift 2
            ;;
        --no-auto-retry)
            NO_AUTO_RETRY=true
            shift
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
        --category)
            CATEGORY=$2
            if [[ "$CATEGORY" != "cooperative" && "$CATEGORY" != "competitive" && "$CATEGORY" != "mixed" ]]; then
                echo "Error: --category must be 'cooperative', 'competitive', or 'mixed'"
                exit 1
            fi
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
    explore)
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
    judge)
        run_judge
        ;;
    all)
        run_all
        ;;
esac

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
