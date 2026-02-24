#!/bin/bash
# Bulk EMTOM Task Generation
# Runs task generation across all GPUs with 3 processes per GPU
# Generates all 3 categories: cooperative, competitive, mixed
#
# Usage: ./emtom/bulk_generate.sh [options]
#
# Examples:
#   ./emtom/bulk_generate.sh                  # 24 processes (8 GPUs x 3)
#   ./emtom/bulk_generate.sh --per-gpu 2      # 16 processes (8 GPUs x 2)
#   ./emtom/bulk_generate.sh --model gpt-5    # Use different model
#   ./emtom/bulk_generate.sh --dry-run        # Preview without running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# All 3 task categories (round-robin across processes)
CATEGORIES=("cooperative" "competitive" "mixed")

# Defaults
PER_GPU=3
MODEL="gpt-5.2"
NUM_TASKS=3
ITERATIONS_PER_TASK=300  # Max iterations per task (total = this * num-tasks)
SUBTASKS_MIN=""
SUBTASKS_MAX=""
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_usage() {
    echo -e "${BOLD}Bulk EMTOM Task Generation${NC}"
    echo ""
    echo "Generates tasks across all GPUs (3 processes per GPU)"
    echo "All 3 categories covered: cooperative, competitive, mixed"
    echo ""
    echo "Usage: ./emtom/bulk_generate.sh [options]"
    echo ""
    echo "Options:"
    echo "  --per-gpu N         Processes per GPU (default: 3, one per category)"
    echo "  --model MODEL       LLM model (default: gpt-5.2)"
    echo "  --num-tasks N       Tasks per process (default: 1)"
    echo "  --iterations-per-task N  Max iterations per task (default: 100)"
    echo "  --subtasks-min N    Minimum subtasks per task"
    echo "  --subtasks-max N    Maximum subtasks per task"
    echo "  --dry-run           Show commands without executing"
    echo ""
    echo "Examples:"
    echo "  ./emtom/bulk_generate.sh                  # 24 processes (8 GPUs x 3)"
    echo "  ./emtom/bulk_generate.sh --per-gpu 6      # 48 processes (2 per category per GPU)"
    echo "  ./emtom/bulk_generate.sh --model gpt-5"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --per-gpu)
            PER_GPU=$2
            shift 2
            ;;
        --model)
            MODEL=$2
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS=$2
            shift 2
            ;;
        --iterations-per-task)
            ITERATIONS_PER_TASK=$2
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
        --dry-run)
            DRY_RUN=true
            shift
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

# Detect GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo -e "${RED}Error: No GPUs detected${NC}"
    exit 1
fi

NUM_CATEGORIES=${#CATEGORIES[@]}
TOTAL_PROCESSES=$((NUM_GPUS * PER_GPU))

# Create log and task directories
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="outputs/bulk_gen_logs/${TIMESTAMP}-bulk-generate"
TASK_DIR="data/emtom/tasks"
mkdir -p "$LOG_DIR"
mkdir -p "$TASK_DIR"

echo -e "${BOLD}=============================================="
echo -e "Bulk EMTOM Task Generation"
echo -e "==============================================${NC}"
echo -e "GPUs:               ${GREEN}$NUM_GPUS${NC}"
echo -e "Processes per GPU:  ${GREEN}$PER_GPU${NC}"
echo -e "Total processes:    ${GREEN}$TOTAL_PROCESSES${NC}"
echo -e "Categories:         ${GREEN}${CATEGORIES[*]}${NC} (all 3)"
echo -e "Model:              ${GREEN}$MODEL${NC}"
echo -e "Tasks per process:  ${GREEN}$NUM_TASKS${NC}"
echo -e "Iterations/task:    ${GREEN}$ITERATIONS_PER_TASK${NC}"
[ -n "$SUBTASKS_MIN" ] && echo -e "Subtasks min:       ${GREEN}$SUBTASKS_MIN${NC}"
[ -n "$SUBTASKS_MAX" ] && echo -e "Subtasks max:       ${GREEN}$SUBTASKS_MAX${NC}"
echo -e "Task directory:     ${CYAN}$TASK_DIR${NC}"
echo -e "Log directory:      ${CYAN}$LOG_DIR${NC}"
echo "=============================================="
echo ""

# Track PIDs
declare -a PIDS
declare -a PROCESS_INFO

process_idx=0

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    for slot in $(seq 0 $((PER_GPU - 1))); do
        # Round-robin category assignment
        category_idx=$((process_idx % NUM_CATEGORIES))
        category=${CATEGORIES[$category_idx]}

        log_file="$LOG_DIR/gpu${gpu}_slot${slot}_${category}.log"

        # Build subtask flags
        SUBTASK_FLAGS=""
        [ -n "$SUBTASKS_MIN" ] && SUBTASK_FLAGS="$SUBTASK_FLAGS --subtasks-min $SUBTASKS_MIN"
        [ -n "$SUBTASKS_MAX" ] && SUBTASK_FLAGS="$SUBTASK_FLAGS --subtasks-max $SUBTASKS_MAX"

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN]${NC} GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC}"
            echo "  CUDA_VISIBLE_DEVICES=$gpu ./emtom/run_emtom.sh generate --model $MODEL --num-tasks $NUM_TASKS --iterations-per-task $ITERATIONS_PER_TASK --category $category --output-dir $TASK_DIR$SUBTASK_FLAGS"
        else
            echo -e "${GREEN}Starting${NC} GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC} -> $log_file"

            CUDA_VISIBLE_DEVICES=$gpu ./emtom/run_emtom.sh generate \
                --model "$MODEL" \
                --num-tasks "$NUM_TASKS" \
                --iterations-per-task "$ITERATIONS_PER_TASK" \
                --category "$category" \
                --output-dir "$TASK_DIR" \
                $SUBTASK_FLAGS \
                > "$log_file" 2>&1 &

            pid=$!
            PIDS+=($pid)
            PROCESS_INFO+=("GPU$gpu:$category:$pid")
        fi

        ((++process_idx))
    done
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}Dry run complete. No processes started.${NC}"
    exit 0
fi

echo ""
echo "=============================================="
echo -e "${BOLD}All $TOTAL_PROCESSES processes started${NC}"
echo "=============================================="
echo ""
echo -e "${CYAN}Monitoring:${NC}"
echo "  watch -n 1 nvidia-smi          # GPU usage"
echo "  tail -f $LOG_DIR/*.log         # All logs"
echo "  pkill -f 'run_emtom.sh gen'    # Kill all"
echo ""

# Wait for all processes
echo -e "${BOLD}Waiting for completion...${NC}"
echo ""

failed=0
succeeded=0

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    info=${PROCESS_INFO[$i]}

    if wait $pid; then
        echo -e "${GREEN}[DONE]${NC} $info"
        ((succeeded++))
    else
        echo -e "${RED}[FAIL]${NC} $info"
        ((failed++))
    fi
done

echo ""
echo "=============================================="
echo -e "${BOLD}Summary${NC}"
echo "=============================================="
echo -e "Succeeded: ${GREEN}$succeeded${NC}"
echo -e "Failed:    ${RED}$failed${NC}"
echo -e "Logs:      $LOG_DIR/"
echo -e "Tasks:     $TASK_DIR/"
echo "=============================================="

# Count saved task files in the output directory
task_count=$(find "$TASK_DIR" -name '*.json' 2>/dev/null | wc -l)

if [ "$task_count" -gt 0 ]; then
    echo ""
    echo -e "${BOLD}${GREEN}=============================================="
    echo -e "Saved Tasks ($task_count)"
    echo -e "==============================================${NC}"
    for path in "$TASK_DIR"/*.json; do
        echo -e "  ${CYAN}${BOLD}$path${NC}"
    done
    echo -e "${BOLD}${GREEN}==============================================${NC}"
else
    echo ""
    echo -e "${YELLOW}No tasks were saved during this run.${NC}"
fi
