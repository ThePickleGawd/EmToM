#!/bin/bash
# Bulk EMTOM Task Generation
# Runs task generation across all GPUs with 3 processes per GPU
# Generates all 3 categories: cooperative, competitive, mixed
#
# Usage: ./emtom/bulk_generate.sh [options] [-- <run_emtom.sh args>]
#
# Examples:
#   ./emtom/bulk_generate.sh                              # 24 processes (8 GPUs x 3)
#   ./emtom/bulk_generate.sh --per-gpu 2                  # 16 processes (8 GPUs x 2)
#   ./emtom/bulk_generate.sh --model gpt-5                # Use different model
#   ./emtom/bulk_generate.sh --dry-run                    # Preview without running
#   ./emtom/bulk_generate.sh --per-gpu 5 -- --difficulty hard  # Difficulty preset
#   ./emtom/bulk_generate.sh -- --difficulty hard --subtasks-max 20  # Override preset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

mkdir -p outputs
LOCK_FILE="outputs/.bulk_generate.lock"

cleanup_lock() {
    rm -f "$LOCK_FILE"
}

if [ -f "$LOCK_FILE" ]; then
    existing_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        echo -e "${RED}Error: bulk generation already running (pid=$existing_pid).${NC}"
        echo "If that process is stale, stop it and remove $LOCK_FILE."
        exit 1
    fi
    rm -f "$LOCK_FILE"
fi

echo "$$" > "$LOCK_FILE"
trap cleanup_lock EXIT INT TERM

# Defaults
PER_GPU=3
MODEL="gpt-5.2"
NUM_TASKS=3
DRY_RUN=false
CATEGORY_FILTER=""  # Empty = all 3 categories (round-robin)
OUTPUT_DIR="data/emtom/tasks"
EXTRA_ARGS=()  # Extra args forwarded verbatim to run_emtom.sh generate
DIFFICULTY=""
K_LEVEL=""  # Allowed k-levels (e.g. "2 3"). Empty = random per task.

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
    echo "Usage: ./emtom/bulk_generate.sh [options] [-- <run_emtom.sh args>]"
    echo ""
    echo "Options:"
    echo "  --per-gpu N         Processes per GPU (default: 3, one per category)"
    echo "  --model MODEL       LLM model (default: gpt-5.2)"
    echo "  --num-tasks N       Tasks per process (default: 3)"
    echo "  --category CAT      Only generate this category (cooperative, competitive, mixed)"
    echo "  --difficulty LEVEL  Difficulty for generation (easy, medium, hard)"
    echo "  --k-level L [L ...] Allowed k-levels, e.g. --k-level 2 3 (default: random per task)"
    echo "  --output-dir DIR    Output directory for submitted tasks (default: data/emtom/tasks)"
    echo "  --dry-run           Show commands without executing"
    echo ""
    echo "Everything after -- is forwarded to run_emtom.sh generate:"
    echo "  --difficulty, --subtasks-*, --tom-target-*, --iterations-per-task, etc."
    echo "  See: ./emtom/run_emtom.sh generate --help"
    echo ""
    echo "Examples:"
    echo "  ./emtom/bulk_generate.sh                                  # 24 processes (8 GPUs x 3)"
    echo "  ./emtom/bulk_generate.sh --per-gpu 6                      # 48 processes"
    echo "  ./emtom/bulk_generate.sh --per-gpu 5 -- --difficulty hard  # Hard presets"
    echo "  ./emtom/bulk_generate.sh -- --difficulty hard --subtasks-max 20  # Override preset"
}

# Parse arguments — everything after -- goes to EXTRA_ARGS
while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
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
        --category)
            CATEGORY_FILTER=$2
            shift 2
            ;;
        --difficulty)
            DIFFICULTY=$2
            if [[ "$DIFFICULTY" != "easy" && "$DIFFICULTY" != "medium" && "$DIFFICULTY" != "hard" ]]; then
                echo "Error: --difficulty must be 'easy', 'medium', or 'hard'"
                exit 1
            fi
            shift 2
            ;;
        --k-level)
            shift
            K_LEVEL=""
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                K_LEVEL="$K_LEVEL $1"
                shift
            done
            K_LEVEL="${K_LEVEL# }"
            if [ -z "$K_LEVEL" ]; then
                echo "Error: --k-level requires at least one integer (1, 2, or 3)"
                exit 1
            fi
            ;;
        --output-dir)
            OUTPUT_DIR=$2
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
            echo "Unknown option: $1 (use -- to pass args to run_emtom.sh)"
            print_usage
            exit 1
            ;;
    esac
done

# Build category list
if [ -n "$CATEGORY_FILTER" ]; then
    CATEGORIES=("$CATEGORY_FILTER")
else
    CATEGORIES=("cooperative" "competitive" "mixed")
fi

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
TASK_DIR="$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TASK_DIR"

echo -e "${BOLD}=============================================="
echo -e "Bulk EMTOM Task Generation"
echo -e "==============================================${NC}"
echo -e "GPUs:               ${GREEN}$NUM_GPUS${NC}"
echo -e "Processes per GPU:  ${GREEN}$PER_GPU${NC}"
echo -e "Total processes:    ${GREEN}$TOTAL_PROCESSES${NC}"
echo -e "Categories:         ${GREEN}${CATEGORIES[*]}${NC}"
echo -e "Model:              ${GREEN}$MODEL${NC}"
[ -n "$DIFFICULTY" ] && echo -e "Difficulty:         ${GREEN}$DIFFICULTY${NC}"
[ -n "$K_LEVEL" ] && echo -e "K-level:            ${GREEN}${K_LEVEL}${NC}" || echo -e "K-level:            ${GREEN}random per task${NC}"
echo -e "Tasks per process:  ${GREEN}$NUM_TASKS${NC}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo -e "Extra args:         ${GREEN}${EXTRA_ARGS[*]}${NC}"
fi
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

        # Build flags for first-class options
        DIFFICULTY_FLAGS=""
        [ -n "$DIFFICULTY" ] && DIFFICULTY_FLAGS="--difficulty $DIFFICULTY"
        K_LEVEL_FLAGS=""
        [ -n "$K_LEVEL" ] && K_LEVEL_FLAGS="--k-level $K_LEVEL"

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN]${NC} GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC}"
            echo "  CUDA_VISIBLE_DEVICES=$gpu ./emtom/run_emtom.sh generate --model $MODEL --num-tasks $NUM_TASKS --category $category --output-dir $TASK_DIR $DIFFICULTY_FLAGS $K_LEVEL_FLAGS ${EXTRA_ARGS[*]}"
        else
            echo -e "${GREEN}Starting${NC} GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC} -> $log_file"

            CUDA_VISIBLE_DEVICES=$gpu ./emtom/run_emtom.sh generate \
                --model "$MODEL" \
                --num-tasks "$NUM_TASKS" \
                --category "$category" \
                --output-dir "$TASK_DIR" \
                $DIFFICULTY_FLAGS \
                $K_LEVEL_FLAGS \
                "${EXTRA_ARGS[@]}" \
                > "$log_file" 2>&1 &

            pid=$!
            PIDS+=($pid)
            PROCESS_INFO+=("GPU$gpu:$category:$pid")
        fi

        process_idx=$((process_idx + 1))
    done
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}Dry run complete. No processes started.${NC}"
    exit 0
fi

BULK_START_EPOCH=$(date +%s)

echo ""
echo "=============================================="
echo -e "${BOLD}All $TOTAL_PROCESSES processes started${NC}"
echo "=============================================="
echo ""
echo -e "${CYAN}Monitoring:${NC}"
echo "  watch -n 1 nvidia-smi          # GPU usage"
echo "  tail -f $LOG_DIR/*.log         # All logs"
echo "  pkill -9 -f 'emtom/task_gen/runner.py'  # Kill all"
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
        succeeded=$((succeeded + 1))
    else
        echo -e "${RED}[FAIL]${NC} $info"
        failed=$((failed + 1))
    fi
done

BULK_END_EPOCH=$(date +%s)
WALL_CLOCK=$((BULK_END_EPOCH - BULK_START_EPOCH))

echo ""
echo "=============================================="
echo -e "${BOLD}Processes complete${NC} (${succeeded} ok, ${failed} failed)"
echo "=============================================="
echo ""

# Accumulate log dirs for multi-run reports.
# When running bulk_generate.sh multiple times in series (e.g., from a wrapper
# or loop), set BULK_LOG_DIRS=dir1:dir2:... to get a combined report at the end.
# The last invocation's report will include ALL accumulated dirs.
if [ -n "$BULK_LOG_DIRS" ]; then
    export BULK_LOG_DIRS="${BULK_LOG_DIRS}:${LOG_DIR}"
else
    export BULK_LOG_DIRS="$LOG_DIR"
fi

# Build the log dir args for the report
IFS=':' read -ra LOG_DIR_ARGS <<< "$BULK_LOG_DIRS"

# Use accumulated wall clock from wrapper, or just this run's duration
if [ -n "$BULK_WALL_CLOCK" ]; then
    REPORT_WALL_CLOCK="$BULK_WALL_CLOCK"
else
    REPORT_WALL_CLOCK="$WALL_CLOCK"
fi

python -m emtom.bulk_report "${LOG_DIR_ARGS[@]}" --wall-clock "$REPORT_WALL_CLOCK"
