#!/bin/bash
# Bulk EMTOM Task Generation
# Runs task generation across all GPUs with configurable concurrency
# and keeps launching attempts until the requested total is reached
# or a full batch fails.
#
# Usage: ./emtom/bulk_generate.sh [options] [-- <run_emtom.sh args>]
#
# Examples:
#   ./emtom/bulk_generate.sh                              # 24 processes (8 GPUs x 3)
#   ./emtom/bulk_generate.sh --per-gpu 2                  # 16 concurrent slots (8 GPUs x 2)
#   ./emtom/bulk_generate.sh --total-tasks 4              # Keep going until 4 tasks or a full batch fails
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
TOTAL_TASKS=""
DRY_RUN=false
CATEGORY_FILTER=""  # Empty = all 3 categories (round-robin)
OUTPUT_DIR="data/emtom/tasks"
TASK_GEN_AGENT="mini"
EXTRA_ARGS=()  # Extra args forwarded verbatim to run_emtom.sh generate
DIFFICULTY=""
K_LEVEL=""  # Allowed k-levels (e.g. "2 3"). Empty = random per task.
K_DISTRIBUTION=""  # Slot distribution (e.g. "1:2,2:3,3:3"). Overrides --k-level.

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
    echo "Generates tasks across all GPUs with configurable concurrency"
    echo "All 3 categories covered: cooperative, competitive, mixed"
    echo ""
    echo "Usage: ./emtom/bulk_generate.sh [options] [-- <run_emtom.sh args>]"
    echo ""
    echo "Options:"
    echo "  --per-gpu N         Concurrent processes per GPU (default: 3, one per category)"
    echo "  --model MODEL       LLM model (default: gpt-5.2)"
    echo "  --num-tasks N       Tasks per process (default: 3)"
    echo "  --total-tasks N     Keep launching one-task attempts until N tasks are submitted"
    echo "                      or a full batch fails"
    echo "  --task-gen-agent A  External generator agent: mini|claude|codex (default: mini)"
    echo "  --category CAT      Only generate this category (cooperative, competitive, mixed)"
    echo "  --difficulty LEVEL  Difficulty for generation (easy, medium, hard)"
    echo "  --k-level L [L ...] Allowed k-levels, e.g. --k-level 2 3 (default: random per task)"
    echo "  --k-distribution D  Slots per k-level, e.g. 1:2,2:3,3:3 = 2 slots K=1, 3 K=2, 3 K=3"
    echo "                      Slot counts must sum to --per-gpu. Overrides --k-level."
    echo "  --output-dir DIR    Output directory for submitted tasks (default: data/emtom/tasks)"
    echo "  --dry-run           Show commands without executing"
    echo ""
    echo "Everything after -- is forwarded to run_emtom.sh generate:"
    echo "  --difficulty, --subtasks-*, --tom-target-*, --iterations-per-task, etc."
    echo "  See: ./emtom/run_emtom.sh generate --help"
    echo ""
    echo "Examples:"
    echo "  ./emtom/bulk_generate.sh                                  # 24 concurrent slots (8 GPUs x 3)"
    echo "  ./emtom/bulk_generate.sh --per-gpu 6                      # 48 concurrent slots"
    echo "  ./emtom/bulk_generate.sh --total-tasks 4                  # stop after 4 tasks or a full failed batch"
    echo "  ./emtom/bulk_generate.sh --per-gpu 8 --k-distribution 1:2,2:3,3:3  # Weighted K-levels"
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
        --total-tasks)
            TOTAL_TASKS=$2
            shift 2
            ;;
        --task-gen-agent)
            TASK_GEN_AGENT=$2
            if [[ "$TASK_GEN_AGENT" != "mini" && "$TASK_GEN_AGENT" != "claude" && "$TASK_GEN_AGENT" != "codex" ]]; then
                echo "Error: --task-gen-agent must be 'mini', 'claude', or 'codex'"
                exit 1
            fi
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
        --k-distribution)
            K_DISTRIBUTION=$2
            shift 2
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

# Build per-slot k-level array from --k-distribution
declare -a SLOT_K_LEVELS=()
if [ -n "$K_DISTRIBUTION" ]; then
    if [ -n "$K_LEVEL" ]; then
        echo -e "${RED}Error: --k-distribution and --k-level are mutually exclusive${NC}"
        exit 1
    fi
    # Parse "1:2,2:3,3:3" into SLOT_K_LEVELS array
    IFS=',' read -ra K_PAIRS <<< "$K_DISTRIBUTION"
    for pair in "${K_PAIRS[@]}"; do
        k_val="${pair%%:*}"
        k_count="${pair##*:}"
        if ! [[ "$k_val" =~ ^[123]$ ]] || ! [[ "$k_count" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}Error: --k-distribution entries must be K:COUNT where K is 1-3 (got '$pair')${NC}"
            exit 1
        fi
        for _ in $(seq 1 "$k_count"); do
            SLOT_K_LEVELS+=("$k_val")
        done
    done
    # Validate total matches --per-gpu
    if [ "${#SLOT_K_LEVELS[@]}" -ne "$PER_GPU" ]; then
        echo -e "${RED}Error: --k-distribution slot counts sum to ${#SLOT_K_LEVELS[@]} but --per-gpu is $PER_GPU${NC}"
        exit 1
    fi
fi

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
MAX_PROCESSES=$((NUM_GPUS * PER_GPU))
TOTAL_PROCESSES=$MAX_PROCESSES

if [ -n "$TOTAL_TASKS" ]; then
    if ! [[ "$TOTAL_TASKS" =~ ^[0-9]+$ ]] || [ "$TOTAL_TASKS" -le 0 ]; then
        echo -e "${RED}Error: --total-tasks must be a positive integer${NC}"
        exit 1
    fi
fi

if [ -n "$TOTAL_TASKS" ] && [ "$TOTAL_TASKS" -lt "$TOTAL_PROCESSES" ]; then
    INITIAL_BATCH_SIZE="$TOTAL_TASKS"
else
    INITIAL_BATCH_SIZE="$TOTAL_PROCESSES"
fi

# Create generation/task directories
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
GENERATION_RUN_ID="${TIMESTAMP}-generation"
GENERATION_DIR="outputs/generations/${GENERATION_RUN_ID}"
LOG_DIR="${GENERATION_DIR}/logs"
WORKERS_DIR="${GENERATION_DIR}/workers"
TASK_DIR="$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$WORKERS_DIR"
mkdir -p "$TASK_DIR"
LAUNCHER_LOG="${GENERATION_DIR}/launcher.log"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

echo -e "${BOLD}=============================================="
echo -e "Bulk EMTOM Task Generation"
echo -e "==============================================${NC}"
echo -e "GPUs:               ${GREEN}$NUM_GPUS${NC}"
echo -e "Processes per GPU:  ${GREEN}$PER_GPU${NC}"
echo -e "Parallel slots:     ${GREEN}$MAX_PROCESSES${NC}"
echo -e "Categories:         ${GREEN}${CATEGORIES[*]}${NC}"
echo -e "Model:              ${GREEN}$MODEL${NC}"
echo -e "Task-gen agent:     ${GREEN}$TASK_GEN_AGENT${NC}"
[ -n "$DIFFICULTY" ] && echo -e "Difficulty:         ${GREEN}$DIFFICULTY${NC}"
if [ -n "$K_DISTRIBUTION" ]; then
    echo -e "K-distribution:     ${GREEN}${K_DISTRIBUTION}${NC} (per GPU: ${SLOT_K_LEVELS[*]})"
elif [ -n "$K_LEVEL" ]; then
    echo -e "K-level:            ${GREEN}${K_LEVEL}${NC}"
else
    echo -e "K-level:            ${GREEN}random per task${NC}"
fi
if [ -n "$TOTAL_TASKS" ]; then
    echo -e "Total tasks:        ${GREEN}$TOTAL_TASKS${NC}"
    echo -e "Tasks per attempt:  ${GREEN}1${NC}"
    echo -e "Initial batch size: ${GREEN}$INITIAL_BATCH_SIZE${NC}"
else
    echo -e "Total processes:    ${GREEN}$TOTAL_PROCESSES${NC}"
    echo -e "Tasks per process:  ${GREEN}$NUM_TASKS${NC}"
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo -e "Extra args:         ${GREEN}${EXTRA_ARGS[*]}${NC}"
fi
echo -e "Task directory:     ${CYAN}$TASK_DIR${NC}"
echo -e "Generation dir:     ${CYAN}$GENERATION_DIR${NC}"
echo -e "Log directory:      ${CYAN}$LOG_DIR${NC}"
echo "=============================================="
echo ""

# Track PIDs
declare -a PIDS
declare -a PROCESS_INFO
REQUESTED_TASKS_TOTAL="${TOTAL_TASKS:-$((TOTAL_PROCESSES * NUM_TASKS))}"
ATTEMPT_COUNTER=0

build_worker_command() {
    local gpu="$1"
    local slot="$2"
    local category="$3"
    local process_num_tasks="$4"
    local worker_id="$5"
    local worker_dir="$6"
    local log_file="$7"

    local -a cmd=(
        ./emtom/run_emtom.sh generate
        --task-gen-agent "$TASK_GEN_AGENT"
        --model "$MODEL"
        --num-tasks "$process_num_tasks"
        --category "$category"
        --output-dir "$TASK_DIR"
    )
    if [ -n "$DIFFICULTY" ]; then
        cmd+=(--difficulty "$DIFFICULTY")
    fi
    if [ "${#SLOT_K_LEVELS[@]}" -gt 0 ]; then
        cmd+=(--k-level "${SLOT_K_LEVELS[$slot]}")
    elif [ -n "$K_LEVEL" ]; then
        # shellcheck disable=SC2206
        local k_parts=($K_LEVEL)
        cmd+=(--k-level "${k_parts[@]}")
    fi
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    if [ "$DRY_RUN" = true ]; then
        printf '  EMTOM_GENERATION_RUN_ID=%q EMTOM_GENERATION_RUN_DIR=%q EMTOM_GENERATION_WORKER_ID=%q EMTOM_GENERATION_WORKER_DIR=%q EMTOM_GENERATION_TOTAL_WORKERS=%q EMTOM_GENERATION_REQUESTED_TASKS=%q EMTOM_GENERATION_MODE=%q EMTOM_GENERATION_GPU=%q EMTOM_GENERATION_SLOT=%q EMTOM_GENERATION_STDOUT_LOG=%q CUDA_VISIBLE_DEVICES=%q' \
            "$GENERATION_RUN_ID" "$GENERATION_DIR" "$worker_id" "$worker_dir" "$MAX_PROCESSES" "$REQUESTED_TASKS_TOTAL" "bulk" "$gpu" "$slot" "$log_file" "$gpu"
        printf ' %q' "${cmd[@]}"
        printf '\n'
        return
    fi

    EMTOM_GENERATION_RUN_ID="$GENERATION_RUN_ID" \
    EMTOM_GENERATION_RUN_DIR="$GENERATION_DIR" \
    EMTOM_GENERATION_WORKER_ID="$worker_id" \
    EMTOM_GENERATION_WORKER_DIR="$worker_dir" \
    EMTOM_GENERATION_TOTAL_WORKERS="$MAX_PROCESSES" \
    EMTOM_GENERATION_REQUESTED_TASKS="$REQUESTED_TASKS_TOTAL" \
    EMTOM_GENERATION_MODE="bulk" \
    EMTOM_GENERATION_GPU="$gpu" \
    EMTOM_GENERATION_SLOT="$slot" \
    EMTOM_GENERATION_STDOUT_LOG="$log_file" \
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    if [ "${#SLOT_K_LEVELS[@]}" -gt 0 ]; then
        PROCESS_INFO+=("GPU$gpu:slot$slot:$category:K${SLOT_K_LEVELS[$slot]}:tasks=${process_num_tasks}:attempt=$(printf '%04d' "$ATTEMPT_COUNTER"):$pid")
    else
        PROCESS_INFO+=("GPU$gpu:slot$slot:$category:tasks=${process_num_tasks}:attempt=$(printf '%04d' "$ATTEMPT_COUNTER"):$pid")
    fi
}

launch_batch() {
    local batch_size="$1"
    local process_num_tasks="$2"
    local batch_label="$3"

    PIDS=()
    PROCESS_INFO=()

    for ((process_idx=0; process_idx<batch_size; process_idx++)); do
        local gpu=$((process_idx % NUM_GPUS))
        local slot=$((process_idx / NUM_GPUS))
        local category_idx=$((process_idx % NUM_CATEGORIES))
        local category=${CATEGORIES[$category_idx]}

        ATTEMPT_COUNTER=$((ATTEMPT_COUNTER + 1))
        local attempt_tag
        attempt_tag=$(printf '%04d' "$ATTEMPT_COUNTER")
        local worker_id="gpu${gpu}-slot${slot}-${category}-attempt${attempt_tag}"
        local worker_dir="${WORKERS_DIR}/${worker_id}"
        mkdir -p "$worker_dir"

        local log_file
        if [ "${#SLOT_K_LEVELS[@]}" -gt 0 ]; then
            local slot_k="${SLOT_K_LEVELS[$slot]}"
            log_file="$LOG_DIR/gpu${gpu}_slot${slot}_${category}_k${slot_k}_attempt${attempt_tag}.log"
        else
            log_file="$LOG_DIR/gpu${gpu}_slot${slot}_${category}_attempt${attempt_tag}.log"
        fi

        local slot_label="GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC}, Tasks=${process_num_tasks}, Attempt=${attempt_tag}"
        if [ "${#SLOT_K_LEVELS[@]}" -gt 0 ]; then
            slot_label="GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC}, K=${SLOT_K_LEVELS[$slot]}, Tasks=${process_num_tasks}, Attempt=${attempt_tag}"
        fi

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN]${NC} ${batch_label} $slot_label"
        else
            echo -e "${GREEN}Starting${NC} ${batch_label} $slot_label -> $log_file"
        fi
        build_worker_command "$gpu" "$slot" "$category" "$process_num_tasks" "$worker_id" "$worker_dir" "$log_file"
    done
}

BULK_START_EPOCH=$(date +%s)

if [ -n "$TOTAL_TASKS" ]; then
    launch_batch "$INITIAL_BATCH_SIZE" 1 "Batch 1 |"
    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo -e "${YELLOW}Dry run complete. In live mode the launcher keeps refilling slots with one-task attempts until ${TOTAL_TASKS} tasks are submitted or a full batch fails.${NC}"
        exit 0
    fi
else
    launch_batch "$TOTAL_PROCESSES" "$NUM_TASKS" ""
    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo -e "${YELLOW}Dry run complete. No processes started.${NC}"
        exit 0
    fi
fi

echo ""
echo "=============================================="
if [ -n "$TOTAL_TASKS" ]; then
    echo -e "${BOLD}Scheduler started${NC} (target ${TOTAL_TASKS} tasks, up to ${MAX_PROCESSES} concurrent attempts)"
else
    echo -e "${BOLD}All $TOTAL_PROCESSES processes started${NC}"
fi
echo "=============================================="
echo ""
echo -e "${CYAN}Monitoring:${NC}"
echo "  watch -n 1 nvidia-smi          # GPU usage"
echo "  tail -f $LOG_DIR/*.log         # Worker stdout logs"
echo "  pkill -9 -f 'emtom/task_gen/runner.py'  # Kill all"
echo ""

echo -e "${BOLD}Waiting for completion...${NC}"
echo ""

failed=0
succeeded=0
submitted_total=0
batches_run=0
stop_reason=""

wait_for_batch() {
    local batch_success=0
    local batch_failed=0

    for i in "${!PIDS[@]}"; do
        local pid=${PIDS[$i]}
        local info=${PROCESS_INFO[$i]}

        if wait "$pid"; then
            echo -e "${GREEN}[DONE]${NC} $info"
            batch_success=$((batch_success + 1))
            succeeded=$((succeeded + 1))
        else
            echo -e "${RED}[FAIL]${NC} $info"
            batch_failed=$((batch_failed + 1))
            failed=$((failed + 1))
        fi
    done

    submitted_total=$((submitted_total + batch_success))
    batches_run=$((batches_run + 1))
    LAST_BATCH_SUCCESS="$batch_success"
    LAST_BATCH_FAILED="$batch_failed"
}

if [ -n "$TOTAL_TASKS" ]; then
    while true; do
        wait_for_batch
        echo -e "${CYAN}Batch ${batches_run} summary:${NC} submitted ${LAST_BATCH_SUCCESS}, failed ${LAST_BATCH_FAILED}, total submitted ${submitted_total}/${TOTAL_TASKS}"

        if [ "$submitted_total" -ge "$TOTAL_TASKS" ]; then
            stop_reason="reached target"
            break
        fi
        if [ "$LAST_BATCH_SUCCESS" -eq 0 ]; then
            stop_reason="all attempts in the latest batch failed"
            break
        fi

        remaining_tasks=$((TOTAL_TASKS - submitted_total))
        next_batch_size=$MAX_PROCESSES
        if [ "$remaining_tasks" -lt "$next_batch_size" ]; then
            next_batch_size="$remaining_tasks"
        fi
        launch_batch "$next_batch_size" 1 "Batch $((batches_run + 1)) |"
    done
else
    wait_for_batch
    stop_reason="all processes completed"
fi

BULK_END_EPOCH=$(date +%s)
WALL_CLOCK=$((BULK_END_EPOCH - BULK_START_EPOCH))

echo ""
echo "=============================================="
if [ -n "$TOTAL_TASKS" ]; then
    echo -e "${BOLD}Bulk generation complete${NC} (${submitted_total}/${TOTAL_TASKS} tasks, ${succeeded} successful attempts, ${failed} failed attempts)"
    echo -e "Stop reason: ${CYAN}${stop_reason}${NC}"
else
    echo -e "${BOLD}Processes complete${NC} (${succeeded} ok, ${failed} failed)"
fi
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
