#!/bin/bash
# Bulk EMTOM Task Generation
# Runs task generation across all GPUs with configurable concurrency.
# Each worker generates exactly 1 task then exits. On success, the slot
# respawns a fresh worker (clean context). On failure, the slot stops.
# Continues until the target number of new tasks is reached.
#
# Usage: ./emtom/bulk_generate.sh [options] [-- <run_emtom.sh args>]
#
# Examples:
#   ./emtom/bulk_generate.sh                              # One full saturated run
#   ./emtom/bulk_generate.sh --per-gpu 2                  # 16 fixed workers (8 GPUs x 2)
#   ./emtom/bulk_generate.sh --num-tasks 4                # Produce 4 total tasks with fixed workers
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
QUEUE_DIR="outputs/.bulk_queue"
SHUTDOWN_IN_PROGRESS=false
EXIT_STATUS=0
QUEUE_TICKET=""

mkdir -p "$QUEUE_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

cleanup_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local holder_pid
        holder_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
        if [ "$holder_pid" = "$$" ]; then
            rm -f "$LOCK_FILE"
        fi
    fi
    # Remove our queue ticket if it exists
    if [ -n "$QUEUE_TICKET" ] && [ -f "$QUEUE_TICKET" ]; then
        rm -f "$QUEUE_TICKET"
    fi
}

kill_worker_groups() {
    if [ "${#PROCESS_GROUPS[@]}" -eq 0 ]; then
        return
    fi

    local pgid
    local sent_signal=false
    for pgid in "${PROCESS_GROUPS[@]}"; do
        if [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null; then
            if [ "$sent_signal" = false ]; then
                echo -e "${YELLOW}Stopping bulk worker processes...${NC}"
                sent_signal=true
            fi
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
    done

    if [ "$sent_signal" = true ]; then
        sleep 2
        for pgid in "${PROCESS_GROUPS[@]}"; do
            if [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null; then
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
        done
    fi
}

cleanup_all() {
    if [ "$SHUTDOWN_IN_PROGRESS" = true ]; then
        return
    fi
    SHUTDOWN_IN_PROGRESS=true
    kill_worker_groups
    cleanup_lock
}

handle_signal() {
    local signal_name="$1"
    case "$signal_name" in
        INT) EXIT_STATUS=130 ;;
        TERM) EXIT_STATUS=143 ;;
        *) EXIT_STATUS=1 ;;
    esac
    echo -e "${YELLOW}Received ${signal_name}. Shutting down bulk generation...${NC}"
    cleanup_all
    exit "$EXIT_STATUS"
}

handle_exit() {
    local status=$?
    trap - EXIT
    EXIT_STATUS="$status"
    cleanup_all
    exit "$EXIT_STATUS"
}

# ── Queue logic ──────────────────────────────────────────────────────
# If a bulk generation is already running, join the FIFO queue and wait.
# Queue tickets are files named by <epoch_nanoseconds>_<pid> so that
# lexicographic sort gives FIFO order.

queue_ticket_name() {
    # nanosecond timestamp + PID guarantees uniqueness and ordering
    echo "${QUEUE_DIR}/$(date +%s%N)_$$"
}

sorted_queue_basenames() {
    local tickets=("$QUEUE_DIR"/*)
    if [ ${#tickets[@]} -eq 1 ] && [ "${tickets[0]}" = "$QUEUE_DIR/*" ]; then
        return
    fi
    for ticket in "${tickets[@]}"; do
        [ -f "$ticket" ] || continue
        basename "$ticket"
    done | LC_ALL=C sort
}

is_lock_held() {
    if [ -f "$LOCK_FILE" ]; then
        local holder_pid
        holder_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
        if [ -n "$holder_pid" ] && kill -0 "$holder_pid" 2>/dev/null; then
            return 0  # held
        fi
        # Stale lock — clean it up
        rm -f "$LOCK_FILE"
    fi
    return 1  # not held
}

try_acquire_lock() {
    ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null
}

am_i_next_in_queue() {
    # Return 0 if our ticket is the earliest (first in sorted order)
    local first
    first="$(sorted_queue_basenames | head -1)"
    [ -n "$first" ] && [ "$QUEUE_DIR/$first" = "$QUEUE_TICKET" ]
}

queue_is_empty() {
    local first
    first="$(sorted_queue_basenames | head -1)"
    [ -z "$first" ]
}

purge_stale_tickets() {
    # Remove tickets whose owning PID is no longer alive
    for ticket in "$QUEUE_DIR"/*; do
        [ -f "$ticket" ] || continue
        local ticket_pid
        ticket_pid="$(cat "$ticket" 2>/dev/null || true)"
        if [ -n "$ticket_pid" ] && ! kill -0 "$ticket_pid" 2>/dev/null; then
            rm -f "$ticket"
        fi
    done
}

acquire_lock_or_queue() {
    purge_stale_tickets
    if ! is_lock_held && queue_is_empty && try_acquire_lock; then
        return
    fi

    if [ -z "$QUEUE_TICKET" ]; then
        QUEUE_TICKET="$(queue_ticket_name)"
        echo "$$" > "$QUEUE_TICKET"
        local holder_pid
        local position
        holder_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
        position="$(sorted_queue_basenames | grep -n "^$(basename "$QUEUE_TICKET")\$" | cut -d: -f1)"
        echo -e "${YELLOW}Bulk generation busy${holder_pid:+ (pid=$holder_pid)}. Queued at position ${position:-1}. Waiting...${NC}"
    fi

    # Poll until the lock is free AND we are first in the queue
    while true; do
        sleep 5
        purge_stale_tickets
        if ! is_lock_held && am_i_next_in_queue && try_acquire_lock; then
            echo -e "${GREEN}Lock released — starting queued bulk generation.${NC}"
            rm -f "$QUEUE_TICKET"
            QUEUE_TICKET=""
            return
        fi
        local holder_pid=""
        local position=""
        if is_lock_held; then
            holder_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
        fi
        position="$(sorted_queue_basenames | grep -n "^$(basename "$QUEUE_TICKET")\$" | cut -d: -f1)"
        if [ -n "$holder_pid" ]; then
            echo -e "${YELLOW}[queue] Waiting... (running pid=$holder_pid, queue position=${position:-?})${NC}"
        else
            echo -e "${YELLOW}[queue] Waiting... (queue position=${position:-?})${NC}"
        fi
    done
}

# Defaults
PER_GPU=3
MODEL="gpt-5.2"
NUM_TASKS=""
DRY_RUN=false
CATEGORY_FILTER=""  # Empty = all 3 categories (round-robin)
OUTPUT_DIR="data/emtom/tasks"
TASK_GEN_AGENT="mini"
EXTRA_ARGS=()  # Extra args forwarded verbatim to run_emtom.sh generate
DIFFICULTY=""
K_LEVEL=""  # Allowed k-levels (e.g. "2 3"). Empty = random per task.
K_DISTRIBUTION=""  # Slot distribution (e.g. "1:2,2:3,3:3"). Overrides --k-level.
REMOVE_STEPS=""  # Skip pipeline components (e.g. "pddl llm-council simulation task-evolution")
SHOW_QUEUE_STATUS=false

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
    echo "  --num-tasks N       Total tasks for the whole bulk run (default: one full saturated run)"
    echo "                      The launcher divides N across fixed workers and starts them once"
    echo "  --task-gen-agent A  External generator agent: mini|claude|codex (default: mini)"
    echo "  --category CAT      Only generate this category (cooperative, competitive, mixed)"
    echo "  --difficulty LEVEL  Difficulty for generation (easy, medium, hard)"
    echo "  --k-level L [L ...] Allowed k-levels, e.g. --k-level 2 3 (default: random per task)"
    echo "  --k-distribution D  Slots per k-level, e.g. 1:2,2:3,3:3 = 2 slots K=1, 3 K=2, 3 K=3"
    echo "                      Slot counts must sum to --per-gpu. Overrides --k-level."
    echo "  --remove STEP [...] Skip pipeline components: pddl, llm-council, simulation, task-evolution, tom, structure, test"
    echo "  --output-dir DIR    Output directory for submitted tasks (default: data/emtom/tasks)"
    echo "  --dry-run           Show commands without executing"
    echo "  --queue-status      Show current queue status and exit"
    echo ""
    echo "Everything after -- is forwarded to run_emtom.sh generate:"
    echo "  --difficulty, --subtasks-*, --tom-target-*, --iterations-per-task, etc."
    echo "  See: ./emtom/run_emtom.sh generate --help"
    echo ""
    echo "Examples:"
    echo "  ./emtom/bulk_generate.sh                                  # one full saturated run"
    echo "  ./emtom/bulk_generate.sh --per-gpu 6                      # 48 concurrent slots"
    echo "  ./emtom/bulk_generate.sh --num-tasks 4                    # assign 4 total tasks across workers"
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
        --remove)
            shift
            REMOVE_STEPS=""
            while [[ $# -gt 0 && "$1" != -* ]]; do
                REMOVE_STEPS="$REMOVE_STEPS $1"
                shift
            done
            REMOVE_STEPS="${REMOVE_STEPS# }"
            if [ -z "$REMOVE_STEPS" ]; then
                echo "Error: --remove requires at least one step (pddl, llm-council, simulation, task-evolution, tom, structure, test)"
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
        --queue-status)
            SHOW_QUEUE_STATUS=true
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

if [ "$SHOW_QUEUE_STATUS" = true ]; then
    purge_stale_tickets
    echo -e "${BOLD}Bulk Generation Queue Status${NC}"
    echo "=============================="
    if is_lock_held; then
        holder_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
        echo -e "Running: ${GREEN}pid=$holder_pid${NC}"
    else
        echo -e "Running: ${YELLOW}none${NC}"
    fi

    queue_count=0
    while IFS= read -r ticket_name; do
        [ -n "$ticket_name" ] || continue
        ticket_path="$QUEUE_DIR/$ticket_name"
        t_pid="$(cat "$ticket_path" 2>/dev/null || true)"
        if [ -n "$t_pid" ] && kill -0 "$t_pid" 2>/dev/null; then
            queue_count=$((queue_count + 1))
            echo -e "Queued #${queue_count}: ${CYAN}pid=$t_pid${NC} (${ticket_name})"
        fi
    done < <(sorted_queue_basenames)
    if [ "$queue_count" -eq 0 ]; then
        echo -e "Queue: ${YELLOW}empty${NC}"
    fi
    exit 0
fi

for extra_arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$extra_arg" == "--num-tasks" || "$extra_arg" == "--tasks" ]]; then
        echo -e "${RED}Error: bulk_generate.sh owns task count. Use top-level --num-tasks instead of forwarding ${extra_arg}.${NC}"
        exit 1
    fi
done

if [ "$DRY_RUN" != true ]; then
    acquire_lock_or_queue
    trap handle_exit EXIT
    trap 'handle_signal INT' INT
    trap 'handle_signal TERM' TERM
fi

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

if [ -z "$NUM_TASKS" ]; then
    NUM_TASKS="$TOTAL_PROCESSES"
fi

if ! [[ "$NUM_TASKS" =~ ^[0-9]+$ ]] || [ "$NUM_TASKS" -le 0 ]; then
    echo -e "${RED}Error: --num-tasks must be a positive integer${NC}"
    exit 1
fi

if [ "$NUM_TASKS" -lt "$TOTAL_PROCESSES" ]; then
    ACTIVE_WORKERS="$NUM_TASKS"
else
    ACTIVE_WORKERS="$TOTAL_PROCESSES"
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
echo -e "Active workers:     ${GREEN}$ACTIVE_WORKERS${NC}"
echo -e "Categories:         ${GREEN}${CATEGORIES[*]}${NC}"
echo -e "Model:              ${GREEN}$MODEL${NC}"
echo -e "Task-gen agent:     ${GREEN}$TASK_GEN_AGENT${NC}"
[ -n "$DIFFICULTY" ] && echo -e "Difficulty:         ${GREEN}$DIFFICULTY${NC}"
[ -n "$REMOVE_STEPS" ] && echo -e "Skipping steps:     ${GREEN}$REMOVE_STEPS${NC}"
if [ -n "$K_DISTRIBUTION" ]; then
    echo -e "K-distribution:     ${GREEN}${K_DISTRIBUTION}${NC} (per GPU: ${SLOT_K_LEVELS[*]})"
elif [ -n "$K_LEVEL" ]; then
    echo -e "K-level:            ${GREEN}${K_LEVEL}${NC}"
else
    echo -e "K-level:            ${GREEN}random per task${NC}"
fi
echo -e "Total tasks:        ${GREEN}$NUM_TASKS${NC}"
echo -e "Worker launch mode: ${GREEN}1-task-per-worker (respawn on success, stop on fail)${NC}"
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
declare -a PROCESS_TASK_COUNTS
declare -a PROCESS_GROUPS
REQUESTED_TASKS_TOTAL="$NUM_TASKS"

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
        --num-tasks 1
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
    if [ -n "$REMOVE_STEPS" ]; then
        # shellcheck disable=SC2206
        cmd+=(--remove $REMOVE_STEPS)
    fi
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    if [ "$DRY_RUN" = true ]; then
        printf '  EMTOM_GENERATION_RUN_ID=%q EMTOM_GENERATION_RUN_DIR=%q EMTOM_GENERATION_WORKER_ID=%q EMTOM_GENERATION_WORKER_DIR=%q EMTOM_GENERATION_TOTAL_WORKERS=%q EMTOM_GENERATION_REQUESTED_TASKS=%q EMTOM_GENERATION_MODE=%q EMTOM_GENERATION_GPU=%q EMTOM_GENERATION_SLOT=%q EMTOM_GENERATION_STDOUT_LOG=%q CUDA_VISIBLE_DEVICES=%q' \
            "$GENERATION_RUN_ID" "$GENERATION_DIR" "$worker_id" "$worker_dir" "$ACTIVE_WORKERS" "1" "bulk" "$gpu" "$slot" "$log_file" "$gpu"
        printf ' %q' "${cmd[@]}"
        printf '\n'
        return
    fi

    setsid env \
        EMTOM_GENERATION_RUN_ID="$GENERATION_RUN_ID" \
        EMTOM_GENERATION_RUN_DIR="$GENERATION_DIR" \
        EMTOM_GENERATION_WORKER_ID="$worker_id" \
        EMTOM_GENERATION_WORKER_DIR="$worker_dir" \
        EMTOM_GENERATION_TOTAL_WORKERS="$ACTIVE_WORKERS" \
        EMTOM_GENERATION_REQUESTED_TASKS="1" \
        EMTOM_GENERATION_MODE="bulk" \
        EMTOM_GENERATION_GPU="$gpu" \
        EMTOM_GENERATION_SLOT="$slot" \
        EMTOM_GENERATION_STDOUT_LOG="$log_file" \
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    PROCESS_GROUPS+=("$pid")
    if [ "${#SLOT_K_LEVELS[@]}" -gt 0 ]; then
        PROCESS_INFO+=("GPU$gpu:slot$slot:$category:K${SLOT_K_LEVELS[$slot]}:tasks=${process_num_tasks}:$pid")
    else
        PROCESS_INFO+=("GPU$gpu:slot$slot:$category:tasks=${process_num_tasks}:$pid")
    fi
}

launch_single_worker() {
    # Launch one worker that generates exactly 1 task, then exits.
    local slot_idx="$1"
    local attempt="$2"

    local gpu=$((slot_idx % NUM_GPUS))
    local slot=$((slot_idx / NUM_GPUS))
    local category_idx=$((slot_idx % NUM_CATEGORIES))
    local category=${CATEGORIES[$category_idx]}
    local worker_tag
    worker_tag=$(printf '%04d_%03d' "$slot_idx" "$attempt")
    local worker_id="gpu${gpu}-slot${slot}-${category}-worker${worker_tag}"
    local worker_dir="${WORKERS_DIR}/${worker_id}"
    mkdir -p "$worker_dir"

    local log_file
    if [ "${#SLOT_K_LEVELS[@]}" -gt 0 ]; then
        local slot_k="${SLOT_K_LEVELS[$slot]}"
        log_file="$LOG_DIR/gpu${gpu}_slot${slot}_${category}_k${slot_k}_worker${worker_tag}.log"
    else
        log_file="$LOG_DIR/gpu${gpu}_slot${slot}_${category}_worker${worker_tag}.log"
    fi

    local slot_label="GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC}, Tasks=1, Worker=${worker_tag}"
    if [ "${#SLOT_K_LEVELS[@]}" -gt 0 ]; then
        slot_label="GPU $gpu, Slot $slot, Category: ${CYAN}$category${NC}, K=${SLOT_K_LEVELS[$slot]}, Tasks=1, Worker=${worker_tag}"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $slot_label"
        return
    fi
    echo -e "${GREEN}Starting${NC} $slot_label -> $log_file"
    build_worker_command "$gpu" "$slot" "$category" "1" "$worker_id" "$worker_dir" "$log_file"
    # PIDS[-1] now holds the new PID (appended by build_worker_command)
}

count_output_tasks() {
    local count=0
    for jf in "$TASK_DIR"/*.json; do
        [ -f "$jf" ] || continue
        python3 -c "import json; json.load(open('$jf'))" 2>/dev/null && count=$((count + 1))
    done
    echo "$count"
}

launch_workers() {
    PIDS=()
    PROCESS_INFO=()
    PROCESS_TASK_COUNTS=()
    PROCESS_GROUPS=()

    for ((slot_idx=0; slot_idx<ACTIVE_WORKERS; slot_idx++)); do
        launch_single_worker "$slot_idx" 1
    done
}

BULK_START_EPOCH=$(date +%s)
BASELINE_TASK_COUNT=$(count_output_tasks)

launch_workers
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}Dry run complete. In live mode the launcher keeps ${ACTIVE_WORKERS} slots busy, 1 task per worker, respawning on success until ${NUM_TASKS} tasks are produced.${NC}"
    exit 0
fi

echo ""
echo "=============================================="
echo -e "${BOLD}Workers started${NC} (${ACTIVE_WORKERS} slots, 1 task each, target ${NUM_TASKS} new tasks)"
echo "=============================================="
echo ""
echo -e "${CYAN}Monitoring:${NC}"
echo "  watch -n 1 nvidia-smi          # GPU usage"
echo "  tail -f $LOG_DIR/*.log         # Worker stdout logs"
echo "  pkill -9 -u \$(whoami) -f 'mini.*task_gen|emtom/task_gen/runner.py'  # Kill all your workers"
echo ""

echo -e "${BOLD}Waiting for completion...${NC}"
echo ""

failed=0
succeeded=0
submitted_total=0
stop_reason=""

# Track per-slot attempt counters and active PIDs
declare -a SLOT_PIDS       # Current PID for each slot (-1 = stopped)
declare -a SLOT_ATTEMPTS   # Number of attempts for each slot
declare -a SLOT_INFO       # Last info string for each slot

for ((i=0; i<ACTIVE_WORKERS; i++)); do
    SLOT_PIDS[$i]=${PIDS[$i]}
    SLOT_ATTEMPTS[$i]=1
    SLOT_INFO[$i]=${PROCESS_INFO[$i]}
done

MAX_ATTEMPTS_PER_SLOT=5  # Stop a slot after this many consecutive failures

# ── Main respawn loop: poll workers, respawn on success, stop on fail ──
while true; do
    current_task_count=$(count_output_tasks)
    new_tasks=$((current_task_count - BASELINE_TASK_COUNT))

    # Check if we've reached the target
    if [ "$new_tasks" -ge "$NUM_TASKS" ]; then
        stop_reason="target reached ($new_tasks/$NUM_TASKS new tasks)"
        break
    fi

    # Check if all slots are stopped
    all_stopped=true
    for ((i=0; i<ACTIVE_WORKERS; i++)); do
        if [ "${SLOT_PIDS[$i]}" != "-1" ]; then
            all_stopped=false
            break
        fi
    done
    if [ "$all_stopped" = true ]; then
        stop_reason="all slots stopped (failures or max attempts)"
        break
    fi

    # Poll each slot
    for ((i=0; i<ACTIVE_WORKERS; i++)); do
        local_pid=${SLOT_PIDS[$i]}
        [ "$local_pid" = "-1" ] && continue  # Slot already stopped

        # Check if process is still running
        if kill -0 "$local_pid" 2>/dev/null; then
            continue  # Still running
        fi

        # Process finished — check exit status
        if wait "$local_pid" 2>/dev/null; then
            # SUCCESS: worker submitted 1 task
            succeeded=$((succeeded + 1))
            submitted_total=$((submitted_total + 1))
            SLOT_ATTEMPTS[$i]=1  # Reset attempt counter on success

            new_tasks=$(($(count_output_tasks) - BASELINE_TASK_COUNT))
            echo -e "${GREEN}[DONE]${NC} ${SLOT_INFO[$i]}  (${new_tasks}/${NUM_TASKS} new tasks so far)"

            if [ "$new_tasks" -ge "$NUM_TASKS" ]; then
                SLOT_PIDS[$i]=-1
                continue
            fi

            # Respawn this slot with a fresh worker
            next_attempt=$((SLOT_ATTEMPTS[$i] + 1))
            SLOT_ATTEMPTS[$i]=$next_attempt
            launch_single_worker "$i" "$next_attempt"
            SLOT_PIDS[$i]=${PIDS[-1]}
            SLOT_INFO[$i]=${PROCESS_INFO[-1]}
        else
            # FAIL: worker exited non-zero — stop this slot
            failed=$((failed + 1))
            attempt_num=${SLOT_ATTEMPTS[$i]}

            if [ "$attempt_num" -ge "$MAX_ATTEMPTS_PER_SLOT" ]; then
                echo -e "${RED}[FAIL]${NC} ${SLOT_INFO[$i]}  (${attempt_num}/${MAX_ATTEMPTS_PER_SLOT} attempts, slot stopped)"
                SLOT_PIDS[$i]=-1
            else
                echo -e "${RED}[FAIL]${NC} ${SLOT_INFO[$i]}  (attempt ${attempt_num}, stopping slot)"
                SLOT_PIDS[$i]=-1
            fi
        fi
    done

    sleep 5
done

# Kill any still-running workers (target reached)
for ((i=0; i<ACTIVE_WORKERS; i++)); do
    local_pid=${SLOT_PIDS[$i]}
    if [ "$local_pid" != "-1" ] && kill -0 "$local_pid" 2>/dev/null; then
        kill "$local_pid" 2>/dev/null || true
    fi
done

BULK_END_EPOCH=$(date +%s)
WALL_CLOCK=$((BULK_END_EPOCH - BULK_START_EPOCH))

echo ""
echo "=============================================="
echo -e "${BOLD}Bulk generation complete${NC} (${submitted_total}/${NUM_TASKS} new tasks, ${succeeded} worker successes, ${failed} worker failures)"
echo -e "Stop reason: ${CYAN}${stop_reason}${NC}"
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
