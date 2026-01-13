#!/bin/bash
# Bulk EMTOM Exploration
# Runs exploration across all GPUs (1 process per GPU by default)
# No categories - exploration is category-agnostic
#
# Usage: ./emtom/bulk_explore.sh [options]
#
# Examples:
#   ./emtom/bulk_explore.sh                   # 8 processes (1 per GPU)
#   ./emtom/bulk_explore.sh --per-gpu 2       # 16 processes (2 per GPU)
#   ./emtom/bulk_explore.sh --steps 50        # More exploration steps
#   ./emtom/bulk_explore.sh --dry-run         # Preview without running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Defaults
PER_GPU=1
MODEL="gpt-5.2"
STEPS=20
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_usage() {
    echo -e "${BOLD}Bulk EMTOM Exploration${NC}"
    echo ""
    echo "Runs exploration across all GPUs (1 process per GPU by default)"
    echo "Each process explores a random scene independently"
    echo ""
    echo "Usage: ./emtom/bulk_explore.sh [options]"
    echo ""
    echo "Options:"
    echo "  --per-gpu N      Processes per GPU (default: 1)"
    echo "  --model MODEL    LLM model (default: gpt-5.2)"
    echo "  --steps N        Exploration steps per process (default: 20)"
    echo "  --dry-run        Show commands without executing"
    echo ""
    echo "Examples:"
    echo "  ./emtom/bulk_explore.sh                   # 8 processes (1 per GPU)"
    echo "  ./emtom/bulk_explore.sh --per-gpu 2       # 16 processes"
    echo "  ./emtom/bulk_explore.sh --steps 50        # More steps"
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
        --steps)
            STEPS=$2
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

TOTAL_PROCESSES=$((NUM_GPUS * PER_GPU))

# Create log directory
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="outputs/emtom/${TIMESTAMP}-bulk-explore"
mkdir -p "$LOG_DIR"

echo -e "${BOLD}=============================================="
echo -e "Bulk EMTOM Exploration"
echo -e "==============================================${NC}"
echo -e "GPUs:               ${GREEN}$NUM_GPUS${NC}"
echo -e "Processes per GPU:  ${GREEN}$PER_GPU${NC}"
echo -e "Total processes:    ${GREEN}$TOTAL_PROCESSES${NC}"
echo -e "Model:              ${GREEN}$MODEL${NC}"
echo -e "Steps per process:  ${GREEN}$STEPS${NC}"
echo -e "Log directory:      ${CYAN}$LOG_DIR${NC}"
echo "=============================================="
echo ""

# Track PIDs
declare -a PIDS
declare -a PROCESS_INFO

process_idx=0

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    for slot in $(seq 0 $((PER_GPU - 1))); do
        log_file="$LOG_DIR/gpu${gpu}_slot${slot}.log"

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN]${NC} GPU $gpu, Slot $slot"
            echo "  CUDA_VISIBLE_DEVICES=$gpu ./emtom/run_emtom.sh explore --model $MODEL --steps $STEPS"
        else
            echo -e "${GREEN}Starting${NC} GPU $gpu, Slot $slot -> $log_file"

            CUDA_VISIBLE_DEVICES=$gpu ./emtom/run_emtom.sh explore \
                --model "$MODEL" \
                --steps "$STEPS" \
                > "$log_file" 2>&1 &

            pid=$!
            PIDS+=($pid)
            PROCESS_INFO+=("GPU$gpu:slot$slot:$pid")
        fi

        ((process_idx++))
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
echo "  pkill -f 'run_emtom.sh exp'    # Kill all"
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
echo "=============================================="
