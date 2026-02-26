#!/bin/bash
# EMTOM Utilities
# Usage: ./emtom/run_util.sh <command> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default values
OUTPUT_FILE=""
OUTPUT_DIR=""
TASK_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${BLUE}EMTOM Utilities${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC} ./emtom/run_util.sh <command> [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  prompt <task.json>     Show the built prompt for each agent"
    echo "  dag <task.json>        Visualize the PDDL goal DAG as a PNG graph"
    echo "  summarize <output_dir> Summarize benchmark results from a run directory"
    echo "  task-summary [dir]     Summary table of tasks with calibration status"
    echo "  task-stats [dir]       Aggregate stats by category, model, agent count"
    echo "  list-tasks             List all curated task files"
    echo ""
    echo -e "${YELLOW}DAG Options:${NC}"
    echo "  -o, --output FILE   Output file path (default: /tmp/task_dag.png)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./emtom/run_util.sh prompt data/emtom/tasks/my_task.json"
    echo "  ./emtom/run_util.sh dag data/emtom/tasks/my_task.json"
    echo "  ./emtom/run_util.sh dag my_task.json -o dag.png"
    echo "  ./emtom/run_util.sh summarize outputs/emtom/2025-01-05_12-00-00-benchmark"
    echo "  ./emtom/run_util.sh task-summary"
    echo "  ./emtom/run_util.sh task-stats"
    echo "  ./emtom/run_util.sh task-stats data/emtom/tasks"
    echo "  ./emtom/run_util.sh list-tasks"
}

run_prompt() {
    if [ -z "$TASK_FILE" ]; then
        echo -e "${RED}Error: Please specify a task file${NC}"
        echo "Usage: ./emtom/run_util.sh prompt <task.json>"
        exit 1
    fi

    if [ ! -f "$TASK_FILE" ]; then
        echo -e "${RED}Error: Task file not found: $TASK_FILE${NC}"
        exit 1
    fi

    python -m emtom.utils.show_prompt "$TASK_FILE"
}

run_summarize() {
    if [ -z "$OUTPUT_DIR" ]; then
        echo -e "${RED}Error: Please specify an output directory${NC}"
        echo "Usage: ./emtom/run_util.sh summarize <output_dir>"
        exit 1
    fi

    if [ ! -d "$OUTPUT_DIR" ]; then
        echo -e "${RED}Error: Directory not found: $OUTPUT_DIR${NC}"
        exit 1
    fi

    python -m emtom.utils.summarize "$OUTPUT_DIR"
}

run_dag() {
    if [ -z "$TASK_FILE" ]; then
        echo -e "${RED}Error: Please specify a task file${NC}"
        echo "Usage: ./emtom/run_util.sh dag <task.json> [-o output.png]"
        exit 1
    fi

    if [ ! -f "$TASK_FILE" ]; then
        echo -e "${RED}Error: Task file not found: $TASK_FILE${NC}"
        exit 1
    fi

    # Build arguments for the visualizer
    ARGS="$TASK_FILE"
    if [ -n "$OUTPUT_FILE" ]; then
        ARGS="$ARGS -o $OUTPUT_FILE"
    fi

    python -m emtom.cli.visualize_task $ARGS
}

run_task_summary() {
    TASKS_DIR="${TASK_DIR:-data/emtom/tasks}"
    python -m emtom.utils.task_summary "$TASKS_DIR"
}

run_task_stats() {
    TASKS_DIR="${TASK_DIR:-data/emtom/tasks}"
    python -m emtom.utils.task_summary --stats "$TASKS_DIR"
}

run_list_tasks() {
    echo -e "${BLUE}=============================================="
    echo "EMTOM Curated Tasks"
    echo -e "==============================================${NC}"
    echo ""

    TASK_DIR="data/emtom/tasks"

    if [ ! -d "$TASK_DIR" ]; then
        echo "No curated tasks directory found."
        exit 0
    fi

    count=$(ls -1 "$TASK_DIR"/*.json 2>/dev/null | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "No task files found in $TASK_DIR"
        exit 0
    fi

    echo "Found $count task file(s) in $TASK_DIR:"
    echo ""

    for f in "$TASK_DIR"/*.json; do
        # Extract title from JSON
        title=$(python3 -c "import json; print(json.load(open('$f')).get('title', 'Untitled'))" 2>/dev/null || echo "Untitled")
        basename "$f" | sed 's/\.json$//'
        echo "  -> $title"
        echo ""
    done
}

# Parse command line arguments
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        prompt)
            COMMAND="prompt"
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                TASK_FILE=$1
                shift
            fi
            ;;
        summarize)
            COMMAND="summarize"
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                OUTPUT_DIR=$1
                shift
            fi
            ;;
        dag)
            COMMAND="dag"
            shift
            # Next argument should be the task file (if not a flag)
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                TASK_FILE=$1
                shift
            fi
            ;;
        task-summary)
            COMMAND="task-summary"
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                TASK_DIR=$1
                shift
            fi
            ;;
        task-stats)
            COMMAND="task-stats"
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                TASK_DIR=$1
                shift
            fi
            ;;
        list-tasks)
            COMMAND="list-tasks"
            shift
            ;;
        --output|-o)
            OUTPUT_FILE=$2
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            # If we're in dag mode and this looks like a file path, treat it as task file
            if [[ "$COMMAND" == "dag" && -z "$TASK_FILE" && ! "$1" =~ ^-- ]]; then
                TASK_FILE=$1
                shift
            else
                echo -e "${RED}Unknown option: $1${NC}"
                print_usage
                exit 1
            fi
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    print_usage
    exit 1
fi

case $COMMAND in
    prompt)
        run_prompt
        ;;
    summarize)
        run_summarize
        ;;
    dag)
        run_dag
        ;;
    task-summary)
        run_task_summary
        ;;
    task-stats)
        run_task_stats
        ;;
    list-tasks)
        run_list_tasks
        ;;
esac
