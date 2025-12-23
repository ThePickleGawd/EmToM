#!/bin/bash
# EMTOM Utilities
# Usage: ./emtom/run_util.sh <command> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default values
FORMAT="ascii"
OUTPUT_FILE=""
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
    echo "  dag <task.json>   Visualize the subtask DAG for a task file"
    echo "  list-tasks        List all curated task files"
    echo ""
    echo -e "${YELLOW}DAG Visualization Options:${NC}"
    echo "  --format FORMAT   Output format: ascii (default), dot, mermaid"
    echo "  --output FILE     Write output to file instead of stdout"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # View DAG for a task (ASCII art)"
    echo "  ./emtom/run_util.sh dag data/emtom/tasks/task_xxx.json"
    echo ""
    echo "  # Export DAG to Graphviz DOT format"
    echo "  ./emtom/run_util.sh dag data/emtom/tasks/working_task.json --format dot"
    echo ""
    echo "  # Export DAG to Mermaid (for markdown/GitHub)"
    echo "  ./emtom/run_util.sh dag data/emtom/tasks/working_task.json --format mermaid"
    echo ""
    echo "  # Save DOT output to file for rendering"
    echo "  ./emtom/run_util.sh dag task.json --format dot --output task.dot"
    echo "  dot -Tpng task.dot -o task.png"
    echo ""
    echo "  # List available task files"
    echo "  ./emtom/run_util.sh list-tasks"
}

run_dag() {
    if [ -z "$TASK_FILE" ]; then
        echo -e "${RED}Error: Please specify a task file${NC}"
        echo "Usage: ./emtom/run_util.sh dag <task.json> [--format FORMAT]"
        exit 1
    fi

    if [ ! -f "$TASK_FILE" ]; then
        echo -e "${RED}Error: Task file not found: $TASK_FILE${NC}"
        exit 1
    fi

    # Build Python command based on format
    PYTHON_CMD="
import json
from emtom.task_gen import GeneratedTask, visualize_task_dag, to_dot, to_mermaid

with open('$TASK_FILE') as f:
    task_data = json.load(f)

task = GeneratedTask.from_dict(task_data)

if '$FORMAT' == 'ascii':
    output = visualize_task_dag(task)
elif '$FORMAT' == 'dot':
    output = to_dot(task.subtasks, title=task.title)
elif '$FORMAT' == 'mermaid':
    output = to_mermaid(task.subtasks, title=task.title)
else:
    print('Unknown format: $FORMAT')
    exit(1)

print(output)
"

    if [ -n "$OUTPUT_FILE" ]; then
        python3 -c "$PYTHON_CMD" > "$OUTPUT_FILE"
        echo -e "${GREEN}Output written to: $OUTPUT_FILE${NC}"
    else
        python3 -c "$PYTHON_CMD"
    fi
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
        dag)
            COMMAND="dag"
            shift
            # Next argument should be the task file (if not a flag)
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                TASK_FILE=$1
                shift
            fi
            ;;
        list-tasks)
            COMMAND="list-tasks"
            shift
            ;;
        --format)
            FORMAT=$2
            if [[ "$FORMAT" != "ascii" && "$FORMAT" != "dot" && "$FORMAT" != "mermaid" ]]; then
                echo -e "${RED}Error: --format must be 'ascii', 'dot', or 'mermaid'${NC}"
                exit 1
            fi
            shift 2
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
    dag)
        run_dag
        ;;
    list-tasks)
        run_list_tasks
        ;;
esac
