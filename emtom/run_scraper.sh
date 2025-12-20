#!/bin/bash
# EMTOM Scenario Scraper Pipeline
# Usage: ./emtom/run_scraper.sh <command> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default values
COUNT=100
MODEL="gpt-5.2"
TEMPERATURE=0.8
DELAY=1.0
OUTPUT_DIR="data/emtom/scenarios/scraped"
PREMISES_DIR="data/emtom/scenarios/raw"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${BLUE}EMTOM Scenario Scraper Pipeline${NC}"
    echo ""
    echo "Scrapes scenario ideas from games, movies, and interactive fiction,"
    echo "then uses an LLM to generate household puzzle scenarios."
    echo ""
    echo -e "${YELLOW}Usage:${NC} ./emtom/run_scraper.sh <command> [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  scrape      Scrape premises from curated web sources (Wikipedia, etc.)"
    echo "  generate    Generate scenarios from scraped premises using LLM"
    echo "  scratch     Generate scenarios from scratch (no scraping needed)"
    echo "  all         Full pipeline: scrape -> generate"
    echo "  stats       Show statistics about scraped/generated data"
    echo "  list        List generated scenario files"
    echo "  view N      View scenario N (e.g., view 1 shows scenario_001.txt)"
    echo ""
    echo -e "${YELLOW}Generation Options:${NC}"
    echo "  --count N          Number of scenarios to generate (default: $COUNT)"
    echo "  --model MODEL      LLM model to use (default: $MODEL)"
    echo "  --temperature T    Sampling temperature 0-1 (default: $TEMPERATURE)"
    echo ""
    echo -e "${YELLOW}Directory Options:${NC}"
    echo "  --output-dir DIR   Where to save .txt files (default: $OUTPUT_DIR)"
    echo "  --premises-dir DIR Where to save raw premises (default: $PREMISES_DIR)"
    echo ""
    echo -e "${YELLOW}Other Options:${NC}"
    echo "  --delay SECS       Delay between API calls (default: $DELAY)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Scrape premises from curated sources"
    echo "  ./emtom/run_scraper.sh scrape"
    echo ""
    echo "  # Generate 50 scenarios from scraped premises"
    echo "  ./emtom/run_scraper.sh generate --count 50"
    echo ""
    echo "  # Generate 20 scenarios from scratch (no scraping)"
    echo "  ./emtom/run_scraper.sh scratch --count 20"
    echo ""
    echo "  # Full pipeline: scrape then generate 100 scenarios"
    echo "  ./emtom/run_scraper.sh all --count 100"
    echo ""
    echo "  # View a specific scenario"
    echo "  ./emtom/run_scraper.sh view 5"
    echo ""
    echo -e "${YELLOW}Dependencies:${NC}"
    echo "  pip install beautifulsoup4 requests openai"
}

check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    missing=""

    python -c "import bs4" 2>/dev/null || missing="$missing beautifulsoup4"
    python -c "import requests" 2>/dev/null || missing="$missing requests"
    python -c "import openai" 2>/dev/null || missing="$missing openai"

    if [ -n "$missing" ]; then
        echo -e "${RED}Missing dependencies:${NC}$missing"
        echo ""
        echo "Install with: pip install$missing"
        exit 1
    fi

    echo -e "${GREEN}All dependencies installed.${NC}"
}

run_scrape() {
    echo -e "${BLUE}=============================================="
    echo "EMTOM Scenario Scraper - Scraping Premises"
    echo -e "==============================================${NC}"
    echo ""
    echo "Output directory: $PREMISES_DIR"
    echo "Rate limit delay: ${DELAY}s"
    echo ""

    check_dependencies

    python -m emtom.scenario_scraper.run scrape \
        --premises-dir "$PREMISES_DIR" \
        --delay "$DELAY"
}

run_generate() {
    echo -e "${BLUE}=============================================="
    echo "EMTOM Scenario Scraper - Generating Scenarios"
    echo -e "==============================================${NC}"
    echo ""
    echo "Model: $MODEL"
    echo "Count: $COUNT"
    echo "Temperature: $TEMPERATURE"
    echo "Output: $OUTPUT_DIR"
    echo ""

    check_dependencies

    # Check if premises exist
    if [ ! -f "$PREMISES_DIR/premises.json" ]; then
        echo -e "${RED}Error: No premises found at $PREMISES_DIR/premises.json${NC}"
        echo "Run 'scrape' first, or use 'scratch' to generate without premises."
        exit 1
    fi

    python -m emtom.scenario_scraper.run generate \
        --premises-dir "$PREMISES_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --count "$COUNT" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --delay "$DELAY"
}

run_scratch() {
    echo -e "${BLUE}=============================================="
    echo "EMTOM Scenario Scraper - Generating from Scratch"
    echo -e "==============================================${NC}"
    echo ""
    echo "Model: $MODEL"
    echo "Count: $COUNT"
    echo "Temperature: $TEMPERATURE"
    echo "Output: $OUTPUT_DIR"
    echo "(No web scraping - purely LLM-generated scenarios)"
    echo ""

    check_dependencies

    python -m emtom.scenario_scraper.run generate \
        --scratch \
        --output-dir "$OUTPUT_DIR" \
        --count "$COUNT" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --delay "$DELAY"
}

run_all() {
    echo -e "${BLUE}=============================================="
    echo "EMTOM Scenario Scraper - Full Pipeline"
    echo -e "==============================================${NC}"
    echo ""

    run_scrape
    echo ""
    run_generate
}

run_stats() {
    echo -e "${BLUE}=============================================="
    echo "EMTOM Scenario Scraper - Statistics"
    echo -e "==============================================${NC}"
    echo ""

    python -m emtom.scenario_scraper.run stats \
        --premises-dir "$PREMISES_DIR" \
        --output-dir "$OUTPUT_DIR"
}

run_list() {
    echo -e "${BLUE}=============================================="
    echo "Generated Scenario Files"
    echo -e "==============================================${NC}"
    echo ""

    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "No scenarios generated yet."
        echo "Run: ./emtom/run_scraper.sh scratch --count 10"
        exit 0
    fi

    count=$(ls -1 "$OUTPUT_DIR"/*.txt 2>/dev/null | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "No scenarios generated yet."
        echo "Run: ./emtom/run_scraper.sh scratch --count 10"
        exit 0
    fi

    echo "Found $count scenario files in $OUTPUT_DIR:"
    echo ""
    ls -1 "$OUTPUT_DIR"/*.txt | while read f; do
        basename "$f"
    done
}

run_view() {
    local num=$1

    if [ -z "$num" ]; then
        echo -e "${RED}Error: Please specify a scenario number${NC}"
        echo "Usage: ./emtom/run_scraper.sh view N"
        exit 1
    fi

    # Pad with zeros
    local filename=$(printf "scenario_%03d.txt" "$num")
    local filepath="$OUTPUT_DIR/$filename"

    if [ ! -f "$filepath" ]; then
        echo -e "${RED}Error: $filename not found${NC}"
        echo "Available scenarios:"
        ls -1 "$OUTPUT_DIR"/*.txt 2>/dev/null | head -5
        exit 1
    fi

    echo -e "${BLUE}=============================================="
    echo "$filename"
    echo -e "==============================================${NC}"
    echo ""
    cat "$filepath"
    echo ""
}

# Parse command line arguments
COMMAND=""
VIEW_NUM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        scrape|generate|scratch|all|stats|list)
            COMMAND=$1
            shift
            ;;
        view)
            COMMAND="view"
            VIEW_NUM=$2
            shift 2
            ;;
        --count)
            COUNT=$2
            shift 2
            ;;
        --model)
            MODEL=$2
            shift 2
            ;;
        --temperature)
            TEMPERATURE=$2
            shift 2
            ;;
        --delay)
            DELAY=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --premises-dir)
            PREMISES_DIR=$2
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
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
    scrape)
        run_scrape
        ;;
    generate)
        run_generate
        ;;
    scratch)
        run_scratch
        ;;
    all)
        run_all
        ;;
    stats)
        run_stats
        ;;
    list)
        run_list
        ;;
    view)
        run_view "$VIEW_NUM"
        ;;
esac

echo ""
echo -e "${GREEN}=============================================="
echo "Done!"
echo -e "==============================================${NC}"
