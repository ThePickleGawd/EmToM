#!/bin/bash
# GPT-Guided Room Exploration Script
#
# Usage:
#   ./main/exploration/llm_exploration.sh                    # Default
#   ./main/exploration/llm_exploration.sh --episode 334      # Specific episode
#   ./main/exploration/llm_exploration.sh --model gpt-4o     # Use GPT-4o

# Navigate to project root
cd /data4/parth/Partnr-EmToM

# Activate conda environment
source /data4/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Set up X11 forwarding for video display
export DISPLAY=${DISPLAY:-localhost:10.0}
xhost +local: 2>/dev/null || true

echo "X11 Display: $DISPLAY"

# Default settings
EPISODE_ID="334"
MAX_ROOMS="3"
MAX_INTERACTIONS="2"
GPT_MODEL="gpt-4o-mini"
LIVE_DISPLAY="True"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --episode)
            EPISODE_ID="$2"
            shift 2
            ;;
        --max-rooms)
            MAX_ROOMS="$2"
            shift 2
            ;;
        --max-interactions)
            MAX_INTERACTIONS="$2"
            shift 2
            ;;
        --model)
            GPT_MODEL="$2"
            shift 2
            ;;
        --no-display)
            LIVE_DISPLAY="False"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --episode ID           Episode ID to load (default: 334)"
            echo "  --max-rooms N          Max rooms to visit (default: 3)"
            echo "  --max-interactions N   Max interactions per room (default: 2)"
            echo "  --model MODEL          GPT model to use (default: gpt-4o-mini)"
            echo "  --no-display           Disable live X11 display"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "=== GPT-Guided Exploration Configuration ==="
echo "Episode ID: $EPISODE_ID"
echo "GPT Model: $GPT_MODEL"
echo "Max rooms: $MAX_ROOMS"
echo "Max interactions per room: $MAX_INTERACTIONS"
echo "Live display: $LIVE_DISPLAY"
echo "============================================="
echo ""

HYDRA_FULL_ERROR=1 python -m main.exploration.explore_with_gpt \
    hydra.run.dir="." \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    +skill_runner_episode_id="$EPISODE_ID" \
    +live_display=$LIVE_DISPLAY \
    +max_rooms=$MAX_ROOMS \
    +max_interactions_per_room=$MAX_INTERACTIONS \
    +gpt_model="$GPT_MODEL"