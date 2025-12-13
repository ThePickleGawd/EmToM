#!/bin/bash
# Script to run room exploration in Habitat environment
#
# Usage:
#   ./main/exploration/explore_rooms.sh                    # Default (live X11 display)
#   ./main/exploration/explore_rooms.sh --save-video       # Save video instead of live display
#   ./main/exploration/explore_rooms.sh --episode 123      # Specific episode
#   ./main/exploration/explore_rooms.sh --furniture 3      # Max 3 furniture per room

# Navigate to project root
cd /data4/parth/Partnr-EmToM

# Activate conda environment
source /data4/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Set up X11 forwarding for video display
export DISPLAY=${DISPLAY:-localhost:10.0}
xhost +local: 2>/dev/null || true

echo "X11 Display: $DISPLAY"
echo "Testing X11 connection..."
xdpyinfo | head -3 || echo "Warning: X11 may not be working"

# Default settings - live display mode by default
EPISODE_ID="334"
FURNITURE_PER_ROOM=""
LIVE_DISPLAY="True"
SAVE_VIDEO="False"
RANDOMIZE_ROOMS="False"
DISPLAY_SCALE="1.0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --episode)
            EPISODE_ID="$2"
            shift 2
            ;;
        --furniture)
            FURNITURE_PER_ROOM="+furniture_per_room=$2"
            shift 2
            ;;
        --save-video)
            # Switch to video saving mode (disable live display)
            LIVE_DISPLAY="False"
            SAVE_VIDEO="True"
            shift
            ;;
        --scale)
            DISPLAY_SCALE="$2"
            shift 2
            ;;
        --randomize-rooms)
            RANDOMIZE_ROOMS="True"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --episode ID        Episode ID to load (default: 334)"
            echo "  --furniture N       Max furniture to visit per room (default: all)"
            echo "  --save-video        Save video files instead of live X11 display"
            echo "  --scale FACTOR      Scale factor for live display (default: 1.0)"
            echo "  --randomize-rooms   Randomize order of room visits"
            echo "  --help              Show this help message"
            echo ""
            echo "By default, the script shows a LIVE X11 display window that stays open"
            echo "throughout the exploration. Press 'q' in the window to quit early."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Room Exploration Configuration ==="
echo "Episode ID: $EPISODE_ID"
if [ "$LIVE_DISPLAY" = "True" ]; then
    echo "Mode: LIVE X11 DISPLAY (window stays open during exploration)"
    echo "Display scale: $DISPLAY_SCALE"
    echo "Press 'q' in the display window to quit early"
else
    echo "Mode: Save video files"
fi
echo "Randomize rooms: $RANDOMIZE_ROOMS"
if [ -n "$FURNITURE_PER_ROOM" ]; then
    echo "Furniture per room: ${FURNITURE_PER_ROOM#*=}"
else
    echo "Furniture per room: all"
fi
echo "======================================="
echo ""

# Run the exploration script
HYDRA_FULL_ERROR=1 python -m main.exploration.explore_all_rooms \
    hydra.run.dir="." \
    +skill_runner_show_topdown=False \
    +live_display=$LIVE_DISPLAY \
    +display_scale=$DISPLAY_SCALE \
    evaluation.save_video=$SAVE_VIDEO \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    +skill_runner_episode_id="$EPISODE_ID" \
    +randomize_rooms=$RANDOMIZE_ROOMS \
    $FURNITURE_PER_ROOM
