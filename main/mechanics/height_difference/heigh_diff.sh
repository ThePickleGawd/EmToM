#!/bin/bash
# Height Difference Exploration Script
# Runs GPT-guided exploration and generates tasks when surprising findings are detected

# Navigate to project root first
cd /data4/parth/Partnr-EmToM

# Activate conda environment
source /data4/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Set up X11 forwarding for video display
export DISPLAY=${DISPLAY:-localhost:10.0}
# Ensure trusted X11 forwarding
xhost +local: 2>/dev/null || true

echo "X11 Display: $DISPLAY"
echo "Testing X11 connection..."
xdpyinfo | head -3 || echo "Warning: X11 may not be working"

echo ""
echo "=== Running GPT-GUIDED EXPLORATION mode ==="
echo "    (Will stop and generate tasks on first surprising finding)"
echo ""

HYDRA_FULL_ERROR=1 python -m main.mechanics.height_difference.height_difference \
    hydra.run.dir="." \
    +skill_runner_show_topdown=False \
    +skill_runner_show_videos=True \
    evaluation.save_video=True \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    +skill_runner_episode_id="334" \
    +exploration_mode=gpt \
    +live_display=True \
    +gpt_model=gpt-4o-mini \
    +max_rooms=5 \
    +max_interactions_per_room=3 \
    +num_tasks=5