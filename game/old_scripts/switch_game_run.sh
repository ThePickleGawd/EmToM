#!/usr/bin/env bash
set -euo pipefail

# Run the switch game using the emtom house dataset (data/emtom/test_house.json.gz).
# Override SCENE_ID to target a different emtom scene if available.

SCENE_ID="${SCENE_ID:-102817140}"

set -a
if [ -f .env ]; then
  source .env
fi
set +a

python -m habitat_llm.examples.emtom --config-name game/switch_game \
    mode="cli" \
    evaluation.save_video=True \
    +game.manual_agents=[] \
    +evaluation.save_fpv_stills=False \
    habitat.dataset.data_path="data/emtom/test_house.json.gz" \
    habitat.dataset.content_scenes="[${SCENE_ID}]" \
    "$@"
