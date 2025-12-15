#!/usr/bin/env bash
set -euo pipefail

set -a
if [ -f .env ]; then
  source .env
fi
set +a

python -m habitat_llm.examples.emtom --config-name game/bomb_game \
    mode="cli" \
    evaluation.save_video=True \
    +game.manual_agents=[0,1] \
    +evaluation.save_fpv_stills=False \
    # +game.manual_obs_popup=true \ # I think this is for manual popup
    # habitat.environment.max_episode_steps=10000 \
