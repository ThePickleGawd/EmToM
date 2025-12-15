#!/usr/bin/env bash
set -euo pipefail

# Simple viewer launcher for a PARTNR/HSSD scene using habitat_llm/examples/viewer.py.
# Customize via env vars:
#   SCENE_PATH     - path to the scene instance/stage file (.scene_instance.json or .glb)
#   SCENE_DATASET  - path to the scene dataset config (.scene_dataset_config.json)
#   WIDTH / HEIGHT - window size (defaults: 800x600)

SCENE_PATH="${SCENE_PATH:-data/hssd-hab/scenes-partnr-filtered/102817140.scene_instance.json}"
SCENE_DATASET="${SCENE_DATASET:-data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json}"
WIDTH="${WIDTH:-800}"
HEIGHT="${HEIGHT:-600}"

# Ensure the viewer font is available where the script expects it.
FONT_TARGET="habitat_llm/data/fonts/ProggyClean.ttf"
FONT_SOURCE="third_party/habitat-lab/habitat-hitl/habitat_hitl/core/fonts/ProggyClean.ttf"
if [ ! -f "${FONT_TARGET}" ]; then
  mkdir -p "$(dirname "${FONT_TARGET}")"
  if [ -f "${FONT_SOURCE}" ]; then
    ln -sf "$(pwd)/${FONT_SOURCE}" "${FONT_TARGET}"
  else
    echo "Font file not found at ${FONT_SOURCE}. Please set FONT_TARGET manually." >&2
    exit 1
  fi
fi

python -m habitat_llm.examples.viewer \
  --scene "${SCENE_PATH}" \
  --dataset "${SCENE_DATASET}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  "$@"
