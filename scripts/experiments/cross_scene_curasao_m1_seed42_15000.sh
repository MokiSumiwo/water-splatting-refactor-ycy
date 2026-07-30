#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
exec env \
  GPU="${GPU:-6}" \
  SCENE_SLUG="curasao" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Curasao}" \
  "${REPO_DIR}/scripts/experiments/cross_scene_m1_common.sh"

