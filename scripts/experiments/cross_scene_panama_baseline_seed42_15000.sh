#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
exec env \
  GPU="${GPU:-8}" \
  SCENE_SLUG="panama" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Panama}" \
  "${REPO_DIR}/scripts/experiments/cross_scene_baseline_common.sh"

