#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-6}"
export SCENE_SLUG="japanesegradens"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}"
export EXPERIMENT_TAG="c0_m1"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export MCGR_ENABLED="False"
export MCGR_DIAGNOSTIC_ONLY="False"
export BUILD_MCGR_CORRESPONDENCE="${BUILD_MCGR_CORRESPONDENCE:-0}"
exec "${REPO_DIR}/scripts/experiments/mcgr_5k_common.sh"
