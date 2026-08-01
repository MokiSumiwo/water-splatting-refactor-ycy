#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-7}"
export SCENE_SLUG="japanesegradens"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}"
export EXPERIMENT_TAG="c1_diag"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export MCGR_ENABLED="True"
export MCGR_DIAGNOSTIC_ONLY="True"
export MCGR_START_STEP="${MCGR_START_STEP:-0}"
export MCGR_STOP_STEP="${MCGR_STOP_STEP:-10000}"
exec "${REPO_DIR}/scripts/experiments/mcgr_5k_common.sh"
