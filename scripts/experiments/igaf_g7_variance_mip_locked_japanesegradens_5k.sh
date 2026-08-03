#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-8}"
export SCENE_SLUG="japanesegradens"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}"
export EXPERIMENT_TAG="g7_variance_mip_locked"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export IGAF_ENABLED="True"
export IGAF_START_STEP="${IGAF_START_STEP:-2500}"
export IGAF_RAMP_STEPS="${IGAF_RAMP_STEPS:-500}"
export IGAF_MIP_ENABLED="True"
export IGAF_MIP_MODE="variance"
export IGAF_AXIS_MODE="locked"
export IGAF_RESET_SPLIT_COEFFS="True"
exec "${REPO_DIR}/scripts/experiments/igaf_5k_common.sh"
