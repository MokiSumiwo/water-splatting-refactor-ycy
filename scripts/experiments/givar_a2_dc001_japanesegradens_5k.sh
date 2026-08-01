#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-8}"
export SCENE_SLUG="japanesegradens"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}"
export EXPERIMENT_TAG="a2_dc001"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export GIVAR_ENABLED="True"
export GIVAR_DIAGNOSTIC_ONLY="False"
export GIVAR_DC_ENABLED="True"
export GIVAR_SH_ENABLED="False"
export LAMBDA_GIVAR="${LAMBDA_GIVAR:-0.01}"
exec "${REPO_DIR}/scripts/experiments/givar_5k_common.sh"
