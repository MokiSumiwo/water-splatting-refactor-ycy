#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-7}"
export SCENE_SLUG="iui3"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
export EXPERIMENT_TAG="a1_diag"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export GIVAR_ENABLED="True"
export GIVAR_DIAGNOSTIC_ONLY="True"
export LAMBDA_GIVAR="0.0"
exec "${REPO_DIR}/scripts/experiments/givar_5k_common.sh"
