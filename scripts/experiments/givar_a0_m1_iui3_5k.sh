#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-6}"
export SCENE_SLUG="iui3"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
export EXPERIMENT_TAG="a0_m1"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export GIVAR_ENABLED="False"
export GIVAR_DIAGNOSTIC_ONLY="False"
exec "${REPO_DIR}/scripts/experiments/givar_5k_common.sh"
