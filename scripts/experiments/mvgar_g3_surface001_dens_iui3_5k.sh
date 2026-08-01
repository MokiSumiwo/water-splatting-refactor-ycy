#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-9}"
export SCENE_SLUG="iui3"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
export EXPERIMENT_TAG="g3_surface001_dens"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export MVGAR_ENABLED="True"
export MVGAR_DIAGNOSTIC_ONLY="False"
export LAMBDA_MVGAR_SURFACE="0.01"
export MVGAR_DENSIFICATION_ENABLED="True"
export MVGAR_MAX_EXTRA_FRACTION_PER_REFINE="0.002"
exec "${REPO_DIR}/scripts/experiments/mvgar_5k_common.sh"
