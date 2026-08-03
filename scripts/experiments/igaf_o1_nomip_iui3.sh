#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-7}"
export SCENE_SLUG="iui3"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
export EXPERIMENT_TAG="o1_nomip"
export IGAF_MIP_ENABLED="False"
exec "${REPO_DIR}/scripts/experiments/igaf_oracle_common.sh"
