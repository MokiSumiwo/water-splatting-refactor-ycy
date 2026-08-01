#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export GPU="${GPU:-9}"
export SCENE_SLUG="japanesegradens"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}"
export EXPERIMENT_TAG="c3_persistent_coherence"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-5000}"
export MCGR_ENABLED="True"
export MCGR_DIAGNOSTIC_ONLY="False"
export MCGR_GRADIENT_COHERENCE_ENABLED="True"
export MCGR_GRADIENT_COHERENCE_THRESHOLD="${MCGR_GRADIENT_COHERENCE_THRESHOLD:-0.35}"
export MCGR_MAX_EXTRA_FRACTION_PER_REFINE="${MCGR_MAX_EXTRA_FRACTION_PER_REFINE:-0.001}"
export MCGR_PERSISTENT_QUANTILE="${MCGR_PERSISTENT_QUANTILE:-0.85}"
exec "${REPO_DIR}/scripts/experiments/mcgr_5k_common.sh"
