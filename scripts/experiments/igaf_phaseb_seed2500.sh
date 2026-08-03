#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

: "${SCENE_SLUG:?Set SCENE_SLUG, e.g. japanesegradens or iui3}"
case "${SCENE_SLUG}" in
  japanesegradens)
    export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}"
    ;;
  iui3)
    export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
    ;;
  *)
    : "${DATA_PATH:?Set DATA_PATH for custom scene}"
    ;;
esac

export GPU="${GPU:-6}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-phaseb_seed2500_m1}"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-2500}"
export MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-5000}"
export STEPS_PER_SAVE="${STEPS_PER_SAVE:-2500}"
export SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-True}"
export RUN_EVAL="${RUN_EVAL:-0}"
export IGAF_ENABLED="False"

exec "${REPO_DIR}/scripts/experiments/igaf_5k_common.sh"
