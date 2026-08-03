#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

: "${SCENE_SLUG:?Set SCENE_SLUG, e.g. japanesegradens or iui3}"
: "${DATA_PATH:?Set DATA_PATH to the scene directory}"
: "${EXPERIMENT_TAG:?Set EXPERIMENT_TAG, e.g. o2_mip}"

if [[ -z "${LOAD_DIR:-}" ]]; then
  echo "Set LOAD_DIR to an M1 nerfstudio_models directory before running the IGAF oracle." >&2
  exit 2
fi
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
  echo "IGAF oracle uses LOAD_DIR so trainer step is restored; unset LOAD_CHECKPOINT." >&2
  exit 2
fi

export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-1000}"
export MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-16000}"
export STEPS_PER_SAVE="${STEPS_PER_SAVE:-1000}"
export IGAF_ENABLED="${IGAF_ENABLED:-True}"
export IGAF_START_STEP="${IGAF_START_STEP:-0}"
export IGAF_RAMP_STEPS="${IGAF_RAMP_STEPS:-1}"
export IGAF_FREEZE_BASE_GAUSSIANS="${IGAF_FREEZE_BASE_GAUSSIANS:-True}"
export IGAF_FREEZE_MEDIUM="${IGAF_FREEZE_MEDIUM:-True}"
export RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}"

exec "${REPO_DIR}/scripts/experiments/igaf_5k_common.sh"
