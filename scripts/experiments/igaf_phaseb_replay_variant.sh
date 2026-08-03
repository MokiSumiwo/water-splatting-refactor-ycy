#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

: "${SCENE_SLUG:?Set SCENE_SLUG, e.g. japanesegradens or iui3}"
: "${LOAD_DIR:?Set LOAD_DIR to the shared step-2500 nerfstudio_models directory}"
: "${VARIANT:?Set VARIANT to r0_m1, r1_amp0, r2_nomip, or r3_variance_mip}"

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
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-2500}"
export MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-5000}"
export STEPS_PER_SAVE="${STEPS_PER_SAVE:-2500}"
export SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-True}"
export RUN_EVAL="${RUN_EVAL:-1}"
export STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-phaseb_${VARIANT}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-igaf_${EXPERIMENT_TAG}_${SCENE_SLUG}_seed${SEED:-42}_${MAX_NUM_ITERATIONS}}"
export DETERMINISTIC_AUDIT_LOG_PATH="${DETERMINISTIC_AUDIT_LOG_PATH:-${REPO_DIR}/logs/${EXPERIMENT_NAME}_${STAMP}/deterministic.jsonl}"
export DETERMINISTIC_AUDIT_LOG_EVERY="${DETERMINISTIC_AUDIT_LOG_EVERY:-100}"
export SANITIZE_LOAD_DIR="${SANITIZE_LOAD_DIR:-1}"

case "${VARIANT}" in
  r0_m1)
    export IGAF_ENABLED="False"
    ;;
  r1_amp0)
    export IGAF_ENABLED="True"
    export IGAF_START_STEP="${IGAF_START_STEP:-2500}"
    export IGAF_RAMP_STEPS="${IGAF_RAMP_STEPS:-1}"
    export IGAF_AMPLITUDE_MAX="0.0"
    export IGAF_MIP_ENABLED="False"
    ;;
  r2_nomip)
    export IGAF_ENABLED="True"
    export IGAF_START_STEP="${IGAF_START_STEP:-2500}"
    export IGAF_RAMP_STEPS="${IGAF_RAMP_STEPS:-500}"
    export IGAF_MIP_ENABLED="False"
    export IGAF_RESET_SPLIT_COEFFS="True"
    ;;
  r3_variance_mip)
    export IGAF_ENABLED="True"
    export IGAF_START_STEP="${IGAF_START_STEP:-2500}"
    export IGAF_RAMP_STEPS="${IGAF_RAMP_STEPS:-500}"
    export IGAF_MIP_ENABLED="True"
    export IGAF_MIP_MODE="variance"
    export IGAF_AXIS_MODE="locked"
    export IGAF_RESET_SPLIT_COEFFS="True"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "${SANITIZE_LOAD_DIR}" == "1" ]]; then
  SANITIZED_LOAD_DIR="${SANITIZED_LOAD_DIR:-${LOAD_DIR%/}_adam_sanitized}"
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/sanitize_adam_checkpoint.py" \
    --load-dir "${LOAD_DIR}" \
    --output-dir "${SANITIZED_LOAD_DIR}"
  export LOAD_DIR="${SANITIZED_LOAD_DIR}"
fi

exec "${REPO_DIR}/scripts/experiments/igaf_5k_common.sh"
