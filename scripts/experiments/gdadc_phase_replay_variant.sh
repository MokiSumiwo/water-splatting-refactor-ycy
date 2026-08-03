#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

: "${SCENE_SLUG:?Set SCENE_SLUG, e.g. japanesegradens or iui3}"
: "${LOAD_DIR:?Set LOAD_DIR to the shared step-500 nerfstudio_models directory}"
: "${VARIANT:?Set VARIANT to d0_m1, d1_diag, d2_split, or d3_split_clone}"

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
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-4500}"
export MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-5000}"
export STEPS_PER_SAVE="${STEPS_PER_SAVE:-4500}"
export SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-True}"
export RUN_EVAL="${RUN_EVAL:-1}"
export STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-gdadc_${VARIANT}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gdadc_${EXPERIMENT_TAG}_${SCENE_SLUG}_seed${SEED:-42}_${MAX_NUM_ITERATIONS}}"
export DETERMINISTIC_AUDIT_LOG_PATH="${DETERMINISTIC_AUDIT_LOG_PATH:-${REPO_DIR}/logs/${EXPERIMENT_NAME}_${STAMP}/deterministic.jsonl}"
export DETERMINISTIC_AUDIT_LOG_EVERY="${DETERMINISTIC_AUDIT_LOG_EVERY:-100}"
export GDADC_LOG_PATH="${GDADC_LOG_PATH:-${REPO_DIR}/logs/${EXPERIMENT_NAME}_${STAMP}/gdadc.jsonl}"
export IGAF_ENABLED="False"

case "${VARIANT}" in
  d0_m1)
    export GDADC_ENABLED="False"
    export GDADC_LOG_PATH=""
    ;;
  d1_diag)
    export GDADC_ENABLED="True"
    export GDADC_DIAGNOSTIC_ONLY="True"
    ;;
  d2_split)
    export GDADC_ENABLED="True"
    export GDADC_DIAGNOSTIC_ONLY="False"
    export GDADC_SPLIT_ENABLED="True"
    export GDADC_CLONE_ENABLED="False"
    export GDADC_SPLIT_GRAD_THRESH="${GDADC_SPLIT_GRAD_THRESH:-0.000305}"
    ;;
  d3_split_clone)
    export GDADC_ENABLED="True"
    export GDADC_DIAGNOSTIC_ONLY="False"
    export GDADC_SPLIT_ENABLED="True"
    export GDADC_CLONE_ENABLED="True"
    export GDADC_SPLIT_GRAD_THRESH="${GDADC_SPLIT_GRAD_THRESH:-0.000305}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "${SANITIZE_LOAD_DIR:-1}" == "1" ]]; then
  SANITIZED_LOAD_DIR="${SANITIZED_LOAD_DIR:-${LOAD_DIR%/}_adam_sanitized}"
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/sanitize_adam_checkpoint.py" \
    --load-dir "${LOAD_DIR}" \
    --output-dir "${SANITIZED_LOAD_DIR}"
  export LOAD_DIR="${SANITIZED_LOAD_DIR}"
fi

exec "${REPO_DIR}/scripts/experiments/igaf_5k_common.sh"
