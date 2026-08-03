#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

: "${SCENE_SLUG:?Set SCENE_SLUG, e.g. japanesegradens or iui3}"
: "${VARIANT:?Set VARIANT to m0_adc, m1_reloc_birth, m2_reg, or m3_sgld}"

case "${SCENE_SLUG}" in
  japanesegradens)
    export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}"
    DEFAULT_LOAD_DIR="${REPO_DIR}/outputs/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500/water-splatting/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500_20260803_gdadc_seed500_jg/nerfstudio_models_adam_sanitized"
    export MCMC_CAP_MAX="${MCMC_CAP_MAX:-657221}"
    ;;
  iui3)
    export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
    : "${LOAD_DIR:?Set LOAD_DIR to the IUI3 shared step-500 nerfstudio_models directory}"
    DEFAULT_LOAD_DIR="${LOAD_DIR}"
    export MCMC_CAP_MAX="${MCMC_CAP_MAX:--1}"
    ;;
  *)
    : "${DATA_PATH:?Set DATA_PATH for custom scene}"
    : "${LOAD_DIR:?Set LOAD_DIR to the shared step-500 nerfstudio_models directory}"
    DEFAULT_LOAD_DIR="${LOAD_DIR}"
    export MCMC_CAP_MAX="${MCMC_CAP_MAX:--1}"
    ;;
esac

export LOAD_DIR="${LOAD_DIR:-${DEFAULT_LOAD_DIR}}"
export GPU="${GPU:-6}"
export SEED="${SEED:-42}"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-4500}"
export MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-5000}"
export STEPS_PER_SAVE="${STEPS_PER_SAVE:-4500}"
export SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-True}"
export RUN_EVAL="${RUN_EVAL:-1}"
export RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}"
export STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-${VARIANT}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mcmc_ws_${EXPERIMENT_TAG}_${SCENE_SLUG}_seed${SEED}_${MODEL_NUM_STEPS}}"
export MCMC_LOG_PATH="${MCMC_LOG_PATH:-${REPO_DIR}/logs/${EXPERIMENT_NAME}_${STAMP}/mcmc.jsonl}"
export DETERMINISTIC_AUDIT_LOG_PATH="${DETERMINISTIC_AUDIT_LOG_PATH:-${REPO_DIR}/logs/${EXPERIMENT_NAME}_${STAMP}/deterministic.jsonl}"
export DETERMINISTIC_AUDIT_LOG_EVERY="${DETERMINISTIC_AUDIT_LOG_EVERY:-100}"

# Keep previous module lines explicitly disabled for the MCMC backbone test.
export GDADC_ENABLED="False"
export GDADC_DIAGNOSTIC_ONLY="True"
export GDADC_LOG_PATH=""
export IGAF_ENABLED="False"
export GIVAR_ENABLED="False"
export MVGAR_ENABLED="False"
export MCGR_ENABLED="False"
export BACKGROUND_DENSIFICATION_ENABLED="False"
export GAUSSIAN_CLEANUP_ENABLED="False"
export LAMBDA_PSEUDO_DEPTH="0.0"
export LAMBDA_MEDIUM_CONTEXT_RESIDUAL="0.0"

export MCMC_START_STEP="${MCMC_START_STEP:-500}"
export MCMC_STOP_STEP="${MCMC_STOP_STEP:-5000}"
export MCMC_INTERVAL="${MCMC_INTERVAL:-100}"
export MCMC_DEAD_OPACITY_THRESHOLD="${MCMC_DEAD_OPACITY_THRESHOLD:-0.005}"
export MCMC_GROWTH_RATE="${MCMC_GROWTH_RATE:-0.05}"
export MCMC_NOISE_SCALE="${MCMC_NOISE_SCALE:-1.0}"
export MCMC_NOISE_LR="${MCMC_NOISE_LR:-0.00016}"
export MCMC_NOISE_OPACITY_MID="${MCMC_NOISE_OPACITY_MID:-0.995}"
export MCMC_NOISE_OPACITY_TEMPERATURE="${MCMC_NOISE_OPACITY_TEMPERATURE:-0.01}"

case "${VARIANT}" in
  m0_adc)
    export MCMC_ENABLED="False"
    export MCMC_SGLD_ENABLED="False"
    export LAMBDA_MCMC_OPACITY="0.0"
    export LAMBDA_MCMC_SCALE="0.0"
    export MCMC_LOG_PATH=""
    ;;
  m1_reloc_birth)
    export MCMC_ENABLED="True"
    export MCMC_SGLD_ENABLED="False"
    export LAMBDA_MCMC_OPACITY="0.0"
    export LAMBDA_MCMC_SCALE="0.0"
    ;;
  m2_reg)
    export MCMC_ENABLED="True"
    export MCMC_SGLD_ENABLED="False"
    export LAMBDA_MCMC_OPACITY="${LAMBDA_MCMC_OPACITY:-0.001}"
    export LAMBDA_MCMC_SCALE="${LAMBDA_MCMC_SCALE:-0.001}"
    ;;
  m3_sgld)
    export MCMC_ENABLED="True"
    export MCMC_SGLD_ENABLED="True"
    export LAMBDA_MCMC_OPACITY="${LAMBDA_MCMC_OPACITY:-0.001}"
    export LAMBDA_MCMC_SCALE="${LAMBDA_MCMC_SCALE:-0.001}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "${SANITIZE_LOAD_DIR:-0}" == "1" ]]; then
  SANITIZED_LOAD_DIR="${SANITIZED_LOAD_DIR:-${LOAD_DIR%/}_adam_sanitized}"
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/sanitize_adam_checkpoint.py" \
    --load-dir "${LOAD_DIR}" \
    --output-dir "${SANITIZED_LOAD_DIR}"
  export LOAD_DIR="${SANITIZED_LOAD_DIR}"
fi

exec "${REPO_DIR}/scripts/experiments/igaf_5k_common.sh"
