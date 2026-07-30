#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

: "${SCENE_SLUG:?Set SCENE_SLUG, e.g. curasao}"
: "${DATA_PATH:?Set DATA_PATH to the scene directory}"

GPU="${GPU:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
SEED="${SEED:-42}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${MAX_NUM_ITERATIONS}}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
STAMP="${STAMP:-20260730_cross_scene}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cross_scene_${SCENE_SLUG}_m1_seed${SEED}_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-${EXPERIMENT_NAME}_${STAMP}}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-1}"
BUILD_MASKS="${BUILD_MASKS:-1}"
RUN_POST_MASK_DIAGS="${RUN_POST_MASK_DIAGS:-1}"

CONFIG_PATH="${OUTPUT_DIR}/${EXPERIMENT_NAME}/water-splatting/${TIMESTAMP}/config.yml"
RENDER_DIR="${RENDER_ROOT}/${EXPERIMENT_NAME}_${STAMP}"
DIAG_DIR="${RENDER_DIR}/diagnostics"
LOG_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}_${STAMP}"
COMMON_FAR_MASK_DIR="${COMMON_FAR_MASK_DIR:-${REPO_DIR}/common_masks/cross_scene_${SCENE_SLUG}_m1_q90_seed${SEED}_${STAMP}}"
REGION_MASK_DIR="${REGION_MASK_DIR:-${REPO_DIR}/common_masks/cross_scene_${SCENE_SLUG}_m1_eval_regions_seed${SEED}_${STAMP}}"

mkdir -p "${LOG_DIR}" "${DIAG_DIR}"

{
  echo "cross_scene_method=m1"
  echo "scene_slug=${SCENE_SLUG}"
  echo "experiment=${EXPERIMENT_NAME}"
  echo "timestamp=${TIMESTAMP}"
  echo "gpu=${GPU}"
  echo "python=${PYTHON}"
  echo "data=${DATA_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "render_dir=${RENDER_DIR}"
  echo "config_path=${CONFIG_PATH}"
  echo "seed=${SEED}"
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo "model_num_steps=${MODEL_NUM_STEPS}"
  echo "steps_per_save=${STEPS_PER_SAVE}"
  echo "save_only_latest_checkpoint=${SAVE_ONLY_LATEST_CHECKPOINT}"
  echo "common_far_mask_dir=${COMMON_FAR_MASK_DIR}"
  echo "region_mask_dir=${REGION_MASK_DIR}"
} | tee "${LOG_DIR}/cross_scene_manifest.txt"

env \
  GPU="${GPU}" \
  PYTHON="${PYTHON}" \
  DATA_PATH="${DATA_PATH}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RENDER_ROOT="${RENDER_ROOT}" \
  LOG_ROOT="${LOG_ROOT}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  TIMESTAMP="${TIMESTAMP}" \
  STAMP="${STAMP}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT}" \
  SEED="${SEED}" \
  MEDIUM_CONTEXT_MODE="dir_xy_camera" \
  BINF_MODE="tied" \
  BG_COLOR_WEIGHT="0.0" \
  BG_MEDIUM_RENDER_WEIGHT="0.0" \
  BG_TAIL_RENDER_WEIGHT="0.0" \
  BG_CLEAR_GAUSSIAN_WEIGHT="0.0" \
  BG_CLEAR_CHROMA_WEIGHT="0.0" \
  MEDIUM_EXPLAINABILITY_ENABLED="False" \
  LAMBDA_MEDIUM_EXPLAINABILITY="0.0" \
  TRAINING_GRADIENT_ROUTING_ENABLED="False" \
  BUDGETED_CAPACITY_ENABLED="False" \
  LAMBDA_BUDGETED_CAPACITY="0.0" \
  HALO_CAPACITY_ENABLED="False" \
  LAMBDA_HALO_CAPACITY="0.0" \
  LAMBDA_PROXY_CLEAR_LUMA="0.0" \
  BACKGROUND_DENSIFICATION_ENABLED="False" \
  BACKGROUND_DENSIFICATION_DIAGNOSTIC_ONLY="True" \
  RUN_EVAL="${RUN_EVAL}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG}" \
  RUN_FAR_DIAG="0" \
  RUN_REGION_DIAG="0" \
  "${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"

if [[ "${BUILD_MASKS}" == "1" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Missing M1 config: ${CONFIG_PATH}" >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/build_common_far_masks.py" \
    --load-config "${CONFIG_PATH}" \
    --output-dir "${COMMON_FAR_MASK_DIR}" \
    --far-depth-quantile 0.90 \
    --save-png \
    2>&1 | tee "${LOG_DIR}/build_common_far_masks.log"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/build_eval_region_masks.py" \
    --load-config "${CONFIG_PATH}" \
    --output-dir "${REGION_MASK_DIR}" \
    --max-images 4 \
    --save-png \
    2>&1 | tee "${LOG_DIR}/build_eval_region_masks.log"
fi

if [[ "${RUN_POST_MASK_DIAGS}" == "1" ]]; then
  if [[ ! -d "${COMMON_FAR_MASK_DIR}" || ! -d "${REGION_MASK_DIR}" ]]; then
    echo "Missing M1-derived mask dirs for diagnostics." >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_far_water_residual.py" \
    --load-config "${CONFIG_PATH}" \
    --mask-dir "${COMMON_FAR_MASK_DIR}" \
    --output-dir "${DIAG_DIR}/far_water" \
    2>&1 | tee "${LOG_DIR}/far_diag.log"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_eval_regions.py" \
    --load-config "${CONFIG_PATH}" \
    --reference-config "${CONFIG_PATH}" \
    --mask-dir "${REGION_MASK_DIR}" \
    --output-dir "${DIAG_DIR}/eval_regions" \
    --max-images 4 \
    2>&1 | tee "${LOG_DIR}/region_diag.log"
fi

