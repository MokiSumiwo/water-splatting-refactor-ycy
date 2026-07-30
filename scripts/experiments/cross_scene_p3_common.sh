#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

: "${SCENE_SLUG:?Set SCENE_SLUG, e.g. curasao}"
: "${DATA_PATH:?Set DATA_PATH to the scene directory}"

GPU="${GPU:-8}"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
SEED="${SEED:-42}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${MAX_NUM_ITERATIONS}}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
STAMP="${STAMP:-20260730_cross_scene}"
M1_STAMP="${M1_STAMP:-${STAMP}}"
M1_EXPERIMENT_NAME="${M1_EXPERIMENT_NAME:-cross_scene_${SCENE_SLUG}_m1_seed${SEED}_${MAX_NUM_ITERATIONS}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cross_scene_${SCENE_SLUG}_p3_seed${SEED}_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-${EXPERIMENT_NAME}_${STAMP}}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-1}"
RUN_FAR_DIAG="${RUN_FAR_DIAG:-1}"
RUN_REGION_DIAG="${RUN_REGION_DIAG:-1}"

REFERENCE_CONFIG="${REFERENCE_CONFIG:-${OUTPUT_DIR}/${M1_EXPERIMENT_NAME}/water-splatting/${M1_EXPERIMENT_NAME}_${M1_STAMP}/config.yml}"
COMMON_FAR_MASK_DIR="${COMMON_FAR_MASK_DIR:-${REPO_DIR}/common_masks/cross_scene_${SCENE_SLUG}_m1_q90_seed${SEED}_${M1_STAMP}}"
REGION_MASK_DIR="${REGION_MASK_DIR:-${REPO_DIR}/common_masks/cross_scene_${SCENE_SLUG}_m1_eval_regions_seed${SEED}_${M1_STAMP}}"
LOG_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}_${STAMP}"

mkdir -p "${LOG_DIR}"

if [[ "${RUN_FAR_DIAG}" == "1" && ! -d "${COMMON_FAR_MASK_DIR}" ]]; then
  echo "Missing common far mask dir: ${COMMON_FAR_MASK_DIR}" >&2
  exit 1
fi
if [[ "${RUN_REGION_DIAG}" == "1" ]]; then
  if [[ ! -f "${REFERENCE_CONFIG}" ]]; then
    echo "Missing scene M1 reference config: ${REFERENCE_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -d "${REGION_MASK_DIR}" ]]; then
    echo "Missing eval region mask dir: ${REGION_MASK_DIR}" >&2
    exit 1
  fi
fi

{
  echo "cross_scene_method=p3"
  echo "scene_slug=${SCENE_SLUG}"
  echo "experiment=${EXPERIMENT_NAME}"
  echo "timestamp=${TIMESTAMP}"
  echo "gpu=${GPU}"
  echo "python=${PYTHON}"
  echo "data=${DATA_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "render_root=${RENDER_ROOT}"
  echo "seed=${SEED}"
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo "model_num_steps=${MODEL_NUM_STEPS}"
  echo "steps_per_save=${STEPS_PER_SAVE}"
  echo "save_only_latest_checkpoint=${SAVE_ONLY_LATEST_CHECKPOINT}"
  echo "reference_config=${REFERENCE_CONFIG}"
  echo "common_far_mask_dir=${COMMON_FAR_MASK_DIR}"
  echo "region_mask_dir=${REGION_MASK_DIR}"
} | tee "${LOG_DIR}/cross_scene_manifest.txt"

exec env \
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
  REFERENCE_CONFIG="${REFERENCE_CONFIG}" \
  COMMON_FAR_MASK_DIR="${COMMON_FAR_MASK_DIR}" \
  REGION_MASK_DIR="${REGION_MASK_DIR}" \
  RUN_EVAL="${RUN_EVAL}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG}" \
  RUN_FAR_DIAG="${RUN_FAR_DIAG}" \
  RUN_REGION_DIAG="${RUN_REGION_DIAG}" \
  "${REPO_DIR}/scripts/experiments/medium_attr_p3_b02_proxy_geom000_opacity050_iui3.sh"

