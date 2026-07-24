#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
NS_TRAIN="/opt/anaconda3/envs/water_splatting/bin/ns-train"
NS_EVAL="/opt/anaconda3/envs/water_splatting/bin/ns-eval"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
RUN_EVAL="${RUN_EVAL:-1}"
MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
OWNERSHIP_MODE="${OWNERSHIP_MODE:-alpha_depth}"
BINFGT_WEIGHT="${BINFGT_WEIGHT:-0.01}"
ACCUM_ZERO_WEIGHT="${ACCUM_ZERO_WEIGHT:-0.004}"
NEAR_ZERO_WEIGHT="${NEAR_ZERO_WEIGHT:-0.001}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-m2_${OWNERSHIP_MODE}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-${EXPERIMENT_NAME}_${STAMP}}"
LOG_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}_${STAMP}"
TRAIN_LOG="${LOG_DIR}/train.log"
EVAL_LOG="${LOG_DIR}/eval.log"
CONFIG_PATH="${OUTPUT_DIR}/${EXPERIMENT_NAME}/water-splatting/${TIMESTAMP}/config.yml"
RENDER_DIR="${RENDER_ROOT}/${EXPERIMENT_NAME}_${STAMP}"

mkdir -p "${LOG_DIR}" "${RENDER_DIR}"

{
  echo "experiment=${EXPERIMENT_NAME}"
  echo "timestamp=${TIMESTAMP}"
  echo "gpu=${GPU}"
  echo "data=${DATA_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "render_dir=${RENDER_DIR}"
  echo "medium_context_mode=${MEDIUM_CONTEXT_MODE}"
  echo "ownership_mode=${OWNERSHIP_MODE}"
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo "binf_rgb_weight=${BINFGT_WEIGHT}"
  echo "accumulation_zero_weight=${ACCUM_ZERO_WEIGHT}"
  echo "near_zero_weight=${NEAR_ZERO_WEIGHT}"
  git -C "${REPO_DIR}" rev-parse HEAD || true
  git -C "${REPO_DIR}" status --short || true
} | tee "${LOG_DIR}/run_manifest.txt"

CUDA_VISIBLE_DEVICES="${GPU}" "${NS_TRAIN}" water-splatting \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${TIMESTAMP}" \
  --vis tensorboard \
  --max-num-iterations "${MAX_NUM_ITERATIONS}" \
  --pipeline.model.num-steps "${MAX_NUM_ITERATIONS}" \
  --pipeline.model.medium-context-mode "${MEDIUM_CONTEXT_MODE}" \
  --pipeline.model.infinite-water-enabled True \
  --pipeline.model.infinite-water-ownership-mode "${OWNERSHIP_MODE}" \
  --pipeline.model.infinite-water-detach-evidence True \
  --pipeline.model.infinite-water-occupancy-limited True \
  --pipeline.model.infinite-water-loss-start-step 1000 \
  --pipeline.model.infinite-water-loss-ramp-steps 3000 \
  --pipeline.model.lambda-infinite-water-binf-rgb "${BINFGT_WEIGHT}" \
  --pipeline.model.lambda-infinite-water-accumulation-zero "${ACCUM_ZERO_WEIGHT}" \
  --pipeline.model.lambda-infinite-water-near-zero "${NEAR_ZERO_WEIGHT}" \
  colmap \
  --data "${DATA_PATH}" \
  --images-path images/ColorImage \
  --colmap-path sparse/0 \
  --downscale-factor 1 \
  2>&1 | tee "${TRAIN_LOG}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  pushd "${RENDER_DIR}" >/dev/null
  CUDA_VISIBLE_DEVICES="${GPU}" "${NS_EVAL}" \
    --load-config "${CONFIG_PATH}" \
    --render-output-path "${RENDER_DIR}" \
    2>&1 | tee "${EVAL_LOG}"
  popd >/dev/null
fi
