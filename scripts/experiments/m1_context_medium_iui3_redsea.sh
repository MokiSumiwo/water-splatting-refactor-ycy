#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
NS_TRAIN="/opt/anaconda3/envs/water_splatting/bin/ns-train"
NS_EVAL="/opt/anaconda3/envs/water_splatting/bin/ns-eval"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
RUN_EVAL="${RUN_EVAL:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-m1_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}}"
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
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo "depth_context=rendered_expected_depth,detach=True,normalize=p95"
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
  --pipeline.model.medium-camera-context-scale 1.0 \
  --pipeline.model.medium-camera-context-dropout 0.0 \
  --pipeline.model.medium-depth-context-detach True \
  --pipeline.model.medium-depth-context-normalize True \
  --pipeline.model.medium-depth-context-normalize-mode p95 \
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
