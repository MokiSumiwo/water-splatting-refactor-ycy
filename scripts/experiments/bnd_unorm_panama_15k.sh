#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/new/home_old/ycy/water-splatting-refactor}"
NS_TRAIN="${NS_TRAIN:-/opt/anaconda3/envs/water_splatting/bin/ns-train}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/bnd_unorm_panama_20260810}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Panama}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${MAX_NUM_ITERATIONS}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-panama_bnd_unorm_seed${SEED}_step0_to_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-20260810_bnd_unorm}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-1000}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
BOUND_LOGIT_EPS="${BOUND_LOGIT_EPS:-1e-7}"

CUDA_VISIBLE_DEVICES="${GPU}" "${NS_TRAIN}" water-splatting \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${TIMESTAMP}" \
  --vis tensorboard \
  --machine.seed "${SEED}" \
  --max-num-iterations "${MAX_NUM_ITERATIONS}" \
  --steps-per-save "${STEPS_PER_SAVE}" \
  --save-only-latest-checkpoint "${SAVE_ONLY_LATEST_CHECKPOINT}" \
  --pipeline.model.num-steps "${MODEL_NUM_STEPS}" \
  --pipeline.model.sh-degree 3 \
  --pipeline.model.medium-context-mode dir_xy_camera \
  --pipeline.model.b-inf-mode tied \
  --pipeline.model.infinite-water-enabled False \
  --pipeline.model.intrinsic-color-parameterization bounded_sh3 \
  --pipeline.model.photometric-normalization-mode absolute \
  --pipeline.model.bounded-sh-logit-eps "${BOUND_LOGIT_EPS}" \
  --pipeline.model.rasterize-mode classic \
  colmap \
  --data "${DATA_PATH}" \
  --images-path images/ColorImage \
  --colmap-path sparse/0 \
  --downscale-factor 1
