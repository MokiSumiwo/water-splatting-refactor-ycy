#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/new/home_old/ycy/water-splatting-refactor}"
NS_TRAIN="${NS_TRAIN:-/opt/anaconda3/envs/water_splatting/bin/ns-train}"

: "${SCENE:?Set SCENE, e.g. curasao}"
: "${DATA_PATH:?Set DATA_PATH to the scene directory}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${MAX_NUM_ITERATIONS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
STAMP="${STAMP:-m1_legacy_$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-m1_legacy_${SCENE}_seed${SEED}_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-${EXPERIMENT_NAME}_${STAMP}}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"

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
  --pipeline.model.intrinsic-color-parameterization legacy \
  colmap \
  --data "${DATA_PATH}" \
  --images-path images/ColorImage \
  --colmap-path sparse/0 \
  --downscale-factor 1
