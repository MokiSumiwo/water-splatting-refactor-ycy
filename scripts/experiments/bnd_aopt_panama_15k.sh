#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/new/home_old/ycy/water-splatting-refactor}"
NS_TRAIN="${NS_TRAIN:-/opt/anaconda3/envs/water_splatting/bin/ns-train}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
AOPT_SCALE="${AOPT_SCALE:?Set AOPT_SCALE to 2 or 4}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${MAX_NUM_ITERATIONS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/bnd_aopt_equivalence_panama_20260809}"
LOGS_DIR="${LOGS_DIR:-${REPO_DIR}/logs/bnd_aopt_equivalence_panama_20260809}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Panama}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-1000}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
BOUND_LOGIT_EPS="${BOUND_LOGIT_EPS:-1e-7}"

AOPT_TAG="${AOPT_TAG:-k${AOPT_SCALE//./p}}"
STAMP="${STAMP:-20260809_bnd_aopt_${AOPT_TAG}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bnd_aopt_panama_seed${SEED}_${AOPT_TAG}_step0_to_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-${EXPERIMENT_NAME}_${STAMP}}"
AOPT_AUDIT_LOG_DIR="${AOPT_AUDIT_LOG_DIR:-${LOGS_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}}"

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
  --pipeline.model.bounded-sh-logit-eps "${BOUND_LOGIT_EPS}" \
  --pipeline.model.appearance-lr-scale "${AOPT_SCALE}" \
  --pipeline.model.appearance-audit-log-dir "${AOPT_AUDIT_LOG_DIR}" \
  colmap \
  --data "${DATA_PATH}" \
  --images-path images/ColorImage \
  --colmap-path sparse/0 \
  --downscale-factor 1
