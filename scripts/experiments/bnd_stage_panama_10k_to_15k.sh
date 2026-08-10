#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/new/home_old/ycy/water-splatting-refactor}"
NS_TRAIN="${NS_TRAIN:-/opt/anaconda3/envs/water_splatting/bin/ns-train}"

MODE="${MODE:-stage}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/bnd_stage_panama_20260810}"
LOGS_DIR="${LOGS_DIR:-${REPO_DIR}/logs/bnd_stage_panama_20260810}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Panama}"
LOAD_DIR="${LOAD_DIR:-${REPO_DIR}/outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/nerfstudio_models}"
LOAD_STEP="${LOAD_STEP:-10000}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-4999}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-15000}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-500}"
STEPS_PER_EVAL_IMAGE="${STEPS_PER_EVAL_IMAGE:-0}"
STEPS_PER_EVAL_ALL_IMAGES="${STEPS_PER_EVAL_ALL_IMAGES:-0}"
STEPS_PER_EVAL_BATCH="${STEPS_PER_EVAL_BATCH:-0}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
BOUND_LOGIT_EPS="${BOUND_LOGIT_EPS:-1e-7}"
MEDIUM_HOLD_START_STEP="${MEDIUM_HOLD_START_STEP:-10000}"
MEDIUM_HOLD_END_STEP="${MEDIUM_HOLD_END_STEP:-12500}"
MEDIUM_HOLD_AUDIT_STEPS="${MEDIUM_HOLD_AUDIT_STEPS:-10001,10500,11000,11500,12000,12500,12501,13000,14000,14999}"

case "${MODE}" in
  stage)
    RUN_TAG="bnd_stage_mh2500"
    HOLD_START="${MEDIUM_HOLD_START_STEP}"
    HOLD_END="${MEDIUM_HOLD_END_STEP}"
    ;;
  control)
    RUN_TAG="bnd_k1_rst"
    HOLD_START="-1"
    HOLD_END="-1"
    ;;
  smoke)
    RUN_TAG="bnd_stage_smoke"
    HOLD_START="${MEDIUM_HOLD_START_STEP}"
    HOLD_END="${MEDIUM_HOLD_END_STEP}"
    ;;
  *)
    echo "Unsupported MODE=${MODE}; expected stage, control, or smoke" >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-panama_${RUN_TAG}_seed${SEED}_from${LOAD_STEP}}"
TIMESTAMP="${TIMESTAMP:-20260810_${RUN_TAG}}"
AUDIT_LOG_DIR="${AUDIT_LOG_DIR:-${LOGS_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}}"

CUDA_VISIBLE_DEVICES="${GPU}" "${NS_TRAIN}" water-splatting \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${TIMESTAMP}" \
  --vis tensorboard \
  --machine.seed "${SEED}" \
  --load-dir "${LOAD_DIR}" \
  --load-step "${LOAD_STEP}" \
  --load-scheduler True \
  --max-num-iterations "${MAX_NUM_ITERATIONS}" \
  --steps-per-save "${STEPS_PER_SAVE}" \
  --steps-per-eval-image "${STEPS_PER_EVAL_IMAGE}" \
  --steps-per-eval-all-images "${STEPS_PER_EVAL_ALL_IMAGES}" \
  --steps-per-eval-batch "${STEPS_PER_EVAL_BATCH}" \
  --save-only-latest-checkpoint "${SAVE_ONLY_LATEST_CHECKPOINT}" \
  --pipeline.model.num-steps "${MODEL_NUM_STEPS}" \
  --pipeline.model.sh-degree 3 \
  --pipeline.model.medium-context-mode dir_xy_camera \
  --pipeline.model.b-inf-mode tied \
  --pipeline.model.infinite-water-enabled False \
  --pipeline.model.intrinsic-color-parameterization bounded_sh3 \
  --pipeline.model.bounded-sh-logit-eps "${BOUND_LOGIT_EPS}" \
  --pipeline.model.medium-hold-start-step "${HOLD_START}" \
  --pipeline.model.medium-hold-end-step "${HOLD_END}" \
  --pipeline.model.medium-hold-audit-log-dir "${AUDIT_LOG_DIR}" \
  --pipeline.model.medium-hold-audit-steps "${MEDIUM_HOLD_AUDIT_STEPS}" \
  colmap \
  --data "${DATA_PATH}" \
  --images-path images/ColorImage \
  --colmap-path sparse/0 \
  --downscale-factor 1
