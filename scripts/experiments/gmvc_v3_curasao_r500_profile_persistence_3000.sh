#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

VARIANT="${VARIANT:-P30}"
GPU="${GPU:-6}"
LOAD_STEP="${LOAD_STEP:-10000}"
TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-13000}"
STAMP="${STAMP:-20260805_gmvc_v3_curasao_r500_profile_persistence_3k}"

COMMON_ENV=(
  GPU="${GPU}"
  LOAD_STEP="${LOAD_STEP}"
  TARGET_FINAL_STEP="${TARGET_FINAL_STEP}"
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-3000}"
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${TARGET_FINAL_STEP}}"
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-1000}"
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
  RUN_EVAL="${RUN_EVAL:-0}"
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-49}"
)

case "${VARIANT}" in
  A0|a0)
    SLUG="a0"
    EXPERIMENT_NAME_DEFAULT="gmvc_v3_a0_profile_persistence3k_curasao_seed42_step${LOAD_STEP}_to_${TARGET_FINAL_STEP}"
    exec env \
      "${COMMON_ENV[@]}" \
      SCENE="curasao" \
      VARIANT="A0" \
      STAMP="${STAMP}_${SLUG}" \
      EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_DEFAULT}}" \
      "${REPO_DIR}/scripts/experiments/gmvc_v3_alternating_1000.sh"
    ;;
  P40|p40)
    SLUG="p40"
    PROFILE_LAMBDA="40"
    ;;
  P35|p35)
    SLUG="p35"
    PROFILE_LAMBDA="35"
    ;;
  P30|p30)
    SLUG="p30"
    PROFILE_LAMBDA="30"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use A0, P40, P35, or P30." >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME_DEFAULT="gmvc_v3_r500_${SLUG}_profile_persistence3k_curasao_seed42_step${LOAD_STEP}_to_${TARGET_FINAL_STEP}"

exec env \
  "${COMMON_ENV[@]}" \
  VARIANT="${VARIANT}" \
  STAMP="${STAMP}_${SLUG}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_DEFAULT}}" \
  LAMBDA_GMVC_PROFILE="${PROFILE_LAMBDA}" \
  LAMBDA_GMVC_OBJECT="${LAMBDA_GMVC_OBJECT:-0.004}" \
  GMVC_V3_TARGET_CURRENT_CAMERA_TRACKS="True" \
  GMVC_V3_OBJECT_PHASE_MEDIUM_GRAD_SCALE="0.00" \
  "${REPO_DIR}/scripts/experiments/gmvc_v3_curasao_r500_profile_lambda_sweep_1000.sh"
