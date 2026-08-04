#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

VARIANT="${VARIANT:-G100}"
GPU="${GPU:-6}"
STAMP="${STAMP:-20260804_gmvc_v3_curasao_medium_grad_sweep_1000}"

case "${VARIANT}" in
  G100|g100)
    SCALE="1.00"
    SLUG="g100"
    FREEZE="False"
    ;;
  G075|g075)
    SCALE="0.75"
    SLUG="g075"
    FREEZE="False"
    ;;
  G050|g050)
    SCALE="0.50"
    SLUG="g050"
    FREEZE="False"
    ;;
  G025|g025)
    SCALE="0.25"
    SLUG="g025"
    FREEZE="False"
    ;;
  G000|g000)
    SCALE="0.00"
    SLUG="g000"
    FREEZE="False"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use G100, G075, G050, G025, or G000." >&2
    exit 2
    ;;
esac

exec env \
  SCENE="curasao" \
  VARIANT="A2" \
  GPU="${GPU}" \
  LOAD_STEP="${LOAD_STEP:-10000}" \
  TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-11000}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-1000}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-11000}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-1000}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-True}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  STAMP="${STAMP}_${SLUG}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-gmvc_v3_${SLUG}_object_medium_grad_curasao_seed42_step10000_to_11000}" \
  LAMBDA_GMVC_PROFILE="${LAMBDA_GMVC_PROFILE:-40}" \
  LAMBDA_GMVC_OBJECT="${LAMBDA_GMVC_OBJECT:-0.004}" \
  GMVC_V3_TARGET_CURRENT_CAMERA_TRACKS="True" \
  GMVC_V3_FREEZE_MEDIUM_ON_OBJECT_PHASE="${FREEZE}" \
  GMVC_V3_OBJECT_PHASE_MEDIUM_GRAD_SCALE="${SCALE}" \
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-49}" \
  "${REPO_DIR}/scripts/experiments/gmvc_v3_alternating_1000.sh"
