#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gmvc_p2_active_proxy005_japanesegradens_redsea_seed42_step10000_to_10200}"
STAMP="${STAMP:-20260803_gmvc_p2_active_proxy005_short}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

exec env \
  GPU="${GPU:-6}" \
  SCENE_SLUG="japanesegradens_redsea" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}" \
  M1_LOAD_CONFIG="${M1_LOAD_CONFIG:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml}" \
  M1_LOAD_CHECKPOINT="${M1_LOAD_CHECKPOINT:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt}" \
  LOG_ROOT="${LOG_ROOT}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  STAMP="${STAMP}" \
  TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-10200}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-10200}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-200}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-200}" \
  GMVC_VARIANT="p2_active_proxy005" \
  GMVC_NEEDS_BANK="1" \
  GMVC_ENABLED="True" \
  GMVC_DIAGNOSTIC_ONLY="False" \
  LAMBDA_GMVC_J="0.0" \
  LAMBDA_GMVC_RANGE="0.0" \
  LAMBDA_GMVC_BINF="0.0" \
  LAMBDA_GMVC_INTRINSIC="${LAMBDA_GMVC_INTRINSIC:-0.05}" \
  GMVC_INTRINSIC_SOURCE="J_proxy_raw" \
  GMVC_INTRINSIC_USE_DC_PROXY="True" \
  GMVC_GRAD_LOG_PATH="${GMVC_GRAD_LOG_PATH:-${LOG_ROOT}/${EXPERIMENT_NAME}_${STAMP}/gmvc_grad_stats.jsonl}" \
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-50}" \
  GMVC_RAMP_STEPS="${GMVC_RAMP_STEPS:-0}" \
  GMVC_FREEZE_GEOMETRY="False" \
  GMVC_TRAIN_FEATURES_DC="False" \
  GMVC_TRAIN_FEATURES_REST="False" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}" \
  "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
