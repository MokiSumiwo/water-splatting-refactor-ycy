#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

SCENE="${SCENE:-panama}"
VARIANT="${VARIANT:-V0}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
LOAD_STEP="${LOAD_STEP:-10000}"
TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-10500}"
STAMP="${STAMP:-20260804_gmvc_v1_online_500}"

case "${SCENE}" in
  japanesegradens|JapaneseGradens)
    SCENE_SLUG="japanesegradens_redsea"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    CLOSURE_LOW_DEFAULT="0.0005"
    CLOSURE_MID_DEFAULT="0.0015"
    ;;
  panama|Panama)
    SCENE_SLUG="panama"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Panama"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    CLOSURE_LOW_DEFAULT="0.0010"
    CLOSURE_MID_DEFAULT="0.0025"
    ;;
  *)
    echo "Unknown SCENE=${SCENE}. Use panama or japanesegradens." >&2
    exit 2
    ;;
esac

BUDGET_DEFAULT="${BUDGET_DEFAULT:-0.0010}"
CLOSURE_LOW="${CLOSURE_LOW:-${CLOSURE_LOW_DEFAULT}}"
CLOSURE_MID="${CLOSURE_MID:-${CLOSURE_MID_DEFAULT}}"

GMVC_ENABLED_VALUE="True"
GMVC_NEEDS_BANK_VALUE="1"
LAMBDA_BUDGET="0.0"
LAMBDA_CLOSURE="0.0"
VARIANT_SLUG="$(echo "${VARIANT}" | tr '[:upper:]' '[:lower:]')"

case "${VARIANT}" in
  V0|v0)
    GMVC_ENABLED_VALUE="False"
    GMVC_NEEDS_BANK_VALUE="0"
    VARIANT_SLUG="v0_m1"
    ;;
  V1|v1)
    LAMBDA_BUDGET="${LAMBDA_GMVC_RESIDUAL_BUDGET:-${BUDGET_DEFAULT}}"
    VARIANT_SLUG="v1_budget"
    ;;
  V2|v2)
    LAMBDA_CLOSURE="${LAMBDA_GMVC_FIXED_CLOSURE:-${CLOSURE_LOW}}"
    VARIANT_SLUG="v2_fixed_closure"
    ;;
  V3|v3)
    LAMBDA_BUDGET="${LAMBDA_GMVC_RESIDUAL_BUDGET:-${BUDGET_DEFAULT}}"
    LAMBDA_CLOSURE="${LAMBDA_GMVC_FIXED_CLOSURE:-${CLOSURE_LOW}}"
    VARIANT_SLUG="v3_budget_closure_low"
    ;;
  V4|v4)
    LAMBDA_BUDGET="${LAMBDA_GMVC_RESIDUAL_BUDGET:-${BUDGET_DEFAULT}}"
    LAMBDA_CLOSURE="${LAMBDA_GMVC_FIXED_CLOSURE:-${CLOSURE_MID}}"
    VARIANT_SLUG="v4_budget_closure_mid"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use V0, V1, V2, V3, or V4." >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME_DEFAULT="gmvc_v1_${VARIANT_SLUG}_${SCENE_SLUG}_seed${SEED}_step${LOAD_STEP}_to_${TARGET_FINAL_STEP}"
BANK_PATH_DEFAULT="${REPO_DIR}/renders/gmvc_v1_track_banks/${SCENE_SLUG}_m1_step${LOAD_STEP}_train_s4096/gmvc_track_bank.pt"

exec env \
  GPU="${GPU}" \
  SCENE_SLUG="${SCENE_SLUG}" \
  DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}" \
  M1_LOAD_CONFIG="${M1_LOAD_CONFIG:-${M1_CONFIG_DEFAULT}}" \
  M1_LOAD_CHECKPOINT="${M1_LOAD_CHECKPOINT:-${M1_CKPT_DEFAULT}}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_DEFAULT}}" \
  STAMP="${STAMP}" \
  TARGET_FINAL_STEP="${TARGET_FINAL_STEP}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${TARGET_FINAL_STEP}}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-$((TARGET_FINAL_STEP - LOAD_STEP))}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-500}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}" \
  LOAD_STEP="${LOAD_STEP}" \
  GMVC_VARIANT="${VARIANT_SLUG}" \
  GMVC_TRACK_BANK_PATH="${GMVC_TRACK_BANK_PATH:-${BANK_PATH_DEFAULT}}" \
  GMVC_NEEDS_BANK="${GMVC_NEEDS_BANK:-${GMVC_NEEDS_BANK_VALUE}}" \
  GMVC_ENABLED="${GMVC_ENABLED:-${GMVC_ENABLED_VALUE}}" \
  GMVC_DIAGNOSTIC_ONLY="${GMVC_DIAGNOSTIC_ONLY:-False}" \
  GMVC_START_STEP="${GMVC_START_STEP:-10000}" \
  GMVC_STOP_STEP="${GMVC_STOP_STEP:-10500}" \
  GMVC_RAMP_STEPS="${GMVC_RAMP_STEPS:-100}" \
  LAMBDA_GMVC_J="0.0" \
  LAMBDA_GMVC_RANGE="0.0" \
  LAMBDA_GMVC_BINF="0.0" \
  LAMBDA_GMVC_INTRINSIC="0.0" \
  LAMBDA_GMVC_RESIDUAL_BUDGET="${LAMBDA_BUDGET}" \
  LAMBDA_GMVC_FIXED_CLOSURE="${LAMBDA_CLOSURE}" \
  GMVC_RESIDUAL_BETA_LOG_SCALE="${GMVC_RESIDUAL_BETA_LOG_SCALE:-0.15}" \
  GMVC_RESIDUAL_BINF_LOGIT_SCALE="${GMVC_RESIDUAL_BINF_LOGIT_SCALE:-0.10}" \
  GMVC_RESIDUAL_EMA_MOMENTUM="${GMVC_RESIDUAL_EMA_MOMENTUM:-0.99}" \
  GMVC_CLOSURE_SIGNAL_FLOOR="${GMVC_CLOSURE_SIGNAL_FLOOR:-0.03}" \
  GMVC_MAX_TRACKS_PER_STEP="${GMVC_MAX_TRACKS_PER_STEP:-4096}" \
  GMVC_FREEZE_GEOMETRY="False" \
  GMVC_TRAIN_FEATURES_DC="False" \
  GMVC_TRAIN_FEATURES_REST="False" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}" \
  "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
