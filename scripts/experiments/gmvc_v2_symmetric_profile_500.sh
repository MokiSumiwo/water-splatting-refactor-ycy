#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

SCENE="${SCENE:-curasao}"
VARIANT="${VARIANT:-S2}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
LOAD_STEP="${LOAD_STEP:-10000}"
TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-10500}"
STAMP="${STAMP:-20260804_gmvc_v2_500}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

case "${SCENE}" in
  curasao|Curasao)
    SCENE_SLUG="curasao"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Curasao"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    PROFILE_DEFAULT="0.5"
    CLOSURE_DEFAULT="0.02"
    ;;
  panama|Panama)
    SCENE_SLUG="panama"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Panama"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    PROFILE_DEFAULT="0.5"
    CLOSURE_DEFAULT="0.02"
    ;;
  japanesegradens|JapaneseGradens)
    SCENE_SLUG="japanesegradens_redsea"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    PROFILE_DEFAULT="0.75"
    CLOSURE_DEFAULT="0.015"
    ;;
  iui3|IUI3)
    SCENE_SLUG="iui3_redsea"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/nerfstudio_models/step-000010000.ckpt"
    PROFILE_DEFAULT="0.75"
    CLOSURE_DEFAULT="0.015"
    ;;
  *)
    echo "Unknown SCENE=${SCENE}. Use curasao, panama, japanesegradens, or iui3." >&2
    exit 2
    ;;
esac

GMVC_ENABLED_VALUE="True"
GMVC_NEEDS_BANK_VALUE="1"
GMVC_V2_ENABLED_VALUE="True"
LAMBDA_PROFILE="0.0"
LAMBDA_SYMMETRIC_CLOSURE="0.0"
BOUNDED_ENABLED="False"
VARIANT_SLUG="$(echo "${VARIANT}" | tr '[:upper:]' '[:lower:]')"

case "${VARIANT}" in
  S0|s0)
    GMVC_ENABLED_VALUE="False"
    GMVC_NEEDS_BANK_VALUE="0"
    GMVC_V2_ENABLED_VALUE="False"
    VARIANT_SLUG="s0_m1"
    ;;
  S1|s1)
    LAMBDA_SYMMETRIC_CLOSURE="${LAMBDA_GMVC_SYMMETRIC_CLOSURE:-${CLOSURE_DEFAULT}}"
    VARIANT_SLUG="s1_symmetric_closure"
    ;;
  S2|s2)
    LAMBDA_PROFILE="${LAMBDA_GMVC_PROFILE:-${PROFILE_DEFAULT}}"
    VARIANT_SLUG="s2_profile"
    ;;
  S3|s3)
    GMVC_ENABLED_VALUE="False"
    GMVC_NEEDS_BANK_VALUE="0"
    GMVC_V2_ENABLED_VALUE="False"
    BOUNDED_ENABLED="True"
    VARIANT_SLUG="s3_bounded"
    ;;
  S4|s4)
    LAMBDA_PROFILE="${LAMBDA_GMVC_PROFILE:-${PROFILE_DEFAULT}}"
    BOUNDED_ENABLED="True"
    VARIANT_SLUG="s4_bounded_profile"
    ;;
  S5|s5)
    LAMBDA_PROFILE="${LAMBDA_GMVC_PROFILE:-${PROFILE_DEFAULT}}"
    LAMBDA_SYMMETRIC_CLOSURE="${LAMBDA_GMVC_SYMMETRIC_CLOSURE:-${CLOSURE_DEFAULT}}"
    BOUNDED_ENABLED="True"
    VARIANT_SLUG="s5_bounded_profile_closure"
    ;;
  S6|s6)
    LAMBDA_PROFILE="${LAMBDA_GMVC_PROFILE:-${PROFILE_DEFAULT}}"
    LAMBDA_SYMMETRIC_CLOSURE="${LAMBDA_GMVC_SYMMETRIC_CLOSURE:-${CLOSURE_DEFAULT}}"
    VARIANT_SLUG="s6_profile_closure"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use S0, S1, S2, S3, S4, S5, or S6." >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME_DEFAULT="gmvc_v2_${VARIANT_SLUG}_${SCENE_SLUG}_seed${SEED}_step${LOAD_STEP}_to_${TARGET_FINAL_STEP}"
BANK_PATH_DEFAULT="${REPO_DIR}/renders/gmvc_v2_track_banks/${SCENE_SLUG}_m1_step${LOAD_STEP}_train_s4096/gmvc_track_bank.pt"
GRAD_LOG_DEFAULT="${LOG_ROOT}/gmvc_v2_grad_${VARIANT_SLUG}_${SCENE_SLUG}_${STAMP}.jsonl"

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
  GMVC_TRACK_SAMPLES_PER_VIEW="${GMVC_TRACK_SAMPLES_PER_VIEW:-4096}" \
  GMVC_TRACK_MAX_OBS_PER_CAMERA="${GMVC_TRACK_MAX_OBS_PER_CAMERA:-20000}" \
  GMVC_ENABLED="${GMVC_ENABLED:-${GMVC_ENABLED_VALUE}}" \
  GMVC_DIAGNOSTIC_ONLY="${GMVC_DIAGNOSTIC_ONLY:-False}" \
  GMVC_START_STEP="${GMVC_START_STEP:-10000}" \
  GMVC_STOP_STEP="${GMVC_STOP_STEP:-${TARGET_FINAL_STEP}}" \
  GMVC_RAMP_STEPS="${GMVC_RAMP_STEPS:-100}" \
  LAMBDA_GMVC_J="0.0" \
  LAMBDA_GMVC_RANGE="0.0" \
  LAMBDA_GMVC_BINF="0.0" \
  LAMBDA_GMVC_INTRINSIC="0.0" \
  LAMBDA_GMVC_RESIDUAL_BUDGET="${LAMBDA_GMVC_RESIDUAL_BUDGET:-0.0}" \
  LAMBDA_GMVC_FIXED_CLOSURE="0.0" \
  GMVC_V2_ENABLED="${GMVC_V2_ENABLED:-${GMVC_V2_ENABLED_VALUE}}" \
  LAMBDA_GMVC_PROFILE="${LAMBDA_PROFILE}" \
  LAMBDA_GMVC_SYMMETRIC_CLOSURE="${LAMBDA_SYMMETRIC_CLOSURE}" \
  GMVC_PROFILE_DETACH_J_STAR="${GMVC_PROFILE_DETACH_J_STAR:-True}" \
  GMVC_V2_MAX_TRACKS_PER_STEP="${GMVC_V2_MAX_TRACKS_PER_STEP:-512}" \
  GMVC_V2_MIN_OBSERVATIONS_PER_TRACK="${GMVC_V2_MIN_OBSERVATIONS_PER_TRACK:-2}" \
  GMVC_BOUNDED_MEDIUM_ENABLED="${GMVC_BOUNDED_MEDIUM_ENABLED:-${BOUNDED_ENABLED}}" \
  GMVC_BOUNDED_MEDIUM_START_STEP="${GMVC_BOUNDED_MEDIUM_START_STEP:-10000}" \
  GMVC_BOUNDED_MEDIUM_PROJECTION_STEPS="${GMVC_BOUNDED_MEDIUM_PROJECTION_STEPS:-500}" \
  GMVC_BOUNDED_BETA_LOG_SCALE="${GMVC_BOUNDED_BETA_LOG_SCALE:-0.15}" \
  GMVC_BOUNDED_BINF_LOGIT_SCALE="${GMVC_BOUNDED_BINF_LOGIT_SCALE:-0.10}" \
  GMVC_BOUNDED_INIT_FROM_FIRST_BATCH="${GMVC_BOUNDED_INIT_FROM_FIRST_BATCH:-True}" \
  GMVC_CLOSURE_SIGNAL_FLOOR="${GMVC_CLOSURE_SIGNAL_FLOOR:-0.03}" \
  GMVC_MAX_TRACKS_PER_STEP="${GMVC_MAX_TRACKS_PER_STEP:-4096}" \
  GMVC_GRAD_LOG_PATH="${GMVC_GRAD_LOG_PATH:-${GRAD_LOG_DEFAULT}}" \
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-50}" \
  GMVC_FREEZE_GEOMETRY="False" \
  GMVC_TRAIN_FEATURES_DC="False" \
  GMVC_TRAIN_FEATURES_REST="False" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}" \
  "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
