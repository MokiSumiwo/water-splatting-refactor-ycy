#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-9}" \
  PYTHON="/opt/anaconda3/envs/water_splatting/bin/python" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}" \
  OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}" \
  RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}" \
  LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-tacmd_t4_cf015_japanesegradens_seed42_15000}" \
  STAMP="${STAMP:-20260731_tacmd}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-15000}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}" \
  SEED="${SEED:-42}" \
  MEDIUM_CONTEXT_MODE="dir_xy_camera" \
  BINF_MODE="tied" \
  TACMD_ENABLED="True" \
  LAMBDA_TACMD_TAIL_MEAN="${LAMBDA_TACMD_TAIL_MEAN:-0.001}" \
  LAMBDA_TACMD_TAIL_BAND="${LAMBDA_TACMD_TAIL_BAND:-0.0002}" \
  LAMBDA_TACMD_BS_BAND="${LAMBDA_TACMD_BS_BAND:-0.00002}" \
  LAMBDA_TACMD_BS_MONOTONIC="${LAMBDA_TACMD_BS_MONOTONIC:-0.00002}" \
  LAMBDA_TACMD_BS_TERMINAL="${LAMBDA_TACMD_BS_TERMINAL:-0.00001}" \
  LAMBDA_TACMD_CF_CHROMA="${LAMBDA_TACMD_CF_CHROMA:-0.025}" \
  TACMD_CF_PROJECTION_MAX="${TACMD_CF_PROJECTION_MAX:-0.15}" \
  TACMD_CF_RENDER_EVERY="${TACMD_CF_RENDER_EVERY:-4}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-1}" \
  RUN_FAR_DIAG="${RUN_FAR_DIAG:-1}" \
  RUN_REGION_DIAG="${RUN_REGION_DIAG:-1}" \
  COMMON_FAR_MASK_DIR="${COMMON_FAR_MASK_DIR:-${REPO_DIR}/common_masks/cross_scene_japanesegradens_redsea_m1_q90_seed42_20260730_cross_scene}" \
  REGION_MASK_DIR="${REGION_MASK_DIR:-${REPO_DIR}/common_masks/cross_scene_japanesegradens_redsea_m1_eval_regions_seed42_20260730_cross_scene}" \
  REFERENCE_CONFIG="${REFERENCE_CONFIG:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml}" \
  "${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"

