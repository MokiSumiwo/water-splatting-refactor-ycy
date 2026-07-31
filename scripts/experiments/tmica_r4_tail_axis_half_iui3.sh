#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-9}" \
  PYTHON="/opt/anaconda3/envs/water_splatting/bin/python" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}" \
  OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}" \
  RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}" \
  LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-tmica_r4_tail_axis_half_iui3_seed42_15000}" \
  STAMP="${STAMP:-20260731_tmica}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-15000}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}" \
  SEED="${SEED:-42}" \
  MEDIUM_CONTEXT_MODE="dir_xy_camera" \
  BINF_MODE="tied" \
  TACMD_ENABLED="False" \
  TMICA_ENABLED="True" \
  TMICA_USE_CLEAR_PROXY="True" \
  TMICA_AXIS_GRADIENT_PROJECTION="True" \
  LAMBDA_TMICA_TAIL_LITE="${LAMBDA_TMICA_TAIL_LITE:-0.00025}" \
  LAMBDA_TMICA_FAR_AXIS="${LAMBDA_TMICA_FAR_AXIS:-0.000125}" \
  LAMBDA_TMICA_DEPTH_TREND="${LAMBDA_TMICA_DEPTH_TREND:-0.00010}" \
  LAMBDA_TMICA_OVERCORRECTION="${LAMBDA_TMICA_OVERCORRECTION:-0.00010}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-1}" \
  RUN_FAR_DIAG="${RUN_FAR_DIAG:-1}" \
  RUN_REGION_DIAG="${RUN_REGION_DIAG:-1}" \
  COMMON_FAR_MASK_DIR="${COMMON_FAR_MASK_DIR:-${REPO_DIR}/common_masks/m1_q90_iui3_redsea_20260724}" \
  REGION_MASK_DIR="${REGION_MASK_DIR:-${REPO_DIR}/common_masks/m1_auto_eval_regions_iui3_redsea_20260724}" \
  REFERENCE_CONFIG="${REFERENCE_CONFIG:-${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml}" \
  "${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"
