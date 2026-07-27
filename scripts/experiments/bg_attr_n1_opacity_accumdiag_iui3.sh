#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
MASK_DIR="${MASK_DIR:-${REPO_DIR}/common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726}"

exec env \
  GPU="${GPU:-6}" \
  PYTHON="/opt/anaconda3/envs/water_splatting/bin/python" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}" \
  OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}" \
  RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}" \
  LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-bg_attr_n1_opacity_accumdiag_iui3_${MAX_NUM_ITERATIONS:-100}}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-100}" \
  SEED="${SEED:-42}" \
  MEDIUM_CONTEXT_MODE="dir_xy_camera" \
  BINF_MODE="tied" \
  BINF_RESIDUAL_SCALE="0.02" \
  BG_COLOR_WEIGHT="0.005" \
  BG_MEDIUM_RENDER_WEIGHT="0.0" \
  BG_TAIL_RENDER_WEIGHT="0.0" \
  BG_CLEAR_GAUSSIAN_WEIGHT="0.0" \
  BACKSCATTER_REGION_MASK_DIR="${MASK_DIR}" \
  BACKGROUND_WATER_MASK_KEY="water" \
  FOREGROUND_WATER_MASK_KEY="object" \
  BACKGROUND_DENSIFICATION_ENABLED="False" \
  BACKGROUND_DENSIFICATION_DIAGNOSTIC_ONLY="True" \
  OPACITY_ACCUMULATION_DIAGNOSTIC_ENABLED="True" \
  RUN_EVAL="${RUN_EVAL:-0}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}" \
  RUN_FAR_DIAG="${RUN_FAR_DIAG:-0}" \
  RUN_REGION_DIAG="${RUN_REGION_DIAG:-0}" \
  "${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"
