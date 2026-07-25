#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
MASK_DIR="${MASK_DIR:-${REPO_DIR}/common_masks/pseudo_depth_bg_iui3_redsea_20260725}"

exec env \
  GPU="${GPU:-9}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-binf_a3_bounded_residual_s002_bg005_dir_xy_camera_iui3_redsea_15000}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}" \
  BINF_MODE="bounded_residual" \
  BINF_RESIDUAL_SCALE="0.02" \
  BG_COLOR_WEIGHT="0.005" \
  FG_TRANS_WEIGHT="0.0" \
  BACKSCATTER_REGION_MASK_DIR="${MASK_DIR}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-1}" \
  RUN_FAR_DIAG="${RUN_FAR_DIAG:-1}" \
  RUN_REGION_DIAG="${RUN_REGION_DIAG:-1}" \
  "${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"
