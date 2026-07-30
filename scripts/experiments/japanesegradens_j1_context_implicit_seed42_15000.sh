#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-6}" \
  PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}" \
  OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}" \
  RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}" \
  LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-japanesegradens_j1_context_implicit_seed42_15000}" \
  STAMP="${STAMP:-20260730_jgradens_factor_split}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-15000}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}" \
  SEED="${SEED:-42}" \
  MEDIUM_CONTEXT_MODE="dir_xy_camera" \
  BINF_MODE="implicit" \
  BG_COLOR_WEIGHT="0.0" \
  BG_MEDIUM_RENDER_WEIGHT="0.0" \
  BG_TAIL_RENDER_WEIGHT="0.0" \
  BG_CLEAR_GAUSSIAN_WEIGHT="0.0" \
  BG_CLEAR_CHROMA_WEIGHT="0.0" \
  MEDIUM_EXPLAINABILITY_ENABLED="False" \
  LAMBDA_MEDIUM_EXPLAINABILITY="0.0" \
  TRAINING_GRADIENT_ROUTING_ENABLED="False" \
  BUDGETED_CAPACITY_ENABLED="False" \
  LAMBDA_BUDGETED_CAPACITY="0.0" \
  HALO_CAPACITY_ENABLED="False" \
  LAMBDA_HALO_CAPACITY="0.0" \
  LAMBDA_PROXY_CLEAR_LUMA="0.0" \
  CLEAR_PROXY_ENABLED="False" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}" \
  RUN_FAR_DIAG="${RUN_FAR_DIAG:-0}" \
  RUN_REGION_DIAG="${RUN_REGION_DIAG:-0}" \
  "${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"

