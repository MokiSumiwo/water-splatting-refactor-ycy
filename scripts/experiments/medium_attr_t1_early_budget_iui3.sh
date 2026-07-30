#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-6}" \
  PYTHON="/opt/anaconda3/envs/water_splatting/bin/python" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}" \
  OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}" \
  RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}" \
  LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_t1_early_budget_iui3_15000}" \
  STAMP="${STAMP:-20260730_t1_early_budget}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-15000}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}" \
  SEED="${SEED:-42}" \
  BUDGETED_CAPACITY_START_STEP="${BUDGETED_CAPACITY_START_STEP:-1000}" \
  BUDGETED_CAPACITY_RAMP_STEPS="${BUDGETED_CAPACITY_RAMP_STEPS:-3000}" \
  BUDGETED_CAPACITY_POST_SCALE="${BUDGETED_CAPACITY_POST_SCALE:-1.0}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-1}" \
  RUN_FAR_DIAG="${RUN_FAR_DIAG:-1}" \
  RUN_REGION_DIAG="${RUN_REGION_DIAG:-1}" \
  "${REPO_DIR}/scripts/experiments/medium_attr_p3_b02_proxy_geom000_opacity050_iui3.sh"
