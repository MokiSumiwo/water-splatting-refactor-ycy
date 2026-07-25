#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

export GPU="${GPU:-6}"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
export RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
export LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
export RUN_EVAL="${RUN_EVAL:-1}"
export SEED="${SEED:-42}"
export MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
export OWNERSHIP_MODE="${OWNERSHIP_MODE:-alpha_depth}"
export INFINITE_WATER_OCCUPANCY_LIMITED="${INFINITE_WATER_OCCUPANCY_LIMITED:-True}"
export INFINITE_WATER_DEPTH_MID="${INFINITE_WATER_DEPTH_MID:-0.75}"
export INFINITE_WATER_DEPTH_TEMP="${INFINITE_WATER_DEPTH_TEMP:-0.10}"
export INFINITE_WATER_CAPACITY_SUPPORT_MODE="${INFINITE_WATER_CAPACITY_SUPPORT_MODE:-m_inf}"
export INFINITE_WATER_HIT_PROTECTION_ENABLED="${INFINITE_WATER_HIT_PROTECTION_ENABLED:-False}"
export NEAR_ZERO_WEIGHT="${NEAR_ZERO_WEIGHT:-0}"
export LOSS_START_STEP="${LOSS_START_STEP:-1000}"
export LOSS_RAMP_STEPS="${LOSS_RAMP_STEPS:-3000}"

MECHANISM_GRID="${MECHANISM_GRID:-S0:none:0:0:none S1:rgb_mix:0.005:0:none S2:none:0:0.002:current S3:rgb_mix:0.005:0.002:current S4:none:0.005:0.002:current}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export STAMP

for ITEM in ${MECHANISM_GRID}; do
  TAG="${ITEM%%:*}"
  REST="${ITEM#*:}"
  export INFINITE_WATER_COMPOSE_MODE="${REST%%:*}"
  REST="${REST#*:}"
  export BINF_RGB_WEIGHT="${REST%%:*}"
  REST="${REST#*:}"
  export ACCUM_ZERO_WEIGHT="${REST%%:*}"
  export INFINITE_WATER_CAPACITY_LOSS_MODE="${REST#*:}"
  export EXPERIMENT_NAME="${EXPERIMENT_NAME_PREFIX:-m2_p3_mech}_${TAG}_compose${INFINITE_WATER_COMPOSE_MODE}_binf${BINF_RGB_WEIGHT/./p}_cap${ACCUM_ZERO_WEIGHT/./p}_${INFINITE_WATER_CAPACITY_LOSS_MODE}_seed${SEED}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}"
  unset TIMESTAMP
  bash "${REPO_DIR}/scripts/experiments/m2_infinite_water_iui3_redsea.sh"
done
