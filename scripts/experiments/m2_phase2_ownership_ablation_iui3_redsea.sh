#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

export GPU="${GPU:-9}"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
export RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
export LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
export RUN_EVAL="${RUN_EVAL:-1}"
export SEED="${SEED:-42}"
export MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
export INFINITE_WATER_OCCUPANCY_LIMITED="${INFINITE_WATER_OCCUPANCY_LIMITED:-True}"
export INFINITE_WATER_COMPOSE_MODE="${INFINITE_WATER_COMPOSE_MODE:-rgb_mix}"
export BINF_RGB_WEIGHT="${BINF_RGB_WEIGHT:-0.005}"
export ACCUM_ZERO_WEIGHT="${ACCUM_ZERO_WEIGHT:-0.002}"
export NEAR_ZERO_WEIGHT="${NEAR_ZERO_WEIGHT:-0}"
export LOSS_START_STEP="${LOSS_START_STEP:-1000}"
export LOSS_RAMP_STEPS="${LOSS_RAMP_STEPS:-3000}"

OWNERSHIP_MODES="${OWNERSHIP_MODES:-alpha_only alpha_depth alpha_depth_color}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export STAMP

for OWNERSHIP_MODE_VALUE in ${OWNERSHIP_MODES}; do
  export OWNERSHIP_MODE="${OWNERSHIP_MODE_VALUE}"
  export EXPERIMENT_NAME="${EXPERIMENT_NAME_PREFIX:-m2_p2_ownership}_${OWNERSHIP_MODE}_accum${ACCUM_ZERO_WEIGHT/./p}_seed${SEED}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}"
  unset TIMESTAMP
  bash "${REPO_DIR}/scripts/experiments/m2_infinite_water_iui3_redsea.sh"
done
