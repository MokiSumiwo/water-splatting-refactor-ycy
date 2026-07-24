#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

export GPU="${GPU:-8}"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
export RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
export LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
export RUN_EVAL="${RUN_EVAL:-1}"
export MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
export OWNERSHIP_MODE="${OWNERSHIP_MODE:-alpha_depth}"
export INFINITE_WATER_OCCUPANCY_LIMITED="${INFINITE_WATER_OCCUPANCY_LIMITED:-True}"
export INFINITE_WATER_COMPOSE_MODE="${INFINITE_WATER_COMPOSE_MODE:-rgb_mix}"
export BINF_RGB_WEIGHT="${BINF_RGB_WEIGHT:-0.005}"
export NEAR_ZERO_WEIGHT="${NEAR_ZERO_WEIGHT:-0}"
export LOSS_START_STEP="${LOSS_START_STEP:-1000}"
export LOSS_RAMP_STEPS="${LOSS_RAMP_STEPS:-3000}"

ACCUM_ZERO_WEIGHTS="${ACCUM_ZERO_WEIGHTS:-0.0005 0.0010 0.0015 0.0020 0.0025}"
SEEDS="${SEEDS:-42}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export STAMP

for SEED_VALUE in ${SEEDS}; do
  export SEED="${SEED_VALUE}"
  for ACCUM_ZERO_WEIGHT_VALUE in ${ACCUM_ZERO_WEIGHTS}; do
    export ACCUM_ZERO_WEIGHT="${ACCUM_ZERO_WEIGHT_VALUE}"
    ACCUM_TAG="${ACCUM_ZERO_WEIGHT_VALUE/./p}"
    export EXPERIMENT_NAME="${EXPERIMENT_NAME_PREFIX:-m2_p1}_accum${ACCUM_TAG}_seed${SEED}_${OWNERSHIP_MODE}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}"
    unset TIMESTAMP
    bash "${REPO_DIR}/scripts/experiments/m2_infinite_water_iui3_redsea.sh"
  done
done
