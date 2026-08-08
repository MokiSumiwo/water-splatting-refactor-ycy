#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
BASE_SCRIPT="${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"

MODE="${MODE:-full}"
GPU="${GPU:-6}"
RUN_SET="${RUN_SET:-all}"
STAMP_BASE="${STAMP_BASE:-20260808_bounded_sh3_scratch}"

if [[ "${MODE}" == "smoke" ]]; then
  MAX_ITERS="${MAX_ITERS:-20}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/dewater_bounded_sh3_scratch_20260808/smoke}"
  RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders/dewater_bounded_sh3_scratch_20260808/smoke}"
  LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/dewater_bounded_sh3_scratch_20260808/smoke}"
  SAVE_ONLY_LATEST="${SAVE_ONLY_LATEST:-True}"
else
  MAX_ITERS="${MAX_ITERS:-15000}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/dewater_bounded_sh3_scratch_20260808}"
  RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders/dewater_bounded_sh3_scratch_20260808}"
  LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/dewater_bounded_sh3_scratch_20260808}"
  SAVE_ONLY_LATEST="${SAVE_ONLY_LATEST:-False}"
fi

run_one() {
  local run_name="$1"
  local gamma="$2"
  local gamma_tag="$3"
  local experiment_name="dewater_bounded_sh3_curasao_seed42_${run_name,,}_step0_to_${MAX_ITERS}"
  local stamp="${STAMP_BASE}_${MODE}_${run_name,,}_g${gamma_tag}"
  local timestamp="${experiment_name}_${stamp}"

  echo "[${run_name}] mode=${MODE} gamma_D=${gamma} max_iters=${MAX_ITERS}"
  GPU="${GPU}" \
  DATA_PATH="${REPO_DIR}/undistorted_data/undistorted_Curasao" \
  OUTPUT_DIR="${OUTPUT_ROOT}" \
  RENDER_ROOT="${RENDER_ROOT}" \
  LOG_ROOT="${LOG_ROOT}" \
  EXPERIMENT_NAME="${experiment_name}" \
  STAMP="${stamp}" \
  TIMESTAMP="${timestamp}" \
  MAX_NUM_ITERATIONS="${MAX_ITERS}" \
  MODEL_NUM_STEPS="${MAX_ITERS}" \
  STEPS_PER_SAVE=1000 \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST}" \
  RUN_EVAL=0 \
  RUN_CLOSURE_DIAG=0 \
  RUN_FAR_DIAG=0 \
  RUN_REGION_DIAG=0 \
  SEED=42 \
  MEDIUM_CONTEXT_MODE=dir_xy_camera \
  BINF_MODE=tied \
  DIRECT_OPTICAL_DEPTH_SCALE="${gamma}" \
  INTRINSIC_COLOR_PARAMETERIZATION=sigmoid_sh \
  BOUNDED_SH_LOGIT_EPS=1e-7 \
  INTRINSIC_BOUND_LAMBDA=0.0 \
  FOREGROUND_AWARE_WEIGHTING_ENABLED=False \
  FOREGROUND_AWARE_WEIGHTING_LAMBDA=0.0 \
  MEDIUM_BACKGROUND_SUPERVISION_ENABLED=False \
  MEDIUM_BACKGROUND_SUPERVISION_LAMBDA=0.0 \
  GMVC_ENABLED=False \
  GMVC_DIAGNOSTIC_ONLY=False \
  DISABLE_POPULATION_REFINEMENT=False \
  bash "${BASE_SCRIPT}"
}

case "${RUN_SET}" in
  all)
    run_one "BND-SCRATCH" "1.0" "1p00"
    run_one "D010-BND-SCRATCH" "0.1" "0p10"
    ;;
  BND-SCRATCH)
    run_one "BND-SCRATCH" "1.0" "1p00"
    ;;
  D010-BND-SCRATCH)
    run_one "D010-BND-SCRATCH" "0.1" "0p10"
    ;;
  *)
    echo "Unknown RUN_SET=${RUN_SET}; expected all, BND-SCRATCH, or D010-BND-SCRATCH" >&2
    exit 2
    ;;
esac
