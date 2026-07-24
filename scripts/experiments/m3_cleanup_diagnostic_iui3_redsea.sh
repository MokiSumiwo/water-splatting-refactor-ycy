#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-8}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
NS_TRAIN="/opt/anaconda3/envs/water_splatting/bin/ns-train"
NS_EVAL="/opt/anaconda3/envs/water_splatting/bin/ns-eval"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
RUN_EVAL="${RUN_EVAL:-1}"
MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
OWNERSHIP_MODE="${OWNERSHIP_MODE:-alpha_depth}"
BINFGT_WEIGHT="${BINFGT_WEIGHT:-0.01}"
ACCUM_ZERO_WEIGHT="${ACCUM_ZERO_WEIGHT:-0.004}"
NEAR_ZERO_WEIGHT="${NEAR_ZERO_WEIGHT:-0.001}"

CLEANUP_DRY_RUN="${CLEANUP_DRY_RUN:-True}"
CLEANUP_START_STEP="${CLEANUP_START_STEP:-12000}"
CLEANUP_INTERVAL="${CLEANUP_INTERVAL:-500}"
CLEANUP_CONTRIBUTION_THRESHOLD="${CLEANUP_CONTRIBUTION_THRESHOLD:-0.0001}"
CLEANUP_OPACITY_THRESHOLD="${CLEANUP_OPACITY_THRESHOLD:-0.08}"
CLEANUP_VISIBILITY_MIN_COUNT="${CLEANUP_VISIBILITY_MIN_COUNT:-2}"
CLEANUP_ALPHA_THRESHOLD="${CLEANUP_ALPHA_THRESHOLD:-0.25}"
CLEANUP_DEPTH_THRESHOLD="${CLEANUP_DEPTH_THRESHOLD:-0.0}"
CLEANUP_OWNERSHIP_THRESHOLD="${CLEANUP_OWNERSHIP_THRESHOLD:-0.35}"
CLEANUP_OWNERSHIP_SOURCE="${CLEANUP_OWNERSHIP_SOURCE:-m_inf_eff}"
CLEANUP_REQUIRE_ALPHA_GATE="${CLEANUP_REQUIRE_ALPHA_GATE:-True}"
CLEANUP_REQUIRE_DEPTH_GATE="${CLEANUP_REQUIRE_DEPTH_GATE:-False}"
CLEANUP_REQUIRE_OWNERSHIP_GATE="${CLEANUP_REQUIRE_OWNERSHIP_GATE:-True}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-m3_cleanup_diag_${OWNERSHIP_MODE}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-${EXPERIMENT_NAME}_${STAMP}}"
LOG_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}_${STAMP}"
TRAIN_LOG="${LOG_DIR}/train.log"
EVAL_LOG="${LOG_DIR}/eval.log"
CONFIG_PATH="${OUTPUT_DIR}/${EXPERIMENT_NAME}/water-splatting/${TIMESTAMP}/config.yml"
RENDER_DIR="${RENDER_ROOT}/${EXPERIMENT_NAME}_${STAMP}"

mkdir -p "${LOG_DIR}" "${RENDER_DIR}"

{
  echo "experiment=${EXPERIMENT_NAME}"
  echo "timestamp=${TIMESTAMP}"
  echo "gpu=${GPU}"
  echo "data=${DATA_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "render_dir=${RENDER_DIR}"
  echo "medium_context_mode=${MEDIUM_CONTEXT_MODE}"
  echo "ownership_mode=${OWNERSHIP_MODE}"
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo "binf_rgb_weight=${BINFGT_WEIGHT}"
  echo "accumulation_zero_weight=${ACCUM_ZERO_WEIGHT}"
  echo "near_zero_weight=${NEAR_ZERO_WEIGHT}"
  echo "cleanup_dry_run=${CLEANUP_DRY_RUN}"
  echo "cleanup_start_step=${CLEANUP_START_STEP}"
  echo "cleanup_interval=${CLEANUP_INTERVAL}"
  echo "cleanup_contribution_threshold=${CLEANUP_CONTRIBUTION_THRESHOLD}"
  echo "cleanup_opacity_threshold=${CLEANUP_OPACITY_THRESHOLD}"
  echo "cleanup_visibility_min_count=${CLEANUP_VISIBILITY_MIN_COUNT}"
  echo "cleanup_alpha_threshold=${CLEANUP_ALPHA_THRESHOLD}"
  echo "cleanup_depth_threshold=${CLEANUP_DEPTH_THRESHOLD}"
  echo "cleanup_ownership_threshold=${CLEANUP_OWNERSHIP_THRESHOLD}"
  echo "cleanup_ownership_source=${CLEANUP_OWNERSHIP_SOURCE}"
  echo "cleanup_require_alpha_gate=${CLEANUP_REQUIRE_ALPHA_GATE}"
  echo "cleanup_require_depth_gate=${CLEANUP_REQUIRE_DEPTH_GATE}"
  echo "cleanup_require_ownership_gate=${CLEANUP_REQUIRE_OWNERSHIP_GATE}"
  git -C "${REPO_DIR}" rev-parse HEAD || true
  git -C "${REPO_DIR}" status --short || true
} | tee "${LOG_DIR}/run_manifest.txt"

CUDA_VISIBLE_DEVICES="${GPU}" "${NS_TRAIN}" water-splatting \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${TIMESTAMP}" \
  --vis tensorboard \
  --max-num-iterations "${MAX_NUM_ITERATIONS}" \
  --pipeline.model.num-steps "${MAX_NUM_ITERATIONS}" \
  --pipeline.model.medium-context-mode "${MEDIUM_CONTEXT_MODE}" \
  --pipeline.model.infinite-water-enabled True \
  --pipeline.model.infinite-water-ownership-mode "${OWNERSHIP_MODE}" \
  --pipeline.model.infinite-water-detach-evidence True \
  --pipeline.model.infinite-water-occupancy-limited True \
  --pipeline.model.infinite-water-loss-start-step 1000 \
  --pipeline.model.infinite-water-loss-ramp-steps 3000 \
  --pipeline.model.lambda-infinite-water-binf-rgb "${BINFGT_WEIGHT}" \
  --pipeline.model.lambda-infinite-water-accumulation-zero "${ACCUM_ZERO_WEIGHT}" \
  --pipeline.model.lambda-infinite-water-near-zero "${NEAR_ZERO_WEIGHT}" \
  --pipeline.model.gaussian-cleanup-enabled True \
  --pipeline.model.gaussian-cleanup-dry-run "${CLEANUP_DRY_RUN}" \
  --pipeline.model.gaussian-cleanup-start-step "${CLEANUP_START_STEP}" \
  --pipeline.model.gaussian-cleanup-interval "${CLEANUP_INTERVAL}" \
  --pipeline.model.gaussian-cleanup-contribution-threshold "${CLEANUP_CONTRIBUTION_THRESHOLD}" \
  --pipeline.model.gaussian-cleanup-opacity-threshold "${CLEANUP_OPACITY_THRESHOLD}" \
  --pipeline.model.gaussian-cleanup-visibility-min-count "${CLEANUP_VISIBILITY_MIN_COUNT}" \
  --pipeline.model.gaussian-cleanup-alpha-threshold "${CLEANUP_ALPHA_THRESHOLD}" \
  --pipeline.model.gaussian-cleanup-depth-threshold "${CLEANUP_DEPTH_THRESHOLD}" \
  --pipeline.model.gaussian-cleanup-ownership-threshold "${CLEANUP_OWNERSHIP_THRESHOLD}" \
  --pipeline.model.gaussian-cleanup-ownership-source "${CLEANUP_OWNERSHIP_SOURCE}" \
  --pipeline.model.gaussian-cleanup-require-alpha-gate "${CLEANUP_REQUIRE_ALPHA_GATE}" \
  --pipeline.model.gaussian-cleanup-require-depth-gate "${CLEANUP_REQUIRE_DEPTH_GATE}" \
  --pipeline.model.gaussian-cleanup-require-ownership-gate "${CLEANUP_REQUIRE_OWNERSHIP_GATE}" \
  colmap \
  --data "${DATA_PATH}" \
  --images-path images/ColorImage \
  --colmap-path sparse/0 \
  --downscale-factor 1 \
  2>&1 | tee "${TRAIN_LOG}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  pushd "${RENDER_DIR}" >/dev/null
  CUDA_VISIBLE_DEVICES="${GPU}" "${NS_EVAL}" \
    --load-config "${CONFIG_PATH}" \
    --render-output-path "${RENDER_DIR}" \
    2>&1 | tee "${EVAL_LOG}"
  popd >/dev/null
fi
