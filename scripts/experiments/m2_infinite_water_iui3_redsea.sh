#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
NS_TRAIN="/opt/anaconda3/envs/water_splatting/bin/ns-train"
NS_EVAL="/opt/anaconda3/envs/water_splatting/bin/ns-eval"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
RUN_EVAL="${RUN_EVAL:-1}"
SEED="${SEED:-42}"
MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
OWNERSHIP_MODE="${OWNERSHIP_MODE:-alpha_depth}"
BINF_RGB_WEIGHT="${BINF_RGB_WEIGHT:-${BINFGT_WEIGHT:-0.01}}"
ACCUM_ZERO_WEIGHT="${ACCUM_ZERO_WEIGHT:-0.004}"
NEAR_ZERO_WEIGHT="${NEAR_ZERO_WEIGHT:-0.001}"
INFINITE_WATER_OCCUPANCY_LIMITED="${INFINITE_WATER_OCCUPANCY_LIMITED:-True}"
INFINITE_WATER_COMPOSE_MODE="${INFINITE_WATER_COMPOSE_MODE:-rgb_mix}"
INFINITE_WATER_ALPHA_POWER="${INFINITE_WATER_ALPHA_POWER:-1.0}"
INFINITE_WATER_DEPTH_MID="${INFINITE_WATER_DEPTH_MID:-0.75}"
INFINITE_WATER_DEPTH_TEMP="${INFINITE_WATER_DEPTH_TEMP:-0.10}"
INFINITE_WATER_COLOR_TEMP="${INFINITE_WATER_COLOR_TEMP:-0.20}"
INFINITE_WATER_DEPTH_NORMALIZE_MODE="${INFINITE_WATER_DEPTH_NORMALIZE_MODE:-p95}"
INFINITE_WATER_HIT_ALPHA_THRESHOLD="${INFINITE_WATER_HIT_ALPHA_THRESHOLD:-0.20}"
INFINITE_WATER_HIT_ALPHA_TEMP="${INFINITE_WATER_HIT_ALPHA_TEMP:-0.05}"
INFINITE_WATER_HIT_CONCENTRATION_KAPPA="${INFINITE_WATER_HIT_CONCENTRATION_KAPPA:-0.20}"
INFINITE_WATER_CAPACITY_SUPPORT_MODE="${INFINITE_WATER_CAPACITY_SUPPORT_MODE:-m_inf}"
LOSS_START_STEP="${LOSS_START_STEP:-1000}"
LOSS_RAMP_STEPS="${LOSS_RAMP_STEPS:-3000}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-m2_${OWNERSHIP_MODE}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}}"
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
  echo "seed=${SEED}"
  echo "medium_context_mode=${MEDIUM_CONTEXT_MODE}"
  echo "ownership_mode=${OWNERSHIP_MODE}"
  echo "infinite_water_occupancy_limited=${INFINITE_WATER_OCCUPANCY_LIMITED}"
  echo "infinite_water_compose_mode=${INFINITE_WATER_COMPOSE_MODE}"
  echo "infinite_water_alpha_power=${INFINITE_WATER_ALPHA_POWER}"
  echo "infinite_water_depth_mid=${INFINITE_WATER_DEPTH_MID}"
  echo "infinite_water_depth_temp=${INFINITE_WATER_DEPTH_TEMP}"
  echo "infinite_water_color_temp=${INFINITE_WATER_COLOR_TEMP}"
  echo "infinite_water_depth_normalize_mode=${INFINITE_WATER_DEPTH_NORMALIZE_MODE}"
  echo "infinite_water_hit_alpha_threshold=${INFINITE_WATER_HIT_ALPHA_THRESHOLD}"
  echo "infinite_water_hit_alpha_temp=${INFINITE_WATER_HIT_ALPHA_TEMP}"
  echo "infinite_water_hit_concentration_kappa=${INFINITE_WATER_HIT_CONCENTRATION_KAPPA}"
  echo "infinite_water_capacity_support_mode=${INFINITE_WATER_CAPACITY_SUPPORT_MODE}"
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo "binf_rgb_weight=${BINF_RGB_WEIGHT}"
  echo "accumulation_zero_weight=${ACCUM_ZERO_WEIGHT}"
  echo "near_zero_weight=${NEAR_ZERO_WEIGHT}"
  echo "loss_start_step=${LOSS_START_STEP}"
  echo "loss_ramp_steps=${LOSS_RAMP_STEPS}"
  echo -n "git_commit="
  git -C "${REPO_DIR}" rev-parse HEAD || true
  echo "pytorch=$("${REPO_DIR}/.venv/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || /opt/anaconda3/envs/water_splatting/bin/python -c 'import torch; print(torch.__version__)')"
  echo "cuda=$(/opt/anaconda3/envs/water_splatting/bin/python -c 'import torch; print(torch.version.cuda)')"
  git -C "${REPO_DIR}" status --short || true
} | tee "${LOG_DIR}/run_manifest.txt"

CUDA_VISIBLE_DEVICES="${GPU}" "${NS_TRAIN}" water-splatting \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${TIMESTAMP}" \
  --vis tensorboard \
  --machine.seed "${SEED}" \
  --max-num-iterations "${MAX_NUM_ITERATIONS}" \
  --pipeline.model.num-steps "${MAX_NUM_ITERATIONS}" \
  --pipeline.model.medium-context-mode "${MEDIUM_CONTEXT_MODE}" \
  --pipeline.model.infinite-water-enabled True \
  --pipeline.model.infinite-water-ownership-mode "${OWNERSHIP_MODE}" \
  --pipeline.model.infinite-water-detach-evidence True \
  --pipeline.model.infinite-water-occupancy-limited "${INFINITE_WATER_OCCUPANCY_LIMITED}" \
  --pipeline.model.infinite-water-compose-mode "${INFINITE_WATER_COMPOSE_MODE}" \
  --pipeline.model.infinite-water-alpha-power "${INFINITE_WATER_ALPHA_POWER}" \
  --pipeline.model.infinite-water-depth-mid "${INFINITE_WATER_DEPTH_MID}" \
  --pipeline.model.infinite-water-depth-temp "${INFINITE_WATER_DEPTH_TEMP}" \
  --pipeline.model.infinite-water-color-temp "${INFINITE_WATER_COLOR_TEMP}" \
  --pipeline.model.infinite-water-depth-normalize-mode "${INFINITE_WATER_DEPTH_NORMALIZE_MODE}" \
  --pipeline.model.infinite-water-hit-alpha-threshold "${INFINITE_WATER_HIT_ALPHA_THRESHOLD}" \
  --pipeline.model.infinite-water-hit-alpha-temp "${INFINITE_WATER_HIT_ALPHA_TEMP}" \
  --pipeline.model.infinite-water-hit-concentration-kappa "${INFINITE_WATER_HIT_CONCENTRATION_KAPPA}" \
  --pipeline.model.infinite-water-capacity-support-mode "${INFINITE_WATER_CAPACITY_SUPPORT_MODE}" \
  --pipeline.model.infinite-water-loss-start-step "${LOSS_START_STEP}" \
  --pipeline.model.infinite-water-loss-ramp-steps "${LOSS_RAMP_STEPS}" \
  --pipeline.model.lambda-infinite-water-binf-rgb "${BINF_RGB_WEIGHT}" \
  --pipeline.model.lambda-infinite-water-accumulation-zero "${ACCUM_ZERO_WEIGHT}" \
  --pipeline.model.lambda-infinite-water-near-zero "${NEAR_ZERO_WEIGHT}" \
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
