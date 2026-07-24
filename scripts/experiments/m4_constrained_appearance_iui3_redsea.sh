#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-9}"
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

APPEARANCE_SH_DELAY_ENABLED="${APPEARANCE_SH_DELAY_ENABLED:-True}"
APPEARANCE_SH_DELAY_START_STEP="${APPEARANCE_SH_DELAY_START_STEP:-3000}"
APPEARANCE_SH_DELAY_INTERVAL="${APPEARANCE_SH_DELAY_INTERVAL:-2000}"
APPEARANCE_LOSS_START_STEP="${APPEARANCE_LOSS_START_STEP:-1000}"
APPEARANCE_LOSS_RAMP_STEPS="${APPEARANCE_LOSS_RAMP_STEPS:-3000}"
SH_RESIDUAL_WEIGHT="${SH_RESIDUAL_WEIGHT:-0.002}"
DC_SOFTCLIP_WEIGHT="${DC_SOFTCLIP_WEIGHT:-0.001}"
DC_SOFTCLIP_THRESHOLD="${DC_SOFTCLIP_THRESHOLD:-0.95}"
DC_SOFTCLIP_BETA="${DC_SOFTCLIP_BETA:-0.05}"
DC_SOFTCLIP_USE_LOW_TRANS_WEIGHT="${DC_SOFTCLIP_USE_LOW_TRANS_WEIGHT:-True}"
DC_CHANNEL_BALANCE_WEIGHT="${DC_CHANNEL_BALANCE_WEIGHT:-0.0}"
DC_CHANNEL_BALANCE_MARGIN="${DC_CHANNEL_BALANCE_MARGIN:-0.05}"
DC_CHANNEL_BALANCE_BETA="${DC_CHANNEL_BALANCE_BETA:-0.05}"
DC_CHANNEL_BALANCE_USE_LOW_TRANS_WEIGHT="${DC_CHANNEL_BALANCE_USE_LOW_TRANS_WEIGHT:-True}"
MEDIUM_ATTENUATION_ORDER_WEIGHT="${MEDIUM_ATTENUATION_ORDER_WEIGHT:-0.0}"
MEDIUM_ATTENUATION_ORDER_MARGIN="${MEDIUM_ATTENUATION_ORDER_MARGIN:-0.0}"
MEDIUM_ATTENUATION_ORDER_BETA="${MEDIUM_ATTENUATION_ORDER_BETA:-0.05}"
MEDIUM_ATTENUATION_ORDER_USE_LOW_TRANS_WEIGHT="${MEDIUM_ATTENUATION_ORDER_USE_LOW_TRANS_WEIGHT:-True}"
LOW_TRANS_THRESHOLD="${LOW_TRANS_THRESHOLD:-0.35}"
LOW_TRANS_TEMPERATURE="${LOW_TRANS_TEMPERATURE:-0.10}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-m4_constrained_appearance_${OWNERSHIP_MODE}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}}"
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
  echo "appearance_sh_delay_enabled=${APPEARANCE_SH_DELAY_ENABLED}"
  echo "appearance_sh_delay_start_step=${APPEARANCE_SH_DELAY_START_STEP}"
  echo "appearance_sh_delay_interval=${APPEARANCE_SH_DELAY_INTERVAL}"
  echo "appearance_loss_start_step=${APPEARANCE_LOSS_START_STEP}"
  echo "appearance_loss_ramp_steps=${APPEARANCE_LOSS_RAMP_STEPS}"
  echo "sh_residual_weight=${SH_RESIDUAL_WEIGHT}"
  echo "dc_softclip_weight=${DC_SOFTCLIP_WEIGHT}"
  echo "dc_softclip_threshold=${DC_SOFTCLIP_THRESHOLD}"
  echo "dc_softclip_beta=${DC_SOFTCLIP_BETA}"
  echo "dc_softclip_use_low_transmission_weight=${DC_SOFTCLIP_USE_LOW_TRANS_WEIGHT}"
  echo "dc_channel_balance_weight=${DC_CHANNEL_BALANCE_WEIGHT}"
  echo "dc_channel_balance_margin=${DC_CHANNEL_BALANCE_MARGIN}"
  echo "dc_channel_balance_beta=${DC_CHANNEL_BALANCE_BETA}"
  echo "dc_channel_balance_use_low_transmission_weight=${DC_CHANNEL_BALANCE_USE_LOW_TRANS_WEIGHT}"
  echo "medium_attenuation_order_weight=${MEDIUM_ATTENUATION_ORDER_WEIGHT}"
  echo "medium_attenuation_order_margin=${MEDIUM_ATTENUATION_ORDER_MARGIN}"
  echo "medium_attenuation_order_beta=${MEDIUM_ATTENUATION_ORDER_BETA}"
  echo "medium_attenuation_order_use_low_transmission_weight=${MEDIUM_ATTENUATION_ORDER_USE_LOW_TRANS_WEIGHT}"
  echo "low_transmission_threshold=${LOW_TRANS_THRESHOLD}"
  echo "low_transmission_temperature=${LOW_TRANS_TEMPERATURE}"
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
  --pipeline.model.constrained-appearance-enabled True \
  --pipeline.model.appearance-sh-delay-enabled "${APPEARANCE_SH_DELAY_ENABLED}" \
  --pipeline.model.appearance-sh-delay-start-step "${APPEARANCE_SH_DELAY_START_STEP}" \
  --pipeline.model.appearance-sh-delay-interval "${APPEARANCE_SH_DELAY_INTERVAL}" \
  --pipeline.model.appearance-loss-start-step "${APPEARANCE_LOSS_START_STEP}" \
  --pipeline.model.appearance-loss-ramp-steps "${APPEARANCE_LOSS_RAMP_STEPS}" \
  --pipeline.model.lambda-sh-residual-mean "${SH_RESIDUAL_WEIGHT}" \
  --pipeline.model.lambda-dc-softclip "${DC_SOFTCLIP_WEIGHT}" \
  --pipeline.model.dc-softclip-threshold "${DC_SOFTCLIP_THRESHOLD}" \
  --pipeline.model.dc-softclip-beta "${DC_SOFTCLIP_BETA}" \
  --pipeline.model.dc-softclip-use-low-transmission-weight "${DC_SOFTCLIP_USE_LOW_TRANS_WEIGHT}" \
  --pipeline.model.lambda-dc-channel-balance "${DC_CHANNEL_BALANCE_WEIGHT}" \
  --pipeline.model.dc-channel-balance-margin "${DC_CHANNEL_BALANCE_MARGIN}" \
  --pipeline.model.dc-channel-balance-beta "${DC_CHANNEL_BALANCE_BETA}" \
  --pipeline.model.dc-channel-balance-use-low-transmission-weight "${DC_CHANNEL_BALANCE_USE_LOW_TRANS_WEIGHT}" \
  --pipeline.model.lambda-medium-attenuation-order "${MEDIUM_ATTENUATION_ORDER_WEIGHT}" \
  --pipeline.model.medium-attenuation-order-margin "${MEDIUM_ATTENUATION_ORDER_MARGIN}" \
  --pipeline.model.medium-attenuation-order-beta "${MEDIUM_ATTENUATION_ORDER_BETA}" \
  --pipeline.model.medium-attenuation-order-use-low-transmission-weight "${MEDIUM_ATTENUATION_ORDER_USE_LOW_TRANS_WEIGHT}" \
  --pipeline.model.low-transmission-threshold "${LOW_TRANS_THRESHOLD}" \
  --pipeline.model.low-transmission-temperature "${LOW_TRANS_TEMPERATURE}" \
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
