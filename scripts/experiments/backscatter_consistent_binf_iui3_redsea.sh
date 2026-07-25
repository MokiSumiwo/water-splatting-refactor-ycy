#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="/opt/anaconda3/envs/water_splatting/bin/python"
NS_TRAIN="/opt/anaconda3/envs/water_splatting/bin/ns-train"
NS_EVAL="/opt/anaconda3/envs/water_splatting/bin/ns-eval"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"

MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-1}"
RUN_FAR_DIAG="${RUN_FAR_DIAG:-0}"
RUN_REGION_DIAG="${RUN_REGION_DIAG:-0}"
SEED="${SEED:-42}"
MEDIUM_CONTEXT_MODE="${MEDIUM_CONTEXT_MODE:-dir_xy_camera}"
BINF_MODE="${BINF_MODE:-tied}"
BINF_RESIDUAL_SCALE="${BINF_RESIDUAL_SCALE:-0.02}"
BG_COLOR_WEIGHT="${BG_COLOR_WEIGHT:-0.0}"
FG_TRANS_WEIGHT="${FG_TRANS_WEIGHT:-0.0}"
FG_TRANS_GAMMA="${FG_TRANS_GAMMA:-1.0}"
FG_TRANS_MAX_WEIGHT="${FG_TRANS_MAX_WEIGHT:-4.0}"
FG_TRANS_DETACH_WEIGHT="${FG_TRANS_DETACH_WEIGHT:-True}"
BACKSCATTER_REGION_MASK_DIR="${BACKSCATTER_REGION_MASK_DIR:-}"
BACKGROUND_WATER_MASK_KEY="${BACKGROUND_WATER_MASK_KEY:-water}"
FOREGROUND_WATER_MASK_KEY="${FOREGROUND_WATER_MASK_KEY:-object}"
BACKSCATTER_LOSS_START_STEP="${BACKSCATTER_LOSS_START_STEP:-0}"
BACKSCATTER_LOSS_RAMP_STEPS="${BACKSCATTER_LOSS_RAMP_STEPS:-0}"
MEDIUM_PREDICTOR_MODE="${MEDIUM_PREDICTOR_MODE:-single}"
LAMBDA_PSEUDO_DEPTH="${LAMBDA_PSEUDO_DEPTH:-0.0}"
LAMBDA_MEDIUM_CONTEXT_RESIDUAL="${LAMBDA_MEDIUM_CONTEXT_RESIDUAL:-0.0}"
COMMON_FAR_MASK_DIR="${COMMON_FAR_MASK_DIR:-${REPO_DIR}/common_masks/m1_q90_iui3_redsea_20260724}"
REGION_MASK_DIR="${REGION_MASK_DIR:-${REPO_DIR}/common_masks/m1_auto_eval_regions_iui3_redsea_20260724}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

bg_tag="${BG_COLOR_WEIGHT//./p}"
fg_tag="${FG_TRANS_WEIGHT//./p}"
scale_tag="${BINF_RESIDUAL_SCALE//./p}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-binf_${BINF_MODE}_s${scale_tag}_bg${bg_tag}_fg${fg_tag}_${MEDIUM_CONTEXT_MODE}_iui3_redsea_${MAX_NUM_ITERATIONS}}"
TIMESTAMP="${TIMESTAMP:-${EXPERIMENT_NAME}_${STAMP}}"
LOG_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}_${STAMP}"
TRAIN_LOG="${LOG_DIR}/train.log"
EVAL_LOG="${LOG_DIR}/eval.log"
CONFIG_PATH="${OUTPUT_DIR}/${EXPERIMENT_NAME}/water-splatting/${TIMESTAMP}/config.yml"
RENDER_DIR="${RENDER_ROOT}/${EXPERIMENT_NAME}_${STAMP}"
DIAG_DIR="${RENDER_DIR}/diagnostics"

mkdir -p "${LOG_DIR}" "${RENDER_DIR}" "${DIAG_DIR}"

{
  echo "experiment=${EXPERIMENT_NAME}"
  echo "timestamp=${TIMESTAMP}"
  echo "gpu=${GPU}"
  echo "python=${PYTHON}"
  echo "data=${DATA_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "render_dir=${RENDER_DIR}"
  echo "seed=${SEED}"
  echo "medium_context_mode=${MEDIUM_CONTEXT_MODE}"
  echo "b_inf_mode=${BINF_MODE}"
  echo "b_inf_residual_scale=${BINF_RESIDUAL_SCALE}"
  echo "lambda_background_water_color=${BG_COLOR_WEIGHT}"
  echo "lambda_foreground_transmission_reconstruction=${FG_TRANS_WEIGHT}"
  echo "foreground_transmission_gamma=${FG_TRANS_GAMMA}"
  echo "foreground_transmission_max_weight=${FG_TRANS_MAX_WEIGHT}"
  echo "foreground_transmission_detach_weight=${FG_TRANS_DETACH_WEIGHT}"
  echo "backscatter_region_mask_dir=${BACKSCATTER_REGION_MASK_DIR}"
  echo "background_water_mask_key=${BACKGROUND_WATER_MASK_KEY}"
  echo "foreground_water_mask_key=${FOREGROUND_WATER_MASK_KEY}"
  echo "backscatter_loss_start_step=${BACKSCATTER_LOSS_START_STEP}"
  echo "backscatter_loss_ramp_steps=${BACKSCATTER_LOSS_RAMP_STEPS}"
  echo "medium_predictor_mode=${MEDIUM_PREDICTOR_MODE}"
  echo "lambda_pseudo_depth=${LAMBDA_PSEUDO_DEPTH}"
  echo "lambda_medium_context_residual=${LAMBDA_MEDIUM_CONTEXT_RESIDUAL}"
  echo "reference_config=${REFERENCE_CONFIG}"
  echo "max_num_iterations=${MAX_NUM_ITERATIONS}"
  echo -n "git_commit="
  git -C "${REPO_DIR}" rev-parse HEAD || true
  git -C "${REPO_DIR}" status --short || true
  "${PYTHON}" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))'
} | tee "${LOG_DIR}/run_manifest.txt"

model_args=(
  --pipeline.model.num-steps "${MAX_NUM_ITERATIONS}"
  --pipeline.model.medium-context-mode "${MEDIUM_CONTEXT_MODE}"
  --pipeline.model.medium-camera-context-scale 1.0
  --pipeline.model.medium-camera-context-dropout 0.0
  --pipeline.model.medium-depth-context-detach True
  --pipeline.model.medium-depth-context-normalize True
  --pipeline.model.medium-depth-context-normalize-mode p95
  --pipeline.model.infinite-water-enabled False
  --pipeline.model.lambda-infinite-water-binf-rgb 0.0
  --pipeline.model.lambda-infinite-water-accumulation-zero 0.0
  --pipeline.model.lambda-infinite-water-near-zero 0.0
  --pipeline.model.b-inf-mode "${BINF_MODE}"
  --pipeline.model.b-inf-residual-scale "${BINF_RESIDUAL_SCALE}"
  --pipeline.model.lambda-background-water-color "${BG_COLOR_WEIGHT}"
  --pipeline.model.lambda-foreground-transmission-reconstruction "${FG_TRANS_WEIGHT}"
  --pipeline.model.foreground-transmission-gamma "${FG_TRANS_GAMMA}"
  --pipeline.model.foreground-transmission-max-weight "${FG_TRANS_MAX_WEIGHT}"
  --pipeline.model.foreground-transmission-detach-weight "${FG_TRANS_DETACH_WEIGHT}"
  --pipeline.model.background-water-mask-key "${BACKGROUND_WATER_MASK_KEY}"
  --pipeline.model.foreground-water-mask-key "${FOREGROUND_WATER_MASK_KEY}"
  --pipeline.model.backscatter-loss-start-step "${BACKSCATTER_LOSS_START_STEP}"
  --pipeline.model.backscatter-loss-ramp-steps "${BACKSCATTER_LOSS_RAMP_STEPS}"
  --pipeline.model.medium-predictor-mode "${MEDIUM_PREDICTOR_MODE}"
  --pipeline.model.lambda-pseudo-depth "${LAMBDA_PSEUDO_DEPTH}"
  --pipeline.model.lambda-medium-context-residual "${LAMBDA_MEDIUM_CONTEXT_RESIDUAL}"
)

if [[ -n "${BACKSCATTER_REGION_MASK_DIR}" ]]; then
  model_args+=(--pipeline.model.backscatter-region-mask-dir "${BACKSCATTER_REGION_MASK_DIR}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${NS_TRAIN}" water-splatting \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${TIMESTAMP}" \
  --vis tensorboard \
  --machine.seed "${SEED}" \
  --max-num-iterations "${MAX_NUM_ITERATIONS}" \
  "${model_args[@]}" \
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

if [[ "${RUN_CLOSURE_DIAG}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_backscatter_closure.py" \
    --load-config "${CONFIG_PATH}" \
    --output-dir "${DIAG_DIR}" \
    --max-images 4 \
    2>&1 | tee "${LOG_DIR}/closure_diag.log"
fi

if [[ "${RUN_FAR_DIAG}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_far_water_residual.py" \
    --load-config "${CONFIG_PATH}" \
    --mask-dir "${COMMON_FAR_MASK_DIR}" \
    --output-dir "${DIAG_DIR}/far_water" \
    2>&1 | tee "${LOG_DIR}/far_diag.log"
fi

if [[ "${RUN_REGION_DIAG}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_eval_regions.py" \
    --load-config "${CONFIG_PATH}" \
    --reference-config "${REFERENCE_CONFIG}" \
    --mask-dir "${REGION_MASK_DIR}" \
    --output-dir "${DIAG_DIR}/eval_regions" \
    --max-images 4 \
    2>&1 | tee "${LOG_DIR}/region_diag.log"
fi
