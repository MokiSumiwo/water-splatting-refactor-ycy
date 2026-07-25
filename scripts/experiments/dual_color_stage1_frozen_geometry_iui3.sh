#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
NS_TRAIN="/opt/anaconda3/envs/water_splatting/bin/ns-train"
NS_EVAL="/opt/anaconda3/envs/water_splatting/bin/ns-eval"
PYTHON="/opt/anaconda3/envs/water_splatting/bin/python"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
BASE_LOAD_DIR="${BASE_LOAD_DIR:-${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/nerfstudio_models}"
BASE_STEP="${BASE_STEP:-14999}"
FINETUNE_STEPS="${FINETUNE_STEPS:-2000}"
TRAIN_MAX_NUM_ITERATIONS="${TRAIN_MAX_NUM_ITERATIONS:-${FINETUNE_STEPS}}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-$((BASE_STEP + FINETUNE_STEPS))}"
SEED="${SEED:-42}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
CLEAR_SH_LUMINANCE_SCALE="${CLEAR_SH_LUMINANCE_SCALE:-1.0}"
CLEAR_SH_CHROMA_SCALE="${CLEAR_SH_CHROMA_SCALE:-0.0}"
LAMBDA_INTRINSIC_NEAR_ANCHOR="${LAMBDA_INTRINSIC_NEAR_ANCHOR:-0.0}"
LAMBDA_VIEW_RESIDUAL_MEAN="${LAMBDA_VIEW_RESIDUAL_MEAN:-0.0}"
LAMBDA_CLEAR_CHROMA="${LAMBDA_CLEAR_CHROMA:-0.0}"
DUAL_COLOR_LOSS_START_STEP="${DUAL_COLOR_LOSS_START_STEP:-0}"
DUAL_COLOR_LOSS_RAMP_STEPS="${DUAL_COLOR_LOSS_RAMP_STEPS:-0}"
DUAL_COLOR_NEAR_TRANSMISSION_THRESHOLD="${DUAL_COLOR_NEAR_TRANSMISSION_THRESHOLD:-0.70}"
DUAL_COLOR_NEAR_TRANSMISSION_TEMP="${DUAL_COLOR_NEAR_TRANSMISSION_TEMP:-0.10}"
DUAL_COLOR_FREEZE_GEOMETRY="${DUAL_COLOR_FREEZE_GEOMETRY:-True}"
DUAL_COLOR_FREEZE_MEDIUM="${DUAL_COLOR_FREEZE_MEDIUM:-True}"
MEDIUM_LR="${MEDIUM_LR:-0.0001}"
MEDIUM_LR_FINAL="${MEDIUM_LR_FINAL:-0.000015}"
FAR_MASK_DIR="${FAR_MASK_DIR:-${REPO_DIR}/common_masks/m1_q90_iui3_redsea_20260724}"
REGION_MASK_DIR="${REGION_MASK_DIR:-${REPO_DIR}/common_masks/m1_auto_eval_regions_iui3_redsea_20260724}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dual_color_stage1_luma${CLEAR_SH_LUMINANCE_SCALE}_chroma${CLEAR_SH_CHROMA_SCALE}_near${LAMBDA_INTRINSIC_NEAR_ANCHOR}_mean${LAMBDA_VIEW_RESIDUAL_MEAN}_chr${LAMBDA_CLEAR_CHROMA}_seed${SEED}_iui3_redsea_${FINETUNE_STEPS}}"
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
  echo "seed=${SEED}"
  echo "data=${DATA_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "render_dir=${RENDER_DIR}"
  echo "base_load_dir=${BASE_LOAD_DIR}"
  echo "base_step=${BASE_STEP}"
  echo "finetune_steps=${FINETUNE_STEPS}"
  echo "train_max_num_iterations=${TRAIN_MAX_NUM_ITERATIONS}"
  echo "model_num_steps=${MODEL_NUM_STEPS}"
  echo "dual_color_enabled=True"
  echo "dual_color_freeze_geometry=${DUAL_COLOR_FREEZE_GEOMETRY}"
  echo "dual_color_freeze_medium=${DUAL_COLOR_FREEZE_MEDIUM}"
  echo "medium_lr=${MEDIUM_LR}"
  echo "medium_lr_final=${MEDIUM_LR_FINAL}"
  echo "clear_sh_luminance_scale=${CLEAR_SH_LUMINANCE_SCALE}"
  echo "clear_sh_chroma_scale=${CLEAR_SH_CHROMA_SCALE}"
  echo "lambda_intrinsic_near_anchor=${LAMBDA_INTRINSIC_NEAR_ANCHOR}"
  echo "lambda_view_residual_mean=${LAMBDA_VIEW_RESIDUAL_MEAN}"
  echo "lambda_clear_chroma=${LAMBDA_CLEAR_CHROMA}"
  echo "dual_color_loss_start_step=${DUAL_COLOR_LOSS_START_STEP}"
  echo "dual_color_loss_ramp_steps=${DUAL_COLOR_LOSS_RAMP_STEPS}"
  echo -n "git_commit="
  git -C "${REPO_DIR}" rev-parse HEAD || true
  git -C "${REPO_DIR}" status --short || true
} | tee "${LOG_DIR}/run_manifest.txt"

MEDIUM_OPT_ARGS=()
if [[ "${DUAL_COLOR_FREEZE_MEDIUM}" == "False" || "${DUAL_COLOR_FREEZE_MEDIUM}" == "false" ]]; then
  MEDIUM_OPT_ARGS=(
    --optimizers.medium-mlp.optimizer.lr "${MEDIUM_LR}"
    --optimizers.medium-mlp.scheduler.lr-final "${MEDIUM_LR_FINAL}"
    --optimizers.direction-encoding.optimizer.lr "${MEDIUM_LR}"
    --optimizers.direction-encoding.scheduler.lr-final "${MEDIUM_LR_FINAL}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${NS_TRAIN}" water-splatting \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${TIMESTAMP}" \
  --vis tensorboard \
  --machine.seed "${SEED}" \
  --max-num-iterations "${TRAIN_MAX_NUM_ITERATIONS}" \
  --load-dir "${BASE_LOAD_DIR}" \
  --load-step "${BASE_STEP}" \
  --pipeline.model.num-steps "${MODEL_NUM_STEPS}" \
  --pipeline.model.medium-context-mode dir_xy_camera \
  --pipeline.model.medium-camera-context-scale 1.0 \
  --pipeline.model.medium-camera-context-dropout 0.0 \
  --pipeline.model.medium-depth-context-detach True \
  --pipeline.model.medium-depth-context-normalize True \
  --pipeline.model.medium-depth-context-normalize-mode p95 \
  --pipeline.model.infinite-water-enabled False \
  --pipeline.model.constrained-appearance-enabled False \
  --pipeline.model.dual-color-enabled True \
  --pipeline.model.clear-sh-luminance-scale "${CLEAR_SH_LUMINANCE_SCALE}" \
  --pipeline.model.clear-sh-chroma-scale "${CLEAR_SH_CHROMA_SCALE}" \
  --pipeline.model.lambda-intrinsic-near-anchor "${LAMBDA_INTRINSIC_NEAR_ANCHOR}" \
  --pipeline.model.lambda-view-residual-mean "${LAMBDA_VIEW_RESIDUAL_MEAN}" \
  --pipeline.model.lambda-clear-chroma "${LAMBDA_CLEAR_CHROMA}" \
  --pipeline.model.dual-color-loss-start-step "${DUAL_COLOR_LOSS_START_STEP}" \
  --pipeline.model.dual-color-loss-ramp-steps "${DUAL_COLOR_LOSS_RAMP_STEPS}" \
  --pipeline.model.dual-color-near-transmission-threshold "${DUAL_COLOR_NEAR_TRANSMISSION_THRESHOLD}" \
  --pipeline.model.dual-color-near-transmission-temp "${DUAL_COLOR_NEAR_TRANSMISSION_TEMP}" \
  --pipeline.model.dual-color-freeze-geometry "${DUAL_COLOR_FREEZE_GEOMETRY}" \
  --pipeline.model.dual-color-freeze-medium "${DUAL_COLOR_FREEZE_MEDIUM}" \
  --pipeline.model.stop-split-at 0 \
  --pipeline.model.continue-cull-post-densification False \
  "${MEDIUM_OPT_ARGS[@]}" \
  colmap \
  --data "${DATA_PATH}" \
  --images-path images/ColorImage \
  --colmap-path sparse/0 \
  --downscale-factor 1 \
  2>&1 | tee "${TRAIN_LOG}"

"${PYTHON}" - "${CONFIG_PATH}" "${BASE_STEP}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
base_step = sys.argv[2]
text = path.read_text(encoding="utf8")
text = text.replace(f"load_step: {base_step}\n", "load_step: null\n")
path.write_text(text, encoding="utf8")
PY

if [[ "${RUN_EVAL}" == "1" ]]; then
  pushd "${RENDER_DIR}" >/dev/null
  CUDA_VISIBLE_DEVICES="${GPU}" "${NS_EVAL}" \
    --load-config "${CONFIG_PATH}" \
    --render-output-path "${RENDER_DIR}" \
    2>&1 | tee "${EVAL_LOG}"
  popd >/dev/null
fi

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_eval_regions.py" \
    --load-config "${CONFIG_PATH}" \
    --reference-config "${REFERENCE_CONFIG}" \
    --mask-dir "${REGION_MASK_DIR}" \
    --output-dir "${RENDER_DIR}/eval_regions" \
    --max-images 4 \
    --save-heatmaps \
    2>&1 | tee "${LOG_DIR}/eval_regions.log"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_far_water_residual.py" \
    --load-config "${CONFIG_PATH}" \
    --mask-dir "${FAR_MASK_DIR}" \
    --output-dir "${RENDER_DIR}/common_far_m1_q90" \
    2>&1 | tee "${LOG_DIR}/far_water.log"
fi
