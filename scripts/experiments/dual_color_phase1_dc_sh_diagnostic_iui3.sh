#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="/opt/anaconda3/envs/water_splatting/bin/python"
M1_CONFIG="${M1_CONFIG:-${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
FAR_MASK_DIR="${FAR_MASK_DIR:-${REPO_DIR}/common_masks/m1_q90_iui3_redsea_20260724}"
REGION_MASK_DIR="${REGION_MASK_DIR:-${REPO_DIR}/common_masks/m1_auto_eval_regions_iui3_redsea_20260724}"
MAX_IMAGES="${MAX_IMAGES:-4}"
VARIANTS="${VARIANTS:-A0,A1,A2,A3,A4}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dual_color_phase1_dc_sh_iui3_redsea}"
OUT_DIR="${RENDER_ROOT}/${EXPERIMENT_NAME}_${STAMP}"
LOG_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}_${STAMP}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

{
  echo "experiment=${EXPERIMENT_NAME}"
  echo "timestamp=${STAMP}"
  echo "gpu=${GPU}"
  echo "m1_config=${M1_CONFIG}"
  echo "variants=${VARIANTS}"
  echo "max_images=${MAX_IMAGES}"
  echo "far_mask_dir=${FAR_MASK_DIR}"
  echo "region_mask_dir=${REGION_MASK_DIR}"
  echo "output_dir=${OUT_DIR}"
  echo -n "git_commit="
  git -C "${REPO_DIR}" rev-parse HEAD || true
  git -C "${REPO_DIR}" status --short || true
} | tee "${LOG_DIR}/run_manifest.txt"

ARGS=(
  "${REPO_DIR}/scripts/diagnostics/diagnose_dc_sh_clear_appearance.py"
  --load-config "${M1_CONFIG}"
  --output-dir "${OUT_DIR}"
  --max-images "${MAX_IMAGES}"
  --variants "${VARIANTS}"
  --far-mask-dir "${FAR_MASK_DIR}"
  --region-mask-dir "${REGION_MASK_DIR}"
)
if [[ "${SAVE_IMAGES}" == "1" ]]; then
  ARGS+=(--save-images)
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "${LOG_DIR}/diagnose.log"
