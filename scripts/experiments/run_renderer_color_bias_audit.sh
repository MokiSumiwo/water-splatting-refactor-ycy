#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"
GPU="${GPU:-6}"
STAMP="${STAMP:-20260731_renderer_color_bias}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
MAX_IMAGES="${MAX_IMAGES:-4}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
COMPUTE_DC="${COMPUTE_DC:-1}"

IUI3_CONFIG="${IUI3_CONFIG:-${REPO_DIR}/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml}"
IUI3_FAR_MASK="${IUI3_FAR_MASK:-${REPO_DIR}/common_masks/m1_q90_iui3_redsea_20260724}"
IUI3_REGION_MASK="${IUI3_REGION_MASK:-${REPO_DIR}/common_masks/m1_auto_eval_regions_iui3_redsea_20260724}"

JGRAD_CONFIG="${JGRAD_CONFIG:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml}"
JGRAD_FAR_MASK="${JGRAD_FAR_MASK:-${REPO_DIR}/common_masks/cross_scene_japanesegradens_redsea_m1_q90_seed42_20260730_cross_scene}"
JGRAD_REGION_MASK="${JGRAD_REGION_MASK:-${REPO_DIR}/common_masks/cross_scene_japanesegradens_redsea_m1_eval_regions_seed42_20260730_cross_scene}"

OUT_ROOT="${OUT_ROOT:-${RENDER_ROOT}/renderer_color_bias_audit_${STAMP}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/renderer_color_bias_audit_${STAMP}}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

run_scene() {
  local scene_name="$1"
  local config="$2"
  local far_mask="$3"
  local region_mask="$4"
  local out_dir="${OUT_ROOT}/${scene_name}"
  local log_file="${LOG_DIR}/${scene_name}.log"

  if [[ ! -f "${config}" ]]; then
    echo "Missing config for ${scene_name}: ${config}" >&2
    exit 1
  fi
  if [[ ! -d "${far_mask}" ]]; then
    echo "Missing far mask for ${scene_name}: ${far_mask}" >&2
    exit 1
  fi
  if [[ ! -d "${region_mask}" ]]; then
    echo "Missing region mask for ${scene_name}: ${region_mask}" >&2
    exit 1
  fi

  {
    echo "scene=${scene_name}"
    echo "gpu=${GPU}"
    echo "python=${PYTHON}"
    echo "config=${config}"
    echo "far_mask=${far_mask}"
    echo "region_mask=${region_mask}"
    echo "out_dir=${out_dir}"
    echo "max_images=${MAX_IMAGES}"
    echo "save_images=${SAVE_IMAGES}"
    echo "compute_dc=${COMPUTE_DC}"
  } | tee "${LOG_DIR}/${scene_name}_manifest.txt"

  local save_arg=()
  if [[ "${SAVE_IMAGES}" == "1" ]]; then
    save_arg+=(--save-images)
  fi
  local dc_arg=()
  if [[ "${COMPUTE_DC}" == "1" ]]; then
    dc_arg+=(--compute-dc)
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_renderer_color_bias.py" \
    --load-config "${config}" \
    --scene-name "${scene_name}" \
    --output-dir "${out_dir}" \
    --far-mask-dir "${far_mask}" \
    --region-mask-dir "${region_mask}" \
    --max-images "${MAX_IMAGES}" \
    "${save_arg[@]}" \
    "${dc_arg[@]}" \
    2>&1 | tee "${log_file}"
}

run_scene "iui3_redsea_m1" "${IUI3_CONFIG}" "${IUI3_FAR_MASK}" "${IUI3_REGION_MASK}"
run_scene "japanesegradens_redsea_m1" "${JGRAD_CONFIG}" "${JGRAD_FAR_MASK}" "${JGRAD_REGION_MASK}"

echo "saved=${OUT_ROOT}"
