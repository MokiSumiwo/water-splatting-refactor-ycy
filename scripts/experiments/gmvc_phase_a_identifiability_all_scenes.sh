#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/water_splatting/bin/python}"
STAMP="${STAMP:-20260803_gmvc_phase_a}"
SPLIT="${SPLIT:-train}"
SAMPLES_PER_VIEW="${SAMPLES_PER_VIEW:-4096}"
MAX_IMAGES="${MAX_IMAGES:-0}"
TARGET_NEIGHBOR_WINDOW="${TARGET_NEIGHBOR_WINDOW:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders/gmvc_phase_a_identifiability_${STAMP}}"

run_scene() {
  local gpu="$1"
  local scene="$2"
  local config="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/diagnostics/diagnose_gmvc_identifiability.py \
    --load-config "${config}" \
    --split "${SPLIT}" \
    --max-images "${MAX_IMAGES}" \
    --samples-per-view "${SAMPLES_PER_VIEW}" \
    --target-neighbor-window "${TARGET_NEIGHBOR_WINDOW}" \
    --output-dir "${OUTPUT_ROOT}/${scene}"
}

run_scene "${GPU_JG:-6}" "japanesegradens" \
  "outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml" &
run_scene "${GPU_IUI3:-7}" "iui3" \
  "outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml" &
run_scene "${GPU_CURASAO:-8}" "curasao" \
  "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml" &
run_scene "${GPU_PANAMA:-9}" "panama" \
  "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml" &

wait
echo "GMVC Phase A outputs: ${OUTPUT_ROOT}"
