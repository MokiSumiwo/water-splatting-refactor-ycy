#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"
GPU="${GPU:-6}"
SCENE="${SCENE:-Curasao}"
STEP_LIST="${STEP_LIST:-1000 3000 5000 8000 10000 13000 15000}"
MAX_IMAGES="${MAX_IMAGES:-3}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders/dewater_bounded_sh3_scratch_20260808}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/dewater_bounded_sh3_scratch_20260808}"

D100_CONFIG="${D100_CONFIG:-${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml}"
D010_CONFIG="${D010_CONFIG:-${REPO_DIR}/outputs/dewater_d010_scratch_20260807/dewater_d010_scratch_curasao_seed42_step0_to_15000/water-splatting/dewater_d010_scratch_curasao_seed42_step0_to_15000_20260807_d010_scratch_g0p10/config.yml}"
BND_CONFIG="${BND_CONFIG:-${OUTPUT_ROOT}/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/config.yml}"
D010_BND_CONFIG="${D010_BND_CONFIG:-${OUTPUT_ROOT}/dewater_bounded_sh3_curasao_seed42_d010-bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_d010-bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_d010-bnd-scratch_g0p10/config.yml}"

run_diag() {
  local run_name="$1"
  local config_path="$2"
  local nominal_step="$3"
  local load_step="$4"
  local out_dir="${RENDER_ROOT}/diagnostics/${run_name}/step_${nominal_step}"
  echo "[diagnose] ${run_name} nominal_step=${nominal_step} load_step=${load_step}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
    --scene "${SCENE}" \
    --load-config "${config_path}" \
    --load-step "${load_step}" \
    --split eval \
    --test-mode test \
    --max-images "${MAX_IMAGES}" \
    --output-dir "${out_dir}"
}

summary_args=()

run_diag "D100" "${D100_CONFIG}" 15000 14999
summary_args+=(--summary "D100:15000:${RENDER_ROOT}/diagnostics/D100/step_15000/summary.json")

run_diag "D010" "${D010_CONFIG}" 15000 14999
summary_args+=(--summary "D010:15000:${RENDER_ROOT}/diagnostics/D010/step_15000/summary.json")

for step in ${STEP_LIST}; do
  load_step="${step}"
  if [[ "${step}" == "15000" ]]; then
    load_step=14999
  fi
  run_diag "BND" "${BND_CONFIG}" "${step}" "${load_step}"
  summary_args+=(--summary "BND:${step}:${RENDER_ROOT}/diagnostics/BND/step_${step}/summary.json")
  run_diag "D010-BND" "${D010_BND_CONFIG}" "${step}" "${load_step}"
  summary_args+=(--summary "D010-BND:${step}:${RENDER_ROOT}/diagnostics/D010-BND/step_${step}/summary.json")
done

"${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_bounded_sh3_scratch.py" \
  "${summary_args[@]}" \
  --final-step 15000 \
  --output-dir "${RENDER_ROOT}"

"${PYTHON}" "${REPO_DIR}/scripts/diagnostics/render_bounded_sh3_2x2_comparison.py" \
  --scene "${SCENE}" \
  --step 15000 \
  --run "D100=${RENDER_ROOT}/diagnostics/D100/step_15000" \
  --run "D010=${RENDER_ROOT}/diagnostics/D010/step_15000" \
  --run "BND=${RENDER_ROOT}/diagnostics/BND/step_15000" \
  --run "D010-BND=${RENDER_ROOT}/diagnostics/D010-BND/step_15000" \
  --output-dir "${RENDER_ROOT}/visual_compare_2x2_step_15000"
