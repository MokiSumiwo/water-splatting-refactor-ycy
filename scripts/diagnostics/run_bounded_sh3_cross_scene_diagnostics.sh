#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"
GPU="${GPU:-6}"
STEP_LIST="${STEP_LIST:-1000 3000 5000 8000 10000 13000 15000}"
MAX_IMAGES="${MAX_IMAGES:--1}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders/dewater_bounded_sh3_cross_scene_20260808}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/dewater_bounded_sh3_cross_scene_20260808}"

declare -A M1_CONFIGS=(
  [Curasao]="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
  [JapaneseGradens]="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
  [IUI3]="${REPO_DIR}/outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/config.yml"
  [Panama]="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
)

declare -A BND_CONFIGS=(
  [Curasao]="${REPO_DIR}/outputs/dewater_bounded_sh3_scratch_20260808/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/config.yml"
  [JapaneseGradens]="${OUTPUT_ROOT}/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_japanesegradens_bnd_g1p00/config.yml"
  [IUI3]="${OUTPUT_ROOT}/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/config.yml"
  [Panama]="${OUTPUT_ROOT}/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml"
)

run_diag() {
  local scene="$1"
  local run_name="$2"
  local config_path="$3"
  local nominal_step="$4"
  local load_step="$5"
  local out_dir="${RENDER_ROOT}/${scene}/diagnostics/${run_name}/step_${nominal_step}"
  echo "[diagnose] ${scene} ${run_name} nominal_step=${nominal_step} load_step=${load_step}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
    --scene "${scene}" \
    --load-config "${config_path}" \
    --load-step "${load_step}" \
    --split eval \
    --test-mode test \
    --max-images "${MAX_IMAGES}" \
    --output-dir "${out_dir}"
}

summary_args=()
config_args=()

for scene in Curasao JapaneseGradens IUI3 Panama; do
  config_args+=(--baseline-config "${scene}:${M1_CONFIGS[${scene}]}")
  config_args+=(--bnd-config "${scene}:${BND_CONFIGS[${scene}]}")

  run_diag "${scene}" "M1" "${M1_CONFIGS[${scene}]}" 15000 14999
  summary_args+=(--summary "${scene}:M1:15000:${RENDER_ROOT}/${scene}/diagnostics/M1/step_15000/summary.json")

  if [[ "${scene}" == "Curasao" ]]; then
    run_diag "${scene}" "BND" "${BND_CONFIGS[${scene}]}" 15000 14999
    summary_args+=(--summary "${scene}:BND:15000:${RENDER_ROOT}/${scene}/diagnostics/BND/step_15000/summary.json")
  else
    for step in ${STEP_LIST}; do
      load_step="${step}"
      if [[ "${step}" == "15000" ]]; then
        load_step=14999
      fi
      run_diag "${scene}" "BND" "${BND_CONFIGS[${scene}]}" "${step}" "${load_step}"
      summary_args+=(--summary "${scene}:BND:${step}:${RENDER_ROOT}/${scene}/diagnostics/BND/step_${step}/summary.json")
    done
  fi
done

"${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_bounded_sh3_cross_scene.py" \
  "${summary_args[@]}" \
  "${config_args[@]}" \
  --start-head "$(git -C "${REPO_DIR}" rev-parse HEAD)" \
  --seafree-reference-commit "7797e97dae831029ac89ae9f37b3c3d69ec2cf6c" \
  --output-dir "${RENDER_ROOT}/four_scene_summary"

"${PYTHON}" "${REPO_DIR}/scripts/diagnostics/render_bounded_sh3_cross_scene_comparison.py" \
  --scene-pair "Curasao:${RENDER_ROOT}/Curasao/diagnostics/M1/step_15000:${RENDER_ROOT}/Curasao/diagnostics/BND/step_15000" \
  --scene-pair "JapaneseGradens:${RENDER_ROOT}/JapaneseGradens/diagnostics/M1/step_15000:${RENDER_ROOT}/JapaneseGradens/diagnostics/BND/step_15000" \
  --scene-pair "IUI3:${RENDER_ROOT}/IUI3/diagnostics/M1/step_15000:${RENDER_ROOT}/IUI3/diagnostics/BND/step_15000" \
  --scene-pair "Panama:${RENDER_ROOT}/Panama/diagnostics/M1/step_15000:${RENDER_ROOT}/Panama/diagnostics/BND/step_15000" \
  --output-dir "${RENDER_ROOT}"
