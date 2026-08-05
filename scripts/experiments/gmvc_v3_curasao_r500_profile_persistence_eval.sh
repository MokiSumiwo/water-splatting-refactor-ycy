#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

GPU="${GPU:-6}"
VARIANTS="${VARIANTS:-A0,P40,P35,P30}"
STEPS="${STEPS:-11000,12000,13000}"
STAMP="${STAMP:-20260805_gmvc_v3_curasao_r500_profile_persistence_3k}"
ROOT="${ROOT:-${REPO_DIR}/renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k}"
LOAD_STEP_START="${LOAD_STEP_START:-10000}"
TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-13000}"
SAVE_RGB_RENDERS="${SAVE_RGB_RENDERS:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

TRAIN_BANK="${TRAIN_BANK:-${REPO_DIR}/renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"
EVALF_BANK="${EVALF_BANK:-${REPO_DIR}/renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"
EVALG_BANK="${EVALG_BANK:-${REPO_DIR}/renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"

variant_slug() {
  case "$1" in
    A0|a0) echo "a0" ;;
    P40|p40) echo "p40" ;;
    P35|p35) echo "p35" ;;
    P30|p30) echo "p30" ;;
    *) echo "Unknown variant $1" >&2; exit 2 ;;
  esac
}

experiment_name() {
  local slug="$1"
  if [[ "${slug}" == "a0" ]]; then
    echo "gmvc_v3_a0_profile_persistence3k_curasao_seed42_step${LOAD_STEP_START}_to_${TARGET_FINAL_STEP}"
  else
    echo "gmvc_v3_r500_${slug}_profile_persistence3k_curasao_seed42_step${LOAD_STEP_START}_to_${TARGET_FINAL_STEP}"
  fi
}

config_path() {
  local slug="$1"
  local exp
  exp="$(experiment_name "${slug}")"
  if [[ "${slug}" == "a0" ]]; then
    echo "${REPO_DIR}/outputs/${exp}/water-splatting/${exp}_${STAMP}_${slug}/config.yml"
  else
    echo "${REPO_DIR}/outputs/${exp}/water-splatting/${exp}_${STAMP}_${slug}_${slug}_r500_g000/config.yml"
  fi
}

run_fixed_bank() {
  local config="$1"
  local step="$2"
  local bank="$3"
  local out_dir="$4"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_gmvc_fixed_bank.py" \
    --load-config "${config}" \
    --load-step "${step}" \
    --track-bank "${bank}" \
    --max-tracks 30000 \
    --output-dir "${out_dir}" \
    2>&1 | tee "${out_dir}/stdout.log"
}

IFS=',' read -r -a variant_array <<< "${VARIANTS}"
IFS=',' read -r -a step_array <<< "${STEPS}"

for variant in "${variant_array[@]}"; do
  slug="$(variant_slug "${variant}")"
  config="$(config_path "${slug}")"
  if [[ ! -f "${config}" ]]; then
    echo "Missing config for ${variant}: ${config}" >&2
    exit 1
  fi
  for step in "${step_array[@]}"; do
    ckpt="${config%/config.yml}/nerfstudio_models/step-$(printf "%09d" "${step}").ckpt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "Missing checkpoint for ${variant} step ${step}: ${ckpt}" >&2
      exit 1
    fi
    rgb_dir="${ROOT}/${slug}/step${step}/rgb"
    mkdir -p "${rgb_dir}"
    rgb_args=(
      --load-config "${config}"
      --load-step "${step}"
      --output-path "${rgb_dir}/rgb_metrics.json"
    )
    if [[ "${SAVE_RGB_RENDERS}" == "1" ]]; then
      rgb_args+=(--render-output-path "${rgb_dir}/renders")
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/evaluate_checkpoint_metrics.py" \
      "${rgb_args[@]}" \
      2>&1 | tee "${rgb_dir}/stdout.log"
    run_fixed_bank "${config}" "${step}" "${EVALF_BANK}" "${ROOT}/${slug}/step${step}/evalf"
    run_fixed_bank "${config}" "${step}" "${EVALG_BANK}" "${ROOT}/${slug}/step${step}/evalg"
  done
done

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_gmvc_persistence.py" \
    --root "${ROOT}" \
    --variants "${VARIANTS}" \
    --steps "${STEPS}" \
    --output "${ROOT}/summary.json"
fi
