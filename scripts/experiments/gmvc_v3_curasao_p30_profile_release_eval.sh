#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

GPU="${GPU:-6}"
VARIANTS="${VARIANTS:-A0,C30,STOP,DECAY}"
STEPS="${STEPS:-14000,15000}"
STAMP_BASE="${STAMP:-20260805_gmvc_v3_p30_profile_release_13k_to_15k}"
ROOT="${ROOT:-${REPO_DIR}/renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k}"
LOAD_STEP_START="${LOAD_STEP_START:-13000}"
TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-15000}"
SAVE_RGB_RENDERS="${SAVE_RGB_RENDERS:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

EVALF_BANK="${EVALF_BANK:-${REPO_DIR}/renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"
EVALG_BANK="${EVALG_BANK:-${REPO_DIR}/renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"
START_ROOT="${START_ROOT:-${REPO_DIR}/renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k}"

variant_slug() {
  case "$1" in
    A0|a0) echo "a0" ;;
    A0_MHOLD|a0_mhold|A0-MHOLD|a0-mhold) echo "a0_mhold" ;;
    C30|c30) echo "c30" ;;
    STOP|stop) echo "stop" ;;
    H500|h500|H500_STOP|h500_stop) echo "h500" ;;
    MHOLD|mhold|P30_MHOLD|p30_mhold) echo "mhold" ;;
    DECAY|decay) echo "decay" ;;
    *) echo "Unknown variant $1" >&2; exit 2 ;;
  esac
}

experiment_name() {
  local slug="$1"
  echo "gmvc_v3_p30_release_${slug}_curasao_seed42_step${LOAD_STEP_START}_to_${TARGET_FINAL_STEP}"
}

config_path() {
  local slug="$1"
  local exp
  exp="$(experiment_name "${slug}")"
  echo "${REPO_DIR}/outputs/${exp}/water-splatting/${exp}_${STAMP_BASE}_${slug}/config.yml"
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
    --reference-variant C30 \
    --start-root "${START_ROOT}" \
    --start-step 13000 \
    --start-variant P30 \
    --output "${ROOT}/summary.json"
fi
