#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-Curasao}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260807_d010_persistence}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/dewater_d010_persistence_20260807}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
VIS_ROOT="${VIS_ROOT:-${RENDER_ROOT}/dewater_d010_persistence_20260807}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${OUTPUT_DIR}}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"

case "${SCENE}" in
  Curasao|curasao)
    SCENE_NAME="Curasao"
    SCENE_SLUG="curasao"
    DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Curasao}"
    D100_CONFIG="${D100_CONFIG:-${REPO_DIR}/outputs/dewater_direct_d100_curasao_seed42_step10000_to_13000/water-splatting/dewater_direct_d100_curasao_seed42_step10000_to_13000_20260807_dewater_direct_optical_depth_d100_g1p00/config.yml}"
    D100_CKPT_13K="${D100_CKPT_13K:-${REPO_DIR}/outputs/dewater_direct_d100_curasao_seed42_step10000_to_13000/water-splatting/dewater_direct_d100_curasao_seed42_step10000_to_13000_20260807_dewater_direct_optical_depth_d100_g1p00/nerfstudio_models/step-000013000.ckpt}"
    D010_CONFIG="${D010_CONFIG:-${REPO_DIR}/outputs/dewater_direct_d010_curasao_seed42_step10000_to_13000/water-splatting/dewater_direct_d010_curasao_seed42_step10000_to_13000_20260807_dewater_direct_optical_depth_d010_g0p10/config.yml}"
    D010_CKPT_13K="${D010_CKPT_13K:-${REPO_DIR}/outputs/dewater_direct_d010_curasao_seed42_step10000_to_13000/water-splatting/dewater_direct_d010_curasao_seed42_step10000_to_13000_20260807_dewater_direct_optical_depth_d010_g0p10/nerfstudio_models/step-000013000.ckpt}"
    ;;
  *)
    echo "This persistence audit is intentionally limited to Curasao." >&2
    exit 2
    ;;
esac

for required in "${D100_CONFIG}" "${D100_CKPT_13K}" "${D010_CONFIG}" "${D010_CKPT_13K}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required persistence input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}" "${VIS_ROOT}" "${LOG_ROOT}/dewater_d010_persistence_20260807"

runs=(
  "D100-PERSIST:1.00:${D100_CONFIG}:${D100_CKPT_13K}:d100_persist"
  "D010-PERSIST:0.10:${D010_CONFIG}:${D010_CKPT_13K}:d010_persist"
)

for item in "${runs[@]}"; do
  label="$(cut -d: -f1 <<<"${item}")"
  gamma="$(cut -d: -f2 <<<"${item}")"
  source_config="$(cut -d: -f3 <<<"${item}")"
  source_ckpt="$(cut -d: -f4 <<<"${item}")"
  slug="$(cut -d: -f5 <<<"${item}")"
  gamma_tag="${gamma//./p}"
  exp="dewater_${slug}_${SCENE_SLUG}_seed${SEED}_step13000_to_15000"
  stamp="${STAMP_BASE}_${slug}_g${gamma_tag}"
  config_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/config.yml"
  final_ckpt="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-000015000.ckpt"

  if [[ -f "${final_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME} ${label}] Reusing final checkpoint: ${final_ckpt}"
  else
    echo "[${SCENE_NAME} ${label}] Continuing step 13000 -> 15000 with gamma_D=${gamma}."
    env \
      GPU="${GPU}" \
      PYTHON="${PYTHON}" \
      SCENE_SLUG="${SCENE_SLUG}" \
      DATA_PATH="${DATA_PATH}" \
      M1_LOAD_CONFIG="${source_config}" \
      M1_LOAD_CHECKPOINT="${source_ckpt}" \
      OUTPUT_DIR="${OUTPUT_DIR}" \
      RENDER_ROOT="${RENDER_ROOT}" \
      LOG_ROOT="${LOG_ROOT}" \
      EXPERIMENT_NAME="${exp}" \
      STAMP="${stamp}" \
      TIMESTAMP="${exp}_${stamp}" \
      SEED="${SEED}" \
      LOAD_STEP="13000" \
      TARGET_FINAL_STEP="15000" \
      MAX_NUM_ITERATIONS="2000" \
      MODEL_NUM_STEPS="15000" \
      STEPS_PER_SAVE="1000" \
      SAVE_ONLY_LATEST_CHECKPOINT="False" \
      RUN_EVAL="0" \
      RUN_CLOSURE_DIAG="0" \
      GMVC_VARIANT="dewater_${slug}" \
      GMVC_ENABLED="False" \
      GMVC_DIAGNOSTIC_ONLY="False" \
      GMVC_V2_ENABLED="False" \
      GMVC_V3_ENABLED="False" \
      LAMBDA_GMVC_PROFILE="0.0" \
      LAMBDA_GMVC_OBJECT="0.0" \
      DIRECT_OPTICAL_DEPTH_SCALE="${gamma}" \
      MEDIUM_BACKGROUND_SUPERVISION_ENABLED="False" \
      MEDIUM_BACKGROUND_SUPERVISION_LAMBDA="0.0" \
      "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
  fi

  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    for step in 14000 15000; do
      ckpt_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-$(printf "%09d" "${step}").ckpt"
      if [[ ! -f "${ckpt_path}" ]]; then
        echo "Missing checkpoint for diagnostics: ${ckpt_path}" >&2
        exit 1
      fi
      echo "[${SCENE_NAME} ${label}] Rendering diagnostics for step ${step}."
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
        --scene "${SCENE_NAME}" \
        --load-config "${config_path}" \
        --load-step "${step}" \
        --split eval \
        --test-mode test \
        --max-images -1 \
        --output-dir "${VIS_ROOT}/${label}/step_${step}"
    done
  fi
done

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_dewater_d010_persistence.py" \
    --scene "${SCENE_NAME}" \
    --a-root "${RENDER_ROOT}/dewater_optical_depth_20260807/A" \
    --persist-root "${VIS_ROOT}" \
    --output-json "${SUMMARY_ROOT}/d010_persistence_summary.json" \
    --output-csv "${SUMMARY_ROOT}/d010_persistence_summary.csv"
fi
