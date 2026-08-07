#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-Curasao}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260807_dewater_medium_background_supervision}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_MASKS="${FORCE_MASKS:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"

case "${SCENE}" in
  Curasao|curasao)
    SCENE_NAME="Curasao"
    SCENE_SLUG="curasao"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Curasao"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    ;;
  *)
    echo "This first B sweep is intentionally limited to Curasao." >&2
    exit 2
    ;;
esac

DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}"
M1_CONFIG="${M1_CONFIG:-${M1_CONFIG_DEFAULT}}"
M1_CKPT_10K="${M1_CKPT_10K:-${M1_CKPT_DEFAULT}}"
VIS_ROOT="${VIS_ROOT:-${RENDER_ROOT}/dewater_optical_depth_20260807/B}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${OUTPUT_DIR}/dewater_optical_depth_20260807}"
MASK_DIR="${MASK_DIR:-${REPO_DIR}/common_masks/dewater_${SCENE_SLUG}_m1_step10000_train_background_water_20260807}"
DRIVER_LOG_DIR="${LOG_ROOT}/dewater_optical_depth_20260807"

if [[ ! -f "${M1_CONFIG}" ]]; then
  echo "Missing M1 config: ${M1_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${M1_CKPT_10K}" ]]; then
  echo "Missing M1 step-10000 checkpoint: ${M1_CKPT_10K}" >&2
  exit 1
fi

mkdir -p "${VIS_ROOT}" "${SUMMARY_ROOT}" "${DRIVER_LOG_DIR}"

if [[ ! -f "${MASK_DIR}/metadata.json" || "${FORCE_MASKS}" == "1" ]]; then
  echo "[${SCENE_NAME} B] Building detached train background-water masks: ${MASK_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/build_dewater_background_masks.py" \
    --load-config "${M1_CONFIG}" \
    --load-step 10000 \
    --split train \
    --test-mode inference \
    --output-dir "${MASK_DIR}" \
    --save-png \
    --fail-on-coverage-gate \
    2>&1 | tee "${DRIVER_LOG_DIR}/build_dewater_background_masks.log"
else
  echo "[${SCENE_NAME} B] Reusing detached train masks: ${MASK_DIR}"
fi

runs=(
  "BG000:0.00:False"
  "BG010:0.01:True"
)

for item in "${runs[@]}"; do
  label="$(cut -d: -f1 <<<"${item}")"
  lambda_bg="$(cut -d: -f2 <<<"${item}")"
  enabled="$(cut -d: -f3 <<<"${item}")"
  lambda_tag="${lambda_bg//./p}"
  exp="dewater_bg_${label,,}_${SCENE_SLUG}_seed${SEED}_step10000_to_13000"
  stamp="${STAMP_BASE}_${label,,}_l${lambda_tag}"
  config_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/config.yml"
  final_ckpt="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-000013000.ckpt"

  if [[ -f "${final_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME} ${label}] Reusing final checkpoint: ${final_ckpt}"
  else
    echo "[${SCENE_NAME} ${label}] Training medium background supervision lambda=${lambda_bg}."
    env \
      GPU="${GPU}" \
      PYTHON="${PYTHON}" \
      SCENE_SLUG="${SCENE_SLUG}" \
      DATA_PATH="${DATA_PATH}" \
      M1_LOAD_CONFIG="${M1_CONFIG}" \
      M1_LOAD_CHECKPOINT="${M1_CKPT_10K}" \
      OUTPUT_DIR="${OUTPUT_DIR}" \
      RENDER_ROOT="${RENDER_ROOT}" \
      LOG_ROOT="${LOG_ROOT}" \
      EXPERIMENT_NAME="${exp}" \
      STAMP="${stamp}" \
      TIMESTAMP="${exp}_${stamp}" \
      SEED="${SEED}" \
      LOAD_STEP="10000" \
      TARGET_FINAL_STEP="13000" \
      MAX_NUM_ITERATIONS="3000" \
      MODEL_NUM_STEPS="13000" \
      STEPS_PER_SAVE="1000" \
      SAVE_ONLY_LATEST_CHECKPOINT="False" \
      RUN_EVAL="0" \
      RUN_CLOSURE_DIAG="0" \
      GMVC_VARIANT="dewater_${label,,}" \
      GMVC_ENABLED="False" \
      GMVC_DIAGNOSTIC_ONLY="False" \
      GMVC_V2_ENABLED="False" \
      GMVC_V3_ENABLED="False" \
      LAMBDA_GMVC_PROFILE="0.0" \
      LAMBDA_GMVC_OBJECT="0.0" \
      DIRECT_OPTICAL_DEPTH_SCALE="1.0" \
      MEDIUM_BACKGROUND_SUPERVISION_ENABLED="${enabled}" \
      MEDIUM_BACKGROUND_SUPERVISION_LAMBDA="${lambda_bg}" \
      MEDIUM_BACKGROUND_SUPERVISION_EXCLUDE_BOUNDARY="True" \
      MEDIUM_BACKGROUND_SUPERVISION_HIT_EXCLUSION_THRESHOLD="-1.0" \
      BACKSCATTER_REGION_MASK_DIR="${MASK_DIR}" \
      BACKGROUND_WATER_MASK_KEY="water" \
      "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
  fi

  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    for step in 11000 12000 13000; do
      ckpt_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-$(printf "%09d" "${step}").ckpt"
      if [[ ! -f "${ckpt_path}" ]]; then
        echo "Missing checkpoint for diagnostics: ${ckpt_path}" >&2
        exit 1
      fi
      echo "[${SCENE_NAME} ${label}] Rendering eval diagnostics for step ${step}."
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
        --scene "${SCENE_NAME}" \
        --load-config "${config_path}" \
        --load-step "${step}" \
        --split eval \
        --test-mode test \
        --max-images -1 \
        --output-dir "${VIS_ROOT}/${label}/step_${step}/eval"

      echo "[${SCENE_NAME} ${label}] Rendering train-mask background diagnostics for step ${step}."
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
        --scene "${SCENE_NAME}" \
        --load-config "${config_path}" \
        --load-step "${step}" \
        --split train \
        --test-mode inference \
        --max-images -1 \
        --background-mask-dir "${MASK_DIR}" \
        --background-mask-key water \
        --stats-only \
        --output-dir "${VIS_ROOT}/${label}/step_${step}/train_background"
    done
  fi
done

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_dewater_background_supervision.py" \
    --scene "${SCENE_NAME}" \
    --input-root "${VIS_ROOT}" \
    --output-json "${SUMMARY_ROOT}/medium_background_supervision_summary.json" \
    --output-csv "${SUMMARY_ROOT}/medium_background_supervision_summary.csv"
fi
