#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-Curasao}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260807_d010_scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/dewater_d010_scratch_20260807}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
VIS_ROOT="${VIS_ROOT:-${RENDER_ROOT}/dewater_d010_scratch_20260807}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${OUTPUT_DIR}}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"

case "${SCENE}" in
  Curasao|curasao)
    SCENE_NAME="Curasao"
    SCENE_SLUG="curasao"
    DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Curasao}"
    D100_M1_CONFIG="${D100_M1_CONFIG:-${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml}"
    D100_M1_CKPT_5K="${D100_M1_CKPT_5K:-${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000005000.ckpt}"
    D100_M1_CKPT_10K="${D100_M1_CKPT_10K:-${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt}"
    D100_13K_CONFIG="${D100_13K_CONFIG:-${REPO_DIR}/outputs/dewater_direct_d100_curasao_seed42_step10000_to_13000/water-splatting/dewater_direct_d100_curasao_seed42_step10000_to_13000_20260807_dewater_direct_optical_depth_d100_g1p00/config.yml}"
    D100_PERSIST_CONFIG="${D100_PERSIST_CONFIG:-${REPO_DIR}/outputs/dewater_d010_persistence_20260807/dewater_d100_persist_curasao_seed42_step13000_to_15000/water-splatting/dewater_d100_persist_curasao_seed42_step13000_to_15000_20260807_d010_persistence_d100_persist_g1p00/config.yml}"
    ;;
  *)
    echo "This scratch run is intentionally limited to Curasao." >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}" "${VIS_ROOT}" "${LOG_ROOT}/dewater_d010_scratch_20260807"

EXP="dewater_d010_scratch_${SCENE_SLUG}_seed${SEED}_step0_to_15000"
STAMP_RUN="${STAMP_BASE}_g0p10"
CONFIG_PATH="${OUTPUT_DIR}/${EXP}/water-splatting/${EXP}_${STAMP_RUN}/config.yml"
D010_FINAL_LOAD_STEP="${D010_FINAL_LOAD_STEP:-14999}"
FINAL_CKPT="${OUTPUT_DIR}/${EXP}/water-splatting/${EXP}_${STAMP_RUN}/nerfstudio_models/step-$(printf "%09d" "${D010_FINAL_LOAD_STEP}").ckpt"

if [[ -f "${FINAL_CKPT}" && "${FORCE_RERUN}" != "1" ]]; then
  echo "[${SCENE_NAME} D010-SCRATCH] Reusing final checkpoint: ${FINAL_CKPT}"
else
  echo "[${SCENE_NAME} D010-SCRATCH] Training from scratch with gamma_D=0.10."
  env \
    GPU="${GPU}" \
    DATA_PATH="${DATA_PATH}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    RENDER_ROOT="${RENDER_ROOT}" \
    LOG_ROOT="${LOG_ROOT}" \
    EXPERIMENT_NAME="${EXP}" \
    STAMP="${STAMP_RUN}" \
    TIMESTAMP="${EXP}_${STAMP_RUN}" \
    MAX_NUM_ITERATIONS="15000" \
    MODEL_NUM_STEPS="15000" \
    STEPS_PER_SAVE="1000" \
    SAVE_ONLY_LATEST_CHECKPOINT="False" \
    RUN_EVAL="0" \
    RUN_CLOSURE_DIAG="0" \
    RUN_FAR_DIAG="0" \
    RUN_REGION_DIAG="0" \
    SEED="${SEED}" \
    MEDIUM_CONTEXT_MODE="dir_xy_camera" \
    BINF_MODE="tied" \
    BG_COLOR_WEIGHT="0.0" \
    BG_MEDIUM_RENDER_WEIGHT="0.0" \
    BG_TAIL_RENDER_WEIGHT="0.0" \
    BG_CLEAR_GAUSSIAN_WEIGHT="0.0" \
    BG_CLEAR_CHROMA_WEIGHT="0.0" \
    MEDIUM_EXPLAINABILITY_ENABLED="False" \
    LAMBDA_MEDIUM_EXPLAINABILITY="0.0" \
    TRAINING_GRADIENT_ROUTING_ENABLED="False" \
    BUDGETED_CAPACITY_ENABLED="False" \
    LAMBDA_BUDGETED_CAPACITY="0.0" \
    CORE_ZERO_CAPACITY_ENABLED="False" \
    LAMBDA_CORE_ZERO_CAPACITY="0.0" \
    CAPACITY_CONTROL_ENABLED="False" \
    HALO_CAPACITY_ENABLED="False" \
    LAMBDA_HALO_CAPACITY="0.0" \
    LAMBDA_PROXY_CLEAR_LUMA="0.0" \
    OBJECT_RADIANCE_BUDGET_ENABLED="False" \
    LAMBDA_OBJECT_RADIANCE_BUDGET="0.0" \
    CLEAR_PROXY_ENABLED="False" \
    BACKGROUND_GRADIENT_SURGERY_ENABLED="False" \
    BACKGROUND_DENSIFICATION_ENABLED="False" \
    BACKGROUND_DENSIFICATION_DIAGNOSTIC_ONLY="True" \
    OPACITY_ACCUMULATION_DIAGNOSTIC_ENABLED="False" \
    FG_TRANS_WEIGHT="0.0" \
    MEDIUM_PREDICTOR_MODE="single" \
    LAMBDA_PSEUDO_DEPTH="0.0" \
    LAMBDA_MEDIUM_CONTEXT_RESIDUAL="0.0" \
    DIRECT_OPTICAL_DEPTH_SCALE="0.10" \
    MEDIUM_BACKGROUND_SUPERVISION_ENABLED="False" \
    MEDIUM_BACKGROUND_SUPERVISION_LAMBDA="0.0" \
    GMVC_ENABLED="False" \
    GMVC_DIAGNOSTIC_ONLY="False" \
    GMVC_V2_ENABLED="False" \
    GMVC_V3_ENABLED="False" \
    LAMBDA_GMVC_PROFILE="0.0" \
    LAMBDA_GMVC_OBJECT="0.0" \
    "${REPO_DIR}/scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh"
fi

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  # Nerfstudio from-scratch runs save the terminal 15k checkpoint as step 14999.
  # Keep the output label as step_15000 and preserve loaded_step=14999 in summary.json.
  d010_step_items=(
    "1000:1000"
    "3000:3000"
    "5000:5000"
    "8000:8000"
    "10000:10000"
    "13000:13000"
    "15000:${D010_FINAL_LOAD_STEP}"
  )
  for item in "${d010_step_items[@]}"; do
    step_label="$(cut -d: -f1 <<<"${item}")"
    load_step="$(cut -d: -f2 <<<"${item}")"
    ckpt_path="${OUTPUT_DIR}/${EXP}/water-splatting/${EXP}_${STAMP_RUN}/nerfstudio_models/step-$(printf "%09d" "${load_step}").ckpt"
    if [[ ! -f "${ckpt_path}" ]]; then
      echo "Missing D010-SCRATCH checkpoint for diagnostics: ${ckpt_path}" >&2
      exit 1
    fi
    summary_path="${VIS_ROOT}/D010-SCRATCH/step_${step_label}/summary.json"
    if [[ -f "${summary_path}" && "${FORCE_RERUN}" != "1" ]]; then
      echo "[${SCENE_NAME} D010-SCRATCH] Reusing diagnostics for nominal step ${step_label}: ${summary_path}"
      continue
    fi
    echo "[${SCENE_NAME} D010-SCRATCH] Rendering diagnostics for nominal step ${step_label} from checkpoint step ${load_step}."
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
      --scene "${SCENE_NAME}" \
      --load-config "${CONFIG_PATH}" \
      --load-step "${load_step}" \
      --split eval \
      --test-mode test \
      --max-images -1 \
      --output-dir "${VIS_ROOT}/D010-SCRATCH/step_${step_label}"
  done

  # D100-SCRATCH uses existing gamma=1 Curasao M1-compatible checkpoints where available.
  baseline_items=(
    "5000:5000:${D100_M1_CONFIG}"
    "10000:10000:${D100_M1_CONFIG}"
    "13000:13000:${D100_13K_CONFIG}"
    "15000:14999:${D100_M1_CONFIG}"
  )
  for item in "${baseline_items[@]}"; do
    step_label="$(cut -d: -f1 <<<"${item}")"
    load_step="$(cut -d: -f2 <<<"${item}")"
    config="$(cut -d: -f3- <<<"${item}")"
    if [[ ! -f "${config}" ]]; then
      echo "Missing D100 baseline config for step ${step_label}: ${config}" >&2
      exit 1
    fi
    summary_path="${VIS_ROOT}/D100-SCRATCH/step_${step_label}/summary.json"
    if [[ -f "${summary_path}" && "${FORCE_RERUN}" != "1" ]]; then
      echo "[${SCENE_NAME} D100-SCRATCH] Reusing baseline diagnostics for nominal step ${step_label}: ${summary_path}"
      continue
    fi
    echo "[${SCENE_NAME} D100-SCRATCH] Rendering baseline diagnostics for nominal step ${step_label} from checkpoint step ${load_step}."
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
      --scene "${SCENE_NAME}" \
      --load-config "${config}" \
      --load-step "${load_step}" \
      --split eval \
      --test-mode test \
      --max-images -1 \
      --output-dir "${VIS_ROOT}/D100-SCRATCH/step_${step_label}"
  done

  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_dewater_d010_scratch.py" \
    --scene "${SCENE_NAME}" \
    --scratch-root "${VIS_ROOT}" \
    --persistence-summary "${REPO_DIR}/outputs/dewater_d010_persistence_20260807/d010_persistence_summary.json" \
    --train-log "${LOG_ROOT}/${EXP}_${STAMP_RUN}/train.log" \
    --output-json "${SUMMARY_ROOT}/d010_scratch_summary.json" \
    --output-csv "${SUMMARY_ROOT}/d010_scratch_summary.csv"

  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/render_dewater_d010_three_path_comparison.py" \
    --scene "${SCENE_NAME}" \
    --d100-root "${VIS_ROOT}/D100-SCRATCH/step_15000" \
    --switch-root "${RENDER_ROOT}/dewater_d010_persistence_20260807/D010-PERSIST/step_15000" \
    --scratch-root "${VIS_ROOT}/D010-SCRATCH/step_15000" \
    --output-dir "${VIS_ROOT}/three_path_step_15000"
fi
