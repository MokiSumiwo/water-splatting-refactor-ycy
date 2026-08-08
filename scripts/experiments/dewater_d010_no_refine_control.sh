#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-Curasao}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260808_no_refine}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/dewater_seafree_factor_20260808}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
VIS_ROOT="${VIS_ROOT:-${RENDER_ROOT}/dewater_seafree_factor_20260808/no_refine}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"

case "${SCENE}" in
  Curasao|curasao)
    SCENE_NAME="Curasao"
    SCENE_SLUG="curasao"
    DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Curasao}"
    M1_CONFIG="${M1_CONFIG:-${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml}"
    M1_CKPT_10K="${M1_CKPT_10K:-${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt}"
    ;;
  *)
    echo "This no-refine control is intentionally limited to Curasao." >&2
    exit 2
    ;;
esac

for required in "${M1_CONFIG}" "${M1_CKPT_10K}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required no-refine input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}" "${VIS_ROOT}" "${LOG_ROOT}/dewater_seafree_factor_20260808"

runs=(
  "NR-D100:1.00:nr_d100"
  "NR-D010:0.10:nr_d010"
)

for item in "${runs[@]}"; do
  label="$(cut -d: -f1 <<<"${item}")"
  gamma="$(cut -d: -f2 <<<"${item}")"
  slug="$(cut -d: -f3 <<<"${item}")"
  gamma_tag="${gamma//./p}"
  exp="dewater_${slug}_${SCENE_SLUG}_seed${SEED}_step10000_to_15000"
  stamp="${STAMP_BASE}_${slug}_g${gamma_tag}"
  config_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/config.yml"
  final_ckpt="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-000015000.ckpt"

  if [[ -f "${final_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME} ${label}] Reusing final checkpoint: ${final_ckpt}"
  else
    echo "[${SCENE_NAME} ${label}] Continuing M1 step 10000 -> 15000 with fixed Gaussian population and gamma_D=${gamma}."
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
      TARGET_FINAL_STEP="15000" \
      MAX_NUM_ITERATIONS="5000" \
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
      DISABLE_POPULATION_REFINEMENT="True" \
      INTRINSIC_BOUND_LAMBDA="0.0" \
      FOREGROUND_AWARE_WEIGHTING_ENABLED="False" \
      FOREGROUND_AWARE_WEIGHTING_LAMBDA="0.0" \
      MEDIUM_BACKGROUND_SUPERVISION_ENABLED="False" \
      MEDIUM_BACKGROUND_SUPERVISION_LAMBDA="0.0" \
      "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
  fi

  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    for step in 14000 15000; do
      summary_path="${VIS_ROOT}/${label}/step_${step}/summary.json"
      if [[ -f "${summary_path}" && "${FORCE_RERUN}" != "1" ]]; then
        echo "[${SCENE_NAME} ${label}] Reusing diagnostics for step ${step}: ${summary_path}"
        continue
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
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_dewater_seafree_factors.py" \
    --run "NR-D100=${VIS_ROOT}/NR-D100/step_15000/summary.json" \
    --run "NR-D010=${VIS_ROOT}/NR-D010/step_15000/summary.json" \
    --compare "no_refine:NR-D100:NR-D010" \
    --output-json "${OUTPUT_DIR}/no_refine_summary.json" \
    --output-csv "${OUTPUT_DIR}/no_refine_summary.csv"
fi
