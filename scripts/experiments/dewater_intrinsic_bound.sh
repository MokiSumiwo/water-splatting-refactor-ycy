#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-Curasao}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260808_intrinsic_bound}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/dewater_seafree_factor_20260808}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
VIS_ROOT="${VIS_ROOT:-${RENDER_ROOT}/dewater_seafree_factor_20260808/intrinsic_bound}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
GRAD_AUDIT_JSON="${GRAD_AUDIT_JSON:-${OUTPUT_DIR}/loss_gradient_audit.json}"

case "${SCENE}" in
  Curasao|curasao)
    SCENE_NAME="Curasao"
    SCENE_SLUG="curasao"
    DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_Curasao}"
    D010_CONFIG="${D010_CONFIG:-${REPO_DIR}/outputs/dewater_direct_d010_curasao_seed42_step10000_to_13000/water-splatting/dewater_direct_d010_curasao_seed42_step10000_to_13000_20260807_dewater_direct_optical_depth_d010_g0p10/config.yml}"
    D010_CKPT_13K="${D010_CKPT_13K:-${REPO_DIR}/outputs/dewater_direct_d010_curasao_seed42_step10000_to_13000/water-splatting/dewater_direct_d010_curasao_seed42_step10000_to_13000_20260807_dewater_direct_optical_depth_d010_g0p10/nerfstudio_models/step-000013000.ckpt}"
    D010_SWITCH_SUMMARY="${D010_SWITCH_SUMMARY:-${REPO_DIR}/renders/dewater_d010_persistence_20260807/D010-PERSIST/step_15000/summary.json}"
    ;;
  *)
    echo "This intrinsic-bound experiment is intentionally limited to Curasao." >&2
    exit 2
    ;;
esac

for required in "${D010_CONFIG}" "${D010_CKPT_13K}" "${GRAD_AUDIT_JSON}" "${D010_SWITCH_SUMMARY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required intrinsic-bound input: ${required}" >&2
    exit 1
  fi
done

read_lambda() {
  local key="$1"
  "${PYTHON}" - "${GRAD_AUDIT_JSON}" "${key}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf8").read())
print(payload["audit"]["recommended_lambdas"][sys.argv[2]])
PY
}

IB_G01_LAMBDA="${IB_G01_LAMBDA:-$(read_lambda IB-G01)}"
IB_G05_LAMBDA="${IB_G05_LAMBDA:-$(read_lambda IB-G05)}"
IB_G10_LAMBDA="${IB_G10_LAMBDA:-$(read_lambda IB-G10)}"

mkdir -p "${OUTPUT_DIR}" "${VIS_ROOT}" "${LOG_ROOT}/dewater_seafree_factor_20260808"

runs=(
  "IB-G01:${IB_G01_LAMBDA}:ib_g01"
  "IB-G05:${IB_G05_LAMBDA}:ib_g05"
  "IB-G10:${IB_G10_LAMBDA}:ib_g10"
)

summary_args=(--run "D010-SWITCH=${D010_SWITCH_SUMMARY}")

for item in "${runs[@]}"; do
  label="$(cut -d: -f1 <<<"${item}")"
  lambda="$(cut -d: -f2 <<<"${item}")"
  slug="$(cut -d: -f3 <<<"${item}")"
  lambda_tag="$(printf "%.6f" "${lambda}" | sed 's/\\./p/g')"
  exp="dewater_${slug}_${SCENE_SLUG}_seed${SEED}_step13000_to_15000"
  stamp="${STAMP_BASE}_${slug}_l${lambda_tag}"
  config_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/config.yml"
  final_ckpt="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-000015000.ckpt"

  if [[ -f "${final_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME} ${label}] Reusing final checkpoint: ${final_ckpt}"
  else
    echo "[${SCENE_NAME} ${label}] Continuing D010 step 13000 -> 15000 with intrinsic_bound_lambda=${lambda}."
    env \
      GPU="${GPU}" \
      PYTHON="${PYTHON}" \
      SCENE_SLUG="${SCENE_SLUG}" \
      DATA_PATH="${DATA_PATH}" \
      M1_LOAD_CONFIG="${D010_CONFIG}" \
      M1_LOAD_CHECKPOINT="${D010_CKPT_13K}" \
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
      DIRECT_OPTICAL_DEPTH_SCALE="0.10" \
      DISABLE_POPULATION_REFINEMENT="False" \
      INTRINSIC_BOUND_LAMBDA="${lambda}" \
      INTRINSIC_BOUND_VISIBLE_ONLY="True" \
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
  summary_args+=(--run "${label}=${VIS_ROOT}/${label}/step_15000/summary.json")
done

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_dewater_seafree_factors.py" \
    "${summary_args[@]}" \
    --compare "intrinsic_bound:D010-SWITCH:IB-G01" \
    --compare "intrinsic_bound:D010-SWITCH:IB-G05" \
    --compare "intrinsic_bound:D010-SWITCH:IB-G10" \
    --output-json "${OUTPUT_DIR}/intrinsic_bound_summary.json" \
    --output-csv "${OUTPUT_DIR}/intrinsic_bound_summary.csv"
fi
