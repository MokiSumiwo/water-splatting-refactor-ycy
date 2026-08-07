#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-Curasao}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260807_dewater_direct_optical_depth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"

case "${SCENE}" in
  Curasao|curasao)
    SCENE_NAME="Curasao"
    SCENE_SLUG="curasao"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Curasao"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    ;;
  JapaneseGradens|japanesegradens|japanesegradens_redsea)
    SCENE_NAME="JapaneseGradens"
    SCENE_SLUG="japanesegradens_redsea"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    ;;
  IUI3|iui3|iui3_redsea)
    SCENE_NAME="IUI3"
    SCENE_SLUG="iui3_redsea"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/nerfstudio_models/step-000010000.ckpt"
    ;;
  Panama|panama)
    SCENE_NAME="Panama"
    SCENE_SLUG="panama"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Panama"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    ;;
  *)
    echo "Unknown SCENE=${SCENE}. Use Curasao, JapaneseGradens, IUI3, or Panama." >&2
    exit 2
    ;;
esac

DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}"
M1_CONFIG="${M1_CONFIG:-${M1_CONFIG_DEFAULT}}"
M1_CKPT_10K="${M1_CKPT_10K:-${M1_CKPT_DEFAULT}}"
VIS_ROOT="${VIS_ROOT:-${RENDER_ROOT}/dewater_optical_depth_20260807/A}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${OUTPUT_DIR}/dewater_optical_depth_20260807}"

if [[ ! -f "${M1_CONFIG}" ]]; then
  echo "Missing M1 config: ${M1_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${M1_CKPT_10K}" ]]; then
  echo "Missing M1 step-10000 checkpoint: ${M1_CKPT_10K}" >&2
  exit 1
fi

mkdir -p "${VIS_ROOT}" "${SUMMARY_ROOT}"

runs=(
  "D100:1.00"
  "D050:0.50"
  "D025:0.25"
  "D010:0.10"
)

for item in "${runs[@]}"; do
  label="${item%%:*}"
  gamma="${item##*:}"
  gamma_tag="${gamma//./p}"
  exp="dewater_direct_${label,,}_${SCENE_SLUG}_seed${SEED}_step10000_to_13000"
  stamp="${STAMP_BASE}_${label,,}_g${gamma_tag}"
  config_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/config.yml"
  final_ckpt="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-000013000.ckpt"

  if [[ -f "${final_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME} ${label}] Reusing final checkpoint: ${final_ckpt}"
  else
    echo "[${SCENE_NAME} ${label}] Training gamma_D=${gamma} from M1 step 10000 to 13000."
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
      DIRECT_OPTICAL_DEPTH_SCALE="${gamma}" \
      "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
  fi

  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    for step in 11000 12000 13000; do
      ckpt_path="${OUTPUT_DIR}/${exp}/water-splatting/${exp}_${stamp}/nerfstudio_models/step-$(printf "%09d" "${step}").ckpt"
      if [[ ! -f "${ckpt_path}" ]]; then
        echo "Missing checkpoint for diagnostics: ${ckpt_path}" >&2
        exit 1
      fi
      out_dir="${VIS_ROOT}/${label}/step_${step}"
      echo "[${SCENE_NAME} ${label}] Rendering diagnostics for step ${step}."
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_dewater_optical_depth.py" \
        --scene "${SCENE_NAME}" \
        --load-config "${config_path}" \
        --load-step "${step}" \
        --test-mode test \
        --max-images -1 \
        --output-dir "${out_dir}"
    done
  fi
done

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_dewater_optical_depth_sweep.py" \
    --scene "${SCENE_NAME}" \
    --input-root "${VIS_ROOT}" \
    --output-json "${SUMMARY_ROOT}/direct_optical_depth_sweep_summary.json" \
    --output-csv "${SUMMARY_ROOT}/direct_optical_depth_sweep_summary.csv"
fi
