#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-Curasao}"
VARIANT="${VARIANT:-ALL}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260806_gmvc_four_scene_p30_mhold_15k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
RUN_EVAL="${RUN_EVAL:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

case "${SCENE}" in
  Curasao|curasao)
    SCENE_NAME="Curasao"
    SCENE_SLUG="curasao"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Curasao"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    EXISTING_A0_CONFIG="${REPO_DIR}/outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml"
    EXISTING_P30_13K_CKPT="${REPO_DIR}/outputs/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000/water-splatting/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000_20260805_gmvc_v3_curasao_r500_profile_persistence_3k_p30_p30_r500_g000/nerfstudio_models/step-000013000.ckpt"
    EXISTING_P30_MHOLD_CONFIG="${REPO_DIR}/outputs/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_mhold/config.yml"
    ;;
  JapaneseGradens|japanesegradens|japanesegradens_redsea)
    SCENE_NAME="JapaneseGradens"
    SCENE_SLUG="japanesegradens_redsea"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    EXISTING_A0_CONFIG="${REPO_DIR}/outputs/gmvc_v3_japanesegradens_p30_release_a0_seed42_step13000_to_15000/water-splatting/gmvc_v3_japanesegradens_p30_release_a0_seed42_step13000_to_15000_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k_a0/config.yml"
    EXISTING_P30_13K_CKPT="${REPO_DIR}/outputs/gmvc_v3_r500_p30_profile_persistence3k_japanesegradens_redsea_seed42_step10000_to_13000/water-splatting/gmvc_v3_r500_p30_profile_persistence3k_japanesegradens_redsea_seed42_step10000_to_13000_20260806_gmvc_v3_japanesegradens_r500_profile_persistence_3k_p30/nerfstudio_models/step-000013000.ckpt"
    EXISTING_P30_MHOLD_CONFIG="${REPO_DIR}/outputs/gmvc_v3_japanesegradens_p30_release_mhold_seed42_step13000_to_15000/water-splatting/gmvc_v3_japanesegradens_p30_release_mhold_seed42_step13000_to_15000_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k_mhold/config.yml"
    ;;
  IUI3|iui3|iui3_redsea)
    SCENE_NAME="IUI3"
    SCENE_SLUG="iui3_redsea"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea"
    M1_BOOTSTRAP_EXP="gmvc_v3_four_scene_iui3_m1_seed${SEED}_15000"
    M1_BOOTSTRAP_STAMP="${STAMP_BASE}_m1_bootstrap"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/${M1_BOOTSTRAP_EXP}/water-splatting/${M1_BOOTSTRAP_EXP}_${M1_BOOTSTRAP_STAMP}/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/${M1_BOOTSTRAP_EXP}/water-splatting/${M1_BOOTSTRAP_EXP}_${M1_BOOTSTRAP_STAMP}/nerfstudio_models/step-000010000.ckpt"
    EXISTING_A0_CONFIG=""
    EXISTING_P30_13K_CKPT=""
    EXISTING_P30_MHOLD_CONFIG=""
    ;;
  Panama|panama)
    SCENE_NAME="Panama"
    SCENE_SLUG="panama"
    DATA_PATH_DEFAULT="${REPO_DIR}/undistorted_data/undistorted_Panama"
    M1_CONFIG_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
    M1_CKPT_DEFAULT="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt"
    EXISTING_A0_CONFIG=""
    EXISTING_P30_13K_CKPT=""
    EXISTING_P30_MHOLD_CONFIG=""
    ;;
  *)
    echo "Unknown SCENE=${SCENE}. Use Curasao, JapaneseGradens, IUI3, or Panama." >&2
    exit 2
    ;;
esac

DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}"
M1_CONFIG="${M1_CONFIG:-${M1_CONFIG_DEFAULT}}"
M1_CKPT_10K="${M1_CKPT_10K:-${M1_CKPT_DEFAULT}}"
TRAIN_BANK="${TRAIN_BANK:-${RENDER_ROOT}/gmvc_v3_geometry_track_banks/${SCENE_SLUG}_m1_step10000_train_s4096/gmvc_track_bank.pt}"

A0_EXP="gmvc_v3_four_scene_a0_${SCENE_SLUG}_seed${SEED}_step10000_to_15000"
A0_STAMP="${STAMP_BASE}_a0"
A0_CONFIG="${A0_CONFIG:-${OUTPUT_DIR}/${A0_EXP}/water-splatting/${A0_EXP}_${A0_STAMP}/config.yml}"
A0_15K_CKPT="${A0_15K_CKPT:-${OUTPUT_DIR}/${A0_EXP}/water-splatting/${A0_EXP}_${A0_STAMP}/nerfstudio_models/step-000015000.ckpt}"

P30_EXP="gmvc_v3_four_scene_p30_profile_${SCENE_SLUG}_seed${SEED}_step10000_to_13000"
P30_STAMP="${STAMP_BASE}_profile"
P30_13K_CKPT="${P30_13K_CKPT:-${OUTPUT_DIR}/${P30_EXP}/water-splatting/${P30_EXP}_${P30_STAMP}/nerfstudio_models/step-000013000.ckpt}"

MHOLD_EXP="gmvc_v3_four_scene_p30_mhold_${SCENE_SLUG}_seed${SEED}_step13000_to_15000"
MHOLD_STAMP="${STAMP_BASE}_mhold"
MHOLD_CONFIG="${MHOLD_CONFIG:-${OUTPUT_DIR}/${MHOLD_EXP}/water-splatting/${MHOLD_EXP}_${MHOLD_STAMP}/config.yml}"
MHOLD_15K_CKPT="${MHOLD_15K_CKPT:-${OUTPUT_DIR}/${MHOLD_EXP}/water-splatting/${MHOLD_EXP}_${MHOLD_STAMP}/nerfstudio_models/step-000015000.ckpt}"
MHOLD_LOG="${LOG_ROOT}/gmvc_v3_four_scene_p30_mhold_${SCENE_SLUG}_${STAMP_BASE}.jsonl"

if [[ -n "${EXISTING_A0_CONFIG}" && -f "${EXISTING_A0_CONFIG}" && -f "${EXISTING_A0_CONFIG%/config.yml}/nerfstudio_models/step-000015000.ckpt" && "${FORCE_RERUN}" != "1" ]]; then
  A0_CONFIG="${EXISTING_A0_CONFIG}"
  A0_15K_CKPT="${EXISTING_A0_CONFIG%/config.yml}/nerfstudio_models/step-000015000.ckpt"
fi
if [[ -n "${EXISTING_P30_13K_CKPT}" && -f "${EXISTING_P30_13K_CKPT}" && "${FORCE_RERUN}" != "1" ]]; then
  P30_13K_CKPT="${EXISTING_P30_13K_CKPT}"
fi
if [[ -n "${EXISTING_P30_MHOLD_CONFIG}" && -f "${EXISTING_P30_MHOLD_CONFIG}" && -f "${EXISTING_P30_MHOLD_CONFIG%/config.yml}/nerfstudio_models/step-000015000.ckpt" && "${FORCE_RERUN}" != "1" ]]; then
  MHOLD_CONFIG="${EXISTING_P30_MHOLD_CONFIG}"
  MHOLD_15K_CKPT="${EXISTING_P30_MHOLD_CONFIG%/config.yml}/nerfstudio_models/step-000015000.ckpt"
fi

run_m1_bootstrap() {
  if [[ -f "${M1_CKPT_10K}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME}] Reusing M1 step-10000 checkpoint: ${M1_CKPT_10K}"
    return
  fi
  if [[ "${SCENE_NAME}" != "IUI3" ]]; then
    echo "[${SCENE_NAME}] Missing M1 step-10000 checkpoint: ${M1_CKPT_10K}" >&2
    exit 1
  fi
  echo "[${SCENE_NAME}] Training M1 bootstrap to create step-10000 and A0 source checkpoints."
  env \
    GPU="${GPU}" \
    PYTHON="${PYTHON}" \
    SCENE_SLUG="${SCENE_SLUG}" \
    DATA_PATH="${DATA_PATH}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    RENDER_ROOT="${RENDER_ROOT}" \
    LOG_ROOT="${LOG_ROOT}" \
    EXPERIMENT_NAME="${M1_BOOTSTRAP_EXP}" \
    STAMP="${M1_BOOTSTRAP_STAMP}" \
    TIMESTAMP="${M1_BOOTSTRAP_EXP}_${M1_BOOTSTRAP_STAMP}" \
    SEED="${SEED}" \
    MAX_NUM_ITERATIONS="15000" \
    MODEL_NUM_STEPS="15000" \
    STEPS_PER_SAVE="5000" \
    SAVE_ONLY_LATEST_CHECKPOINT="False" \
    RUN_EVAL="0" \
    RUN_CLOSURE_DIAG="0" \
    BUILD_MASKS="0" \
    RUN_POST_MASK_DIAGS="0" \
    "${REPO_DIR}/scripts/experiments/cross_scene_m1_common.sh"
}

run_a0() {
  if [[ -f "${A0_CONFIG}" && -f "${A0_15K_CKPT}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME}] Reusing A0 config: ${A0_CONFIG}"
    return
  fi
  if [[ ! -f "${M1_CKPT_10K}" ]]; then
    echo "[${SCENE_NAME}] Missing M1 source checkpoint for A0: ${M1_CKPT_10K}" >&2
    exit 1
  fi
  echo "[${SCENE_NAME}] Running A0 M1 continuation from step 10000 to 15000."
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
    EXPERIMENT_NAME="${A0_EXP}" \
    STAMP="${A0_STAMP}" \
    TIMESTAMP="${A0_EXP}_${A0_STAMP}" \
    SEED="${SEED}" \
    LOAD_STEP="10000" \
    TARGET_FINAL_STEP="15000" \
    MAX_NUM_ITERATIONS="5000" \
    MODEL_NUM_STEPS="15000" \
    STEPS_PER_SAVE="1000" \
    SAVE_ONLY_LATEST_CHECKPOINT="False" \
    RUN_EVAL="${RUN_EVAL}" \
    RUN_CLOSURE_DIAG="0" \
    GMVC_VARIANT="four_scene_a0" \
    GMVC_ENABLED="False" \
    GMVC_DIAGNOSTIC_ONLY="False" \
    GMVC_V2_ENABLED="False" \
    GMVC_V3_ENABLED="False" \
    LAMBDA_GMVC_PROFILE="0.0" \
    LAMBDA_GMVC_OBJECT="0.0" \
    GMVC_PROFILE_OBSERVABILITY_WEIGHT_ENABLED="False" \
    "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
}

run_p30_profile() {
  if [[ -f "${P30_13K_CKPT}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME}] Reusing P30 profile checkpoint: ${P30_13K_CKPT}"
    return
  fi
  if [[ ! -f "${M1_CKPT_10K}" ]]; then
    echo "[${SCENE_NAME}] Missing M1 source checkpoint for P30 profile: ${M1_CKPT_10K}" >&2
    exit 1
  fi
  echo "[${SCENE_NAME}] Running P30 calibration from step 10000 to 13000."
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
    EXPERIMENT_NAME="${P30_EXP}" \
    STAMP="${P30_STAMP}" \
    TIMESTAMP="${P30_EXP}_${P30_STAMP}" \
    SEED="${SEED}" \
    LOAD_STEP="10000" \
    TARGET_FINAL_STEP="13000" \
    MAX_NUM_ITERATIONS="3000" \
    MODEL_NUM_STEPS="13000" \
    STEPS_PER_SAVE="1000" \
    SAVE_ONLY_LATEST_CHECKPOINT="False" \
    RUN_EVAL="0" \
    RUN_CLOSURE_DIAG="0" \
    GMVC_VARIANT="four_scene_p30_profile" \
    GMVC_TRACK_BANK_PATH="${TRAIN_BANK}" \
    GMVC_NEEDS_BANK="1" \
    GMVC_TRACK_GEOMETRY_ONLY="1" \
    GMVC_TRACK_SAMPLES_PER_VIEW="4096" \
    GMVC_TRACK_MAX_OBS_PER_CAMERA="20000" \
    GMVC_TRACK_SIGNAL_MIN="0.02" \
    GMVC_TRACK_SIGNAL_MAX="0.98" \
    GMVC_ENABLED="True" \
    GMVC_DIAGNOSTIC_ONLY="False" \
    GMVC_V2_ENABLED="True" \
    GMVC_V3_ENABLED="True" \
    GMVC_START_STEP="10000" \
    GMVC_STOP_STEP="13000" \
    GMVC_RAMP_STEPS="500" \
    LAMBDA_GMVC_PROFILE="30" \
    LAMBDA_GMVC_OBJECT="0.004" \
    GMVC_PROFILE_LOSS_MODE="irls_l2" \
    GMVC_PROFILE_TRACK_BALANCED="True" \
    GMVC_PROFILE_IRLS_DELTA="0.03" \
    GMVC_PROFILE_IRLS_MAX_WEIGHT="1.0" \
    GMVC_PROFILE_MIN_HESSIAN="1e-5" \
    GMVC_PROFILE_MIN_TRANSMISSION_SPAN="0.01" \
    GMVC_PROFILE_MIN_DEPTH_SPAN_REL="0.05" \
    GMVC_PROFILE_OBSERVABILITY_WEIGHT_ENABLED="False" \
    GMVC_V3_PROFILE_SCHEDULE="constant" \
    GMVC_MEDIUM_HOLD_ENABLED="False" \
    GMVC_V3_MEDIUM_STEPS="4" \
    GMVC_V3_OBJECT_STEPS="1" \
    GMVC_V3_OBJECT_PHASE_MEDIUM_GRAD_SCALE="0.00" \
    GMVC_V3_TARGET_CURRENT_CAMERA_TRACKS="True" \
    GMVC_V3_OBJECT_SOURCE="J_proxy_raw" \
    GMVC_OBJECT_TRACK_BALANCED="True" \
    GMVC_OBJECT_J_CLAMP_MIN="-0.1" \
    GMVC_OBJECT_J_CLAMP_MAX="1.1" \
    GMVC_OBJECT_MIN_HESSIAN="1e-5" \
    GMVC_OBJECT_MIN_DEPTH_SPAN_REL="0.05" \
    GMVC_V2_MAX_TRACKS_PER_STEP="512" \
    GMVC_V2_MIN_OBSERVATIONS_PER_TRACK="2" \
    GMVC_GRAD_LOG_PATH="${LOG_ROOT}/gmvc_v3_four_scene_p30_profile_${SCENE_SLUG}_${STAMP_BASE}.jsonl" \
    GMVC_GRAD_LOG_EVERY="49" \
    GMVC_GRAD_LOG_FORCE_STEPS="10000,10001,10500,11000,12000,13000" \
    GMVC_MAX_TRACKS_PER_STEP="4096" \
    GMVC_BOUNDED_MEDIUM_ENABLED="False" \
    GMVC_BOUNDED_INIT_FROM_FIRST_BATCH="False" \
    GMVC_CLOSURE_SIGNAL_FLOOR="0.03" \
    GMVC_INTRINSIC_SOURCE="J_proxy_raw" \
    GMVC_INTRINSIC_USE_DC_PROXY="True" \
    "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
}

run_p30_mhold() {
  if [[ -f "${MHOLD_CONFIG}" && -f "${MHOLD_15K_CKPT}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[${SCENE_NAME}] Reusing P30-MHOLD config: ${MHOLD_CONFIG}"
    return
  fi
  if [[ ! -f "${P30_13K_CKPT}" ]]; then
    echo "[${SCENE_NAME}] Missing P30 source checkpoint for MHOLD: ${P30_13K_CKPT}" >&2
    exit 1
  fi
  echo "[${SCENE_NAME}] Running P30-MHOLD from step 13000 to 15000."
  env \
    GPU="${GPU}" \
    PYTHON="${PYTHON}" \
    SCENE_SLUG="${SCENE_SLUG}" \
    DATA_PATH="${DATA_PATH}" \
    M1_LOAD_CONFIG="${M1_CONFIG}" \
    M1_LOAD_CHECKPOINT="${P30_13K_CKPT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    RENDER_ROOT="${RENDER_ROOT}" \
    LOG_ROOT="${LOG_ROOT}" \
    EXPERIMENT_NAME="${MHOLD_EXP}" \
    STAMP="${MHOLD_STAMP}" \
    TIMESTAMP="${MHOLD_EXP}_${MHOLD_STAMP}" \
    SEED="${SEED}" \
    LOAD_STEP="13000" \
    TARGET_FINAL_STEP="15000" \
    MAX_NUM_ITERATIONS="2000" \
    MODEL_NUM_STEPS="15000" \
    STEPS_PER_SAVE="500" \
    SAVE_ONLY_LATEST_CHECKPOINT="False" \
    RUN_EVAL="${RUN_EVAL}" \
    RUN_CLOSURE_DIAG="0" \
    GMVC_VARIANT="four_scene_p30_mhold" \
    GMVC_TRACK_BANK_PATH="${TRAIN_BANK}" \
    GMVC_NEEDS_BANK="0" \
    GMVC_TRACK_GEOMETRY_ONLY="1" \
    GMVC_ENABLED="True" \
    GMVC_DIAGNOSTIC_ONLY="False" \
    GMVC_V2_ENABLED="True" \
    GMVC_V3_ENABLED="True" \
    GMVC_START_STEP="10000" \
    GMVC_STOP_STEP="15000" \
    GMVC_RAMP_STEPS="500" \
    LAMBDA_GMVC_PROFILE="30" \
    LAMBDA_GMVC_OBJECT="0.004" \
    GMVC_PROFILE_LOSS_MODE="irls_l2" \
    GMVC_PROFILE_TRACK_BALANCED="True" \
    GMVC_PROFILE_IRLS_DELTA="0.03" \
    GMVC_PROFILE_IRLS_MAX_WEIGHT="1.0" \
    GMVC_PROFILE_MIN_HESSIAN="1e-5" \
    GMVC_PROFILE_MIN_TRANSMISSION_SPAN="0.01" \
    GMVC_PROFILE_MIN_DEPTH_SPAN_REL="0.05" \
    GMVC_PROFILE_OBSERVABILITY_WEIGHT_ENABLED="False" \
    GMVC_V3_PROFILE_SCHEDULE="stop" \
    GMVC_V3_PROFILE_DECAY_START_STEP="13000" \
    GMVC_V3_PROFILE_DECAY_END_STEP="14000" \
    GMVC_V3_PROFILE_DECAY_FINAL_SCALE="0.0" \
    GMVC_MEDIUM_HOLD_ENABLED="True" \
    GMVC_MEDIUM_HOLD_START_STEP="13001" \
    GMVC_MEDIUM_HOLD_STOP_STEP="15000" \
    GMVC_V3_MEDIUM_STEPS="4" \
    GMVC_V3_OBJECT_STEPS="1" \
    GMVC_V3_OBJECT_PHASE_MEDIUM_GRAD_SCALE="0.00" \
    GMVC_V3_TARGET_CURRENT_CAMERA_TRACKS="True" \
    GMVC_V3_OBJECT_SOURCE="J_proxy_raw" \
    GMVC_OBJECT_TRACK_BALANCED="True" \
    GMVC_OBJECT_J_CLAMP_MIN="-0.1" \
    GMVC_OBJECT_J_CLAMP_MAX="1.1" \
    GMVC_OBJECT_MIN_HESSIAN="1e-5" \
    GMVC_OBJECT_MIN_DEPTH_SPAN_REL="0.05" \
    GMVC_V2_MAX_TRACKS_PER_STEP="512" \
    GMVC_V2_MIN_OBSERVATIONS_PER_TRACK="2" \
    GMVC_GRAD_LOG_PATH="${MHOLD_LOG}" \
    GMVC_GRAD_LOG_EVERY="49" \
    GMVC_GRAD_LOG_FORCE_STEPS="13001,13004,13500,14000,15000" \
    GMVC_MAX_TRACKS_PER_STEP="4096" \
    GMVC_BOUNDED_MEDIUM_ENABLED="False" \
    GMVC_BOUNDED_INIT_FROM_FIRST_BATCH="False" \
    GMVC_CLOSURE_SIGNAL_FLOOR="0.03" \
    GMVC_INTRINSIC_SOURCE="J_proxy_raw" \
    GMVC_INTRINSIC_USE_DC_PROXY="True" \
    "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
}

case "${VARIANT}" in
  M1_BOOTSTRAP|m1_bootstrap)
    run_m1_bootstrap
    ;;
  A0|a0)
    run_m1_bootstrap
    run_a0
    ;;
  P30_PROFILE|p30_profile|PROFILE|profile)
    run_m1_bootstrap
    run_p30_profile
    ;;
  P30_MHOLD|p30_mhold|MHOLD|mhold)
    run_m1_bootstrap
    run_p30_profile
    run_p30_mhold
    ;;
  ALL|all)
    run_m1_bootstrap
    run_a0
    run_p30_profile
    run_p30_mhold
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use M1_BOOTSTRAP, A0, P30_PROFILE, P30_MHOLD, or ALL." >&2
    exit 2
    ;;
esac

echo "[${SCENE_NAME}] done"
echo "A0_CONFIG=${A0_CONFIG}"
echo "A0_15K_CKPT=${A0_15K_CKPT}"
echo "P30_13K_CKPT=${P30_13K_CKPT}"
echo "MHOLD_CONFIG=${MHOLD_CONFIG}"
echo "MHOLD_15K_CKPT=${MHOLD_15K_CKPT}"
echo "TRAIN_BANK=${TRAIN_BANK}"
echo "MHOLD_LOG=${MHOLD_LOG}"
