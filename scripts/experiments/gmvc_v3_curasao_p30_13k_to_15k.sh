#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

VARIANT="${VARIANT:-P30}"
GPU="${GPU:-6}"
LOAD_STEP="${LOAD_STEP:-13000}"
TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-15000}"
STAMP="${STAMP:-20260805_gmvc_v3_curasao_p30_13k_to_15k}"

A0_SOURCE_CKPT="${A0_SOURCE_CKPT:-${REPO_DIR}/outputs/gmvc_v3_a0_profile_persistence3k_curasao_seed42_step10000_to_13000/water-splatting/gmvc_v3_a0_profile_persistence3k_curasao_seed42_step10000_to_13000_20260805_gmvc_v3_curasao_r500_profile_persistence_3k_a0/nerfstudio_models/step-000013000.ckpt}"
P30_SOURCE_CKPT="${P30_SOURCE_CKPT:-${REPO_DIR}/outputs/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000/water-splatting/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000_20260805_gmvc_v3_curasao_r500_profile_persistence_3k_p30_p30_r500_g000/nerfstudio_models/step-000013000.ckpt}"
TRAIN_BANK="${TRAIN_BANK:-${REPO_DIR}/renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"

case "${VARIANT}" in
  A0|a0)
    SLUG="a0"
    SOURCE_CKPT="${A0_SOURCE_CKPT}"
    EXPERIMENT_NAME_DEFAULT="gmvc_v3_a0_15k_curasao_seed42_step${LOAD_STEP}_to_${TARGET_FINAL_STEP}"
    ;;
  P30|p30)
    SLUG="p30"
    SOURCE_CKPT="${P30_SOURCE_CKPT}"
    EXPERIMENT_NAME_DEFAULT="gmvc_v3_r500_p30_15k_curasao_seed42_step${LOAD_STEP}_to_${TARGET_FINAL_STEP}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use A0 or P30." >&2
    exit 2
    ;;
esac

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "Missing source checkpoint for ${VARIANT}: ${SOURCE_CKPT}" >&2
  exit 1
fi

exec env \
  VARIANT="${VARIANT}" \
  GPU="${GPU}" \
  LOAD_STEP="${LOAD_STEP}" \
  TARGET_FINAL_STEP="${TARGET_FINAL_STEP}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-2000}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${TARGET_FINAL_STEP}}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-1000}" \
  SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}" \
  RUN_EVAL="${RUN_EVAL:-0}" \
  STAMP="${STAMP}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_DEFAULT}}" \
  M1_LOAD_CHECKPOINT="${SOURCE_CKPT}" \
  GMVC_TRACK_BANK_PATH="${TRAIN_BANK}" \
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-49}" \
  "${REPO_DIR}/scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_3000.sh"
