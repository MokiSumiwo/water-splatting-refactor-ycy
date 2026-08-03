#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gmvc_p0_nofreeze_japanesegradens_redsea_seed42_step10000_to_10200}"
STAMP="${STAMP:-20260803_gmvc_p0_nofreeze_short}"

exec env \
  GPU="${GPU:-6}" \
  SCENE_SLUG="japanesegradens_redsea" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}" \
  M1_LOAD_CONFIG="${M1_LOAD_CONFIG:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml}" \
  M1_LOAD_CHECKPOINT="${M1_LOAD_CHECKPOINT:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  STAMP="${STAMP}" \
  TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-10200}" \
  MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-10200}" \
  MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-200}" \
  STEPS_PER_SAVE="${STEPS_PER_SAVE:-200}" \
  GMVC_VARIANT="p0_nofreeze" \
  GMVC_NEEDS_BANK="0" \
  GMVC_ENABLED="False" \
  GMVC_DIAGNOSTIC_ONLY="False" \
  LAMBDA_GMVC_INTRINSIC="0.0" \
  GMVC_FREEZE_GEOMETRY="False" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_CLOSURE_DIAG="${RUN_CLOSURE_DIAG:-0}" \
  "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
