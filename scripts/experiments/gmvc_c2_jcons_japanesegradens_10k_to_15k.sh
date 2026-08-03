#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
exec env \
  GPU="${GPU:-6}" \
  SCENE_SLUG="japanesegradens_redsea" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}" \
  M1_LOAD_CONFIG="${M1_LOAD_CONFIG:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml}" \
  M1_LOAD_CHECKPOINT="${M1_LOAD_CHECKPOINT:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt}" \
  GMVC_VARIANT="c2_jcons" \
  GMVC_NEEDS_BANK="1" \
  GMVC_ENABLED="True" \
  GMVC_DIAGNOSTIC_ONLY="False" \
  LAMBDA_GMVC_J="${LAMBDA_GMVC_J:-0.002}" \
  LAMBDA_GMVC_RANGE="0.0" \
  LAMBDA_GMVC_BINF="0.0" \
  GMVC_FREEZE_GEOMETRY="True" \
  GMVC_TRAIN_FEATURES_DC="False" \
  GMVC_TRAIN_FEATURES_REST="False" \
  "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
