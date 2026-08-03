#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
exec env \
  GPU="${GPU:-6}" \
  SCENE_SLUG="japanesegradens_redsea" \
  DATA_PATH="${DATA_PATH:-${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea}" \
  M1_LOAD_CONFIG="${M1_LOAD_CONFIG:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml}" \
  M1_LOAD_CHECKPOINT="${M1_LOAD_CHECKPOINT:-${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt}" \
  GMVC_VARIANT="c1_diag" \
  GMVC_NEEDS_BANK="1" \
  GMVC_ENABLED="False" \
  GMVC_DIAGNOSTIC_ONLY="True" \
  "${REPO_DIR}/scripts/experiments/gmvc_phase_b_common.sh"
