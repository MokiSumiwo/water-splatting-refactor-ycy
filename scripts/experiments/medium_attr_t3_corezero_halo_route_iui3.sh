#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-8}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_t3_corezero_halo_route_iui3_15000}" \
  STAMP="${STAMP:-20260730_t3_corezero_halo_route}" \
  TRAINING_GRADIENT_ROUTING_ENABLED="True" \
  GRADIENT_ROUTING_MIN_SCENE_WEIGHT="${GRADIENT_ROUTING_MIN_SCENE_WEIGHT:-0.70}" \
  "${REPO_DIR}/scripts/experiments/medium_attr_t2_corezero_halo_iui3.sh"
