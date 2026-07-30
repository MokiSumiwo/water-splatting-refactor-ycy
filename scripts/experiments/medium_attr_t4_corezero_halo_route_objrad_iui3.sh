#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-9}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_t4_corezero_halo_route_objrad_iui3_15000}" \
  STAMP="${STAMP:-20260730_t4_corezero_halo_route_objrad}" \
  OBJECT_RADIANCE_BUDGET_ENABLED="True" \
  LAMBDA_OBJECT_RADIANCE_BUDGET="${LAMBDA_OBJECT_RADIANCE_BUDGET:-0.0002}" \
  "${REPO_DIR}/scripts/experiments/medium_attr_t3_corezero_halo_route_iui3.sh"
