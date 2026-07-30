#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-7}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_t5_corezero_halo_route_objrad_amp_iui3_15000}" \
  STAMP="${STAMP:-20260730_t5_corezero_halo_route_objrad_amp}" \
  CORE_CLEARANCE_AMPLIFIER_ENABLED="True" \
  CORE_CLEARANCE_AMPLIFIER_MIN="${CORE_CLEARANCE_AMPLIFIER_MIN:-0.30}" \
  CORE_CLEARANCE_AMPLIFIER_THRESHOLD="${CORE_CLEARANCE_AMPLIFIER_THRESHOLD:-0.20}" \
  CORE_CLEARANCE_AMPLIFIER_TEMPERATURE="${CORE_CLEARANCE_AMPLIFIER_TEMPERATURE:-0.05}" \
  "${REPO_DIR}/scripts/experiments/medium_attr_t4_corezero_halo_route_objrad_iui3.sh"
