#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-7}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_q2_capacity_conflict025_iui3_15000}" \
  STAMP="${STAMP:-20260729_q2_capacity_conflict025}" \
  CAPACITY_CONFLICT_GATE_ENABLED="True" \
  CAPACITY_CONFLICT_RHO="0.25" \
  "${REPO_DIR}/scripts/experiments/medium_attr_q1_capacity_opacity_only_iui3.sh"
