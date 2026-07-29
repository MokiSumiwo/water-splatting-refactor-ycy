#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-8}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_r2_capacity_conflict000_isolated_iui3_15000}" \
  STAMP="${STAMP:-20260729_r2_capacity_conflict000_isolated}" \
  CAPACITY_CONFLICT_GATE_ENABLED="True" \
  CAPACITY_CONFLICT_RHO="0.0" \
  CAPACITY_CONFLICT_REC_GRAD_THRESHOLD="${CAPACITY_CONFLICT_REC_GRAD_THRESHOLD:-1e-10}" \
  "${REPO_DIR}/scripts/experiments/medium_attr_q1_capacity_opacity_only_iui3.sh"
