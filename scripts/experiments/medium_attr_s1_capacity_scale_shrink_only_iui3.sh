#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

exec env \
  GPU="${GPU:-8}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_s1_capacity_scale_shrink_only_iui3_15000}" \
  STAMP="${STAMP:-20260729_s1_capacity_scale_shrink_only}" \
  CAPACITY_CONTROL_GEOMETRY_GRADIENT_SCALE="0.0" \
  CAPACITY_CONTROL_POSITION_GRADIENT_SCALE="0.0" \
  CAPACITY_CONTROL_DEPTH_GRADIENT_SCALE="0.0" \
  CAPACITY_CONTROL_FOOTPRINT_GRADIENT_SCALE="1.0" \
  CAPACITY_CONTROL_OPACITY_GRADIENT_SCALE="1.0" \
  CAPACITY_CONTROL_SCALE_SHRINK_ONLY="True" \
  CAPACITY_CONTROL_SCALE_SHRINK_CLIP_QUANTILE="-1.0" \
  CAPACITY_CONTROL_SCALE_SHRINK_CLIP_VALUE="0.0" \
  CAPACITY_CONFLICT_GATE_ENABLED="False" \
  CAPACITY_CONFLICT_RHO="1.0" \
  "${REPO_DIR}/scripts/experiments/medium_attr_q1_capacity_opacity_only_iui3.sh"
