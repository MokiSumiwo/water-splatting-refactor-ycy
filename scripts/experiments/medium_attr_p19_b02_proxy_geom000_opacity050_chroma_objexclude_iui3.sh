#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
MASK_DIR="${MASK_DIR:-${REPO_DIR}/common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726}"

exec env \
  GPU="${GPU:-6}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-medium_attr_p19_b02_proxy_geom000_opacity050_chroma_objexclude_iui3_15000}" \
  STAMP="${STAMP:-20260729_p19_b02_proxy_geom000_opacity050_chroma_objexclude}" \
  BACKSCATTER_REGION_MASK_DIR="${BACKSCATTER_REGION_MASK_DIR:-${MASK_DIR}}" \
  MEDIUM_SUPPORT_REGION_EXCLUSION_ENABLED="True" \
  MEDIUM_SUPPORT_EXCLUDE_OBJECT="True" \
  MEDIUM_SUPPORT_EXCLUDE_BOUNDARY="False" \
  MEDIUM_SUPPORT_REGION_EXCLUSION_APPLY_CAPACITY="False" \
  MEDIUM_SUPPORT_REGION_EXCLUSION_APPLY_CHROMA="True" \
  "${REPO_DIR}/scripts/experiments/medium_attr_p3_b02_proxy_geom000_opacity050_iui3.sh"
