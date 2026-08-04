#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

VARIANT="${VARIANT:-R100}"
GPU="${GPU:-6}"
STAMP="${STAMP:-20260804_gmvc_v3_curasao_g000_ramp_sweep_1000}"

case "${VARIANT}" in
  R100|r100)
    RAMP_STEPS="100"
    SLUG="r100"
    ;;
  R300|r300)
    RAMP_STEPS="300"
    SLUG="r300"
    ;;
  R500|r500)
    RAMP_STEPS="500"
    SLUG="r500"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use R100, R300, or R500." >&2
    exit 2
    ;;
esac

exec env \
  VARIANT="G000" \
  GPU="${GPU}" \
  STAMP="${STAMP}_${SLUG}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-gmvc_v3_g000_${SLUG}_object_medium_freeze_curasao_seed42_step10000_to_11000}" \
  GMVC_RAMP_STEPS="${RAMP_STEPS}" \
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-49}" \
  "${REPO_DIR}/scripts/experiments/gmvc_v3_curasao_medium_grad_sweep_1000.sh"
