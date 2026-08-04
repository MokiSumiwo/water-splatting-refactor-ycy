#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

VARIANT="${VARIANT:-P35}"
GPU="${GPU:-6}"
STAMP="${STAMP:-20260804_gmvc_v3_curasao_r500_profile_lambda_sweep_1000}"

case "${VARIANT}" in
  P40|p40)
    PROFILE_LAMBDA="40"
    SLUG="p40"
    ;;
  P35|p35)
    PROFILE_LAMBDA="35"
    SLUG="p35"
    ;;
  P30|p30)
    PROFILE_LAMBDA="30"
    SLUG="p30"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use P40, P35, or P30." >&2
    exit 2
    ;;
esac

exec env \
  VARIANT="R500" \
  GPU="${GPU}" \
  STAMP="${STAMP}_${SLUG}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-gmvc_v3_r500_${SLUG}_profile_lambda_curasao_seed42_step10000_to_11000}" \
  LAMBDA_GMVC_PROFILE="${PROFILE_LAMBDA}" \
  LAMBDA_GMVC_OBJECT="${LAMBDA_GMVC_OBJECT:-0.004}" \
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-49}" \
  "${REPO_DIR}/scripts/experiments/gmvc_v3_curasao_g000_ramp_sweep_1000.sh"
