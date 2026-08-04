#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"

VARIANT="${VARIANT:-O003}"
GPU="${GPU:-6}"
STAMP="${STAMP:-20260804_gmvc_v3_curasao_r500_object_lambda_sweep_1000}"

case "${VARIANT}" in
  O004|o004)
    OBJECT_LAMBDA="0.004"
    SLUG="o004"
    ;;
  O003|o003)
    OBJECT_LAMBDA="0.003"
    SLUG="o003"
    ;;
  O002|o002)
    OBJECT_LAMBDA="0.002"
    SLUG="o002"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}. Use O004, O003, or O002." >&2
    exit 2
    ;;
esac

exec env \
  VARIANT="R500" \
  GPU="${GPU}" \
  STAMP="${STAMP}_${SLUG}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-gmvc_v3_r500_${SLUG}_object_lambda_curasao_seed42_step10000_to_11000}" \
  LAMBDA_GMVC_OBJECT="${OBJECT_LAMBDA}" \
  GMVC_GRAD_LOG_EVERY="${GMVC_GRAD_LOG_EVERY:-49}" \
  "${REPO_DIR}/scripts/experiments/gmvc_v3_curasao_g000_ramp_sweep_1000.sh"
