#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_${MAX_NUM_ITERATIONS:-15000}}"
export DC_CHANNEL_BALANCE_WEIGHT="${DC_CHANNEL_BALANCE_WEIGHT:-0.001}"
export DC_CHANNEL_BALANCE_MARGIN="${DC_CHANNEL_BALANCE_MARGIN:-0.05}"
export DC_CHANNEL_BALANCE_BETA="${DC_CHANNEL_BALANCE_BETA:-0.05}"
export DC_CHANNEL_BALANCE_USE_LOW_TRANS_WEIGHT="${DC_CHANNEL_BALANCE_USE_LOW_TRANS_WEIGHT:-True}"
export MEDIUM_ATTENUATION_ORDER_WEIGHT="${MEDIUM_ATTENUATION_ORDER_WEIGHT:-0.0}"

exec "${SCRIPT_DIR}/m4_constrained_appearance_iui3_redsea.sh"
