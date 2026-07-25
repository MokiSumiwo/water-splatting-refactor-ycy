#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-8}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
BASE_LOAD_DIR="${BASE_LOAD_DIR:?Set BASE_LOAD_DIR to the selected stage1 nerfstudio_models directory}"
BASE_STEP="${BASE_STEP:-16999}"
FINETUNE_STEPS="${FINETUNE_STEPS:-2000}"
MEDIUM_LR="${MEDIUM_LR:-0.0001}"
MEDIUM_LR_FINAL="${MEDIUM_LR_FINAL:-0.000015}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dual_color_stage2_joint_medium_seed${SEED:-42}_iui3_redsea_${FINETUNE_STEPS}}"

export GPU BASE_LOAD_DIR BASE_STEP FINETUNE_STEPS MEDIUM_LR MEDIUM_LR_FINAL EXPERIMENT_NAME
export DUAL_COLOR_FREEZE_GEOMETRY=True
export DUAL_COLOR_FREEZE_MEDIUM=False

exec "${REPO_DIR}/scripts/experiments/dual_color_stage1_frozen_geometry_iui3.sh"
