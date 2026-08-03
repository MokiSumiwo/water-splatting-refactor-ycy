#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export SCENE_SLUG="japanesegradens"
export VARIANT="m0_adc"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-m0_adc}"

exec "${REPO_DIR}/scripts/experiments/mcmc_ws_seed500_common.sh"
