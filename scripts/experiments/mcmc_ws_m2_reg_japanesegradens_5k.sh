#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
export SCENE_SLUG="japanesegradens"
export VARIANT="m2_reg"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-m2_reg}"

exec "${REPO_DIR}/scripts/experiments/mcmc_ws_seed500_common.sh"
