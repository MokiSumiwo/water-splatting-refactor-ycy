#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON_BIN="/opt/anaconda3/envs/water_splatting/bin/python"

cd "${REPO_ROOT}"
mkdir -p logs/bnd_cdepth_fixpop_rollout_panama_20260811

"${PYTHON_BIN}" scripts/diagnostics/run_bnd_cdepth_fixpop_rollout.py \
  2>&1 | tee logs/bnd_cdepth_fixpop_rollout_panama_20260811/run.log
