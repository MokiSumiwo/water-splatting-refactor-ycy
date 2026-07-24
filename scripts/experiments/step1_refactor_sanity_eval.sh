#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
ENV_PY="/opt/anaconda3/envs/water_splatting/bin/python"
NS_EVAL="/opt/anaconda3/envs/water_splatting/bin/ns-eval"
DATA_PATH="${REPO_DIR}/undistorted_data/IUI3-RedSea"
CONFIG_PATH="${REPO_DIR}/outputs/baseline_original_watersplatting_iui3_redsea/water-splatting/orig_watersplatting_iui3_redsea_15000_20260723_063201/config.yml"
RENDER_ROOT="${REPO_DIR}/renders"
LOG_ROOT="${REPO_DIR}/logs"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="step1_refactor_sanity_iui3_redsea_${STAMP}"
RENDER_DIR="${RENDER_ROOT}/${EXPERIMENT_NAME}"
LOG_PATH="${LOG_ROOT}/${EXPERIMENT_NAME}.log"

mkdir -p "${RENDER_DIR}" "${LOG_ROOT}"

echo "experiment=${EXPERIMENT_NAME}"
echo "gpu=${GPU}"
echo "data=${DATA_PATH}"
echo "config=${CONFIG_PATH}"
echo "render_dir=${RENDER_DIR}"
echo "python=${ENV_PY}"

pushd "${RENDER_DIR}" >/dev/null
CUDA_VISIBLE_DEVICES="${GPU}" "${NS_EVAL}" \
  --load-config "${CONFIG_PATH}" \
  --render-output-path "${RENDER_DIR}" \
  2>&1 | tee "${LOG_PATH}"
popd >/dev/null
