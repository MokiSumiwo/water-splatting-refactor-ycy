#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/new/home_old/ycy/water-splatting-refactor}"
GPU="${GPU:-6}"
FINAL_STEP="${FINAL_STEP:-15000}"

case "${GPU}" in
  6|7|8|9) ;;
  *)
    echo "BND-OMVC requires physical GPU 6, 7, 8, or 9; got '${GPU}'" >&2
    exit 2
    ;;
esac

set +u
if ! command -v conda >/dev/null 2>&1; then
  for profile in "${HOME}/.bashrc" "${HOME}/.bash_profile" "${HOME}/.profile"; do
    if [ -r "${profile}" ]; then
      # shellcheck disable=SC1090
      . "${profile}"
    fi
    if command -v conda >/dev/null 2>&1; then
      break
    fi
  done
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
  eval "$("${CONDA_EXE}" shell.bash hook)"
else
  echo "conda is not initialized; source your conda shell hook before running this script" >&2
  set -u
  exit 2
fi

conda activate water_splatting
set -u

mkdir -p "${REPO_DIR}/logs/bnd_omvc_panama_20260812"

(
  cd "${REPO_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/diagnostics/run_bnd_omvc_panama.py \
    --repo "${REPO_DIR}" \
    --final-step "${FINAL_STEP}"
) 2>&1 | tee "${REPO_DIR}/logs/bnd_omvc_panama_20260812/run.log"
