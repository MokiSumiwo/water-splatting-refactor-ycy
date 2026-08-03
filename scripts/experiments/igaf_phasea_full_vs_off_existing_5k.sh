#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"
GPU="${GPU:-6}"
MAX_IMAGES="${MAX_IMAGES:-}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/renders/igaf_phasea_full_vs_off_${STAMP}}"
mkdir -p "${OUT_ROOT}"

run_one() {
  local scene="$1"
  local tag="$2"
  local config="$3"
  local baseline="$4"
  local out_json="${OUT_ROOT}/${scene}_${tag}_full_vs_off.json"
  local args=(
    "${REPO_DIR}/scripts/diagnostics/eval_igaf_full_vs_off.py"
    --load-config "${config}"
    --baseline-json "${baseline}"
    --output-json "${out_json}"
  )
  if [[ -n "${MAX_IMAGES}" ]]; then
    args+=(--max-images "${MAX_IMAGES}")
  fi
  echo "[phaseA] ${scene} ${tag}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${args[@]}" 2>&1 | tee "${OUT_ROOT}/${scene}_${tag}.log"
}

run_one \
  "japanesegradens" \
  "g1_nomip" \
  "${REPO_DIR}/outputs/igaf_g1_nomip_japanesegradens_seed42_5000/water-splatting/igaf_g1_nomip_japanesegradens_seed42_5000_20260803_igaf5k_jg/config.yml" \
  "${REPO_DIR}/renders/igaf_g0_m1_japanesegradens_seed42_5000_20260803_igaf5k_jg/output.json"

run_one \
  "japanesegradens" \
  "g2_legacy_mip" \
  "${REPO_DIR}/outputs/igaf_g2_mip_japanesegradens_seed42_5000/water-splatting/igaf_g2_mip_japanesegradens_seed42_5000_20260803_igaf5k_jg/config.yml" \
  "${REPO_DIR}/renders/igaf_g0_m1_japanesegradens_seed42_5000_20260803_igaf5k_jg/output.json"

run_one \
  "iui3" \
  "g1_nomip" \
  "${REPO_DIR}/outputs/igaf_g1_nomip_iui3_seed42_5000/water-splatting/igaf_g1_nomip_iui3_seed42_5000_20260803_igaf5k_iui3/config.yml" \
  "${REPO_DIR}/renders/igaf_g0_m1_iui3_seed42_5000_20260803_igaf5k_iui3/output.json"

run_one \
  "iui3" \
  "g2_legacy_mip" \
  "${REPO_DIR}/outputs/igaf_g2_mip_iui3_seed42_5000/water-splatting/igaf_g2_mip_iui3_seed42_5000_20260803_igaf5k_iui3/config.yml" \
  "${REPO_DIR}/renders/igaf_g0_m1_iui3_seed42_5000_20260803_igaf5k_iui3/output.json"

echo "saved=${OUT_ROOT}"
