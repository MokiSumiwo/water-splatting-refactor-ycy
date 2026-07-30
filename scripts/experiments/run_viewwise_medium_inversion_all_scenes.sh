#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"
CASES_TSV="${CASES_TSV:-${REPO_DIR}/scripts/diagnostics/viewwise_medium_inversion_cases.tsv}"
STAMP="${STAMP:-20260730_medium_inversion}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/renders/viewwise_medium_inversion_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/viewwise_medium_inversion_${STAMP}}"
GPU_DEFAULT="${GPU_DEFAULT:-6}"
MAX_IMAGES="${MAX_IMAGES:--1}"
TRANSMISSION_FLOOR="${TRANSMISSION_FLOOR:-0.05}"
MINIMUM_MASK_PIXELS="${MINIMUM_MASK_PIXELS:-1000}"
CASE_FILTER="${CASE_FILTER:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_case() {
  local scene="$1"
  local method="$2"
  local config_path="$3"
  local far_mask_dir="$4"
  local region_mask_dir="$5"
  local gpu="${GPU_DEFAULT}"
  case "${scene}" in
    iui3) gpu="${GPU_IUI3:-${GPU_DEFAULT}}" ;;
    curasao) gpu="${GPU_CURASAO:-${GPU_DEFAULT}}" ;;
    japanesegradens) gpu="${GPU_JGRADENS:-${GPU_DEFAULT}}" ;;
    panama) gpu="${GPU_PANAMA:-${GPU_DEFAULT}}" ;;
  esac

  local out_dir="${OUTPUT_ROOT}/${scene}/${method}"
  local log_file="${LOG_ROOT}/${scene}_${method}.log"
  mkdir -p "${out_dir}"

  echo "scene=${scene} method=${method} gpu=${gpu}" | tee "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_viewwise_medium_inversion.py" \
    --load-config "${REPO_DIR}/${config_path}" \
    --split eval \
    --max-images "${MAX_IMAGES}" \
    --far-mask-dir "${REPO_DIR}/${far_mask_dir}" \
    --region-mask-dir "${REPO_DIR}/${region_mask_dir}" \
    --output-dir "${out_dir}" \
    --transmission-floor "${TRANSMISSION_FLOOR}" \
    --smooth-kernels 31 61 \
    --minimum-mask-pixels "${MINIMUM_MASK_PIXELS}" \
    --save-full-resolution \
    --save-contact-sheet \
    --save-json \
    2>&1 | tee -a "${log_file}"
}

tail -n +2 "${CASES_TSV}" | while IFS=$'\t' read -r scene method config_path far_mask_dir region_mask_dir; do
  if [[ -n "${CASE_FILTER}" ]]; then
    case_key="${scene}/${method}"
    if [[ "${case_key}" != ${CASE_FILTER} ]]; then
      continue
    fi
  fi
  run_case "${scene}" "${method}" "${config_path}" "${far_mask_dir}" "${region_mask_dir}"
done

"${PYTHON}" - <<'PY' "${OUTPUT_ROOT}"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for aggregate_path in sorted(root.glob("*/*/aggregate.json")):
    data = json.loads(aggregate_path.read_text())
    scene = aggregate_path.parent.parent.name
    method = aggregate_path.parent.name
    variants = data["aggregate"]["variants"]
    forward = data["aggregate"]["forward_recomposition"]
    for key in ["D0_model", "D1_pixel", "D2_A_mean", "D3_bs_mean", "D4_attn_mean", "D5_all_mean", "D7_open_mean", "D9_smooth31", "D10_smooth61"]:
        if key not in variants:
            continue
        item = variants[key]
        rows.append({
            "scene": scene,
            "method": method,
            "variant": key,
            "far_bg_score": item["far_bg_score"],
            "near_bg_score": item["near_bg_score"],
            "far_near_bg_gap": item["far_near_bg_gap"],
            "near_rgb_mae_vs_d0": item["near_rgb_mae_vs_d0"],
            "raw_clip_rate": item["raw_clip_rate"],
            "transmission_floor_hit_rate": item["transmission_floor_hit_rate"],
            "forward_psnr": forward["psnr"],
            "forward_mae": forward["mae"],
        })

summary = {"rows": rows}
(root / "summary").mkdir(exist_ok=True)
(root / "summary" / "summary.json").write_text(json.dumps(summary, indent=2))
print(root / "summary" / "summary.json")
PY
