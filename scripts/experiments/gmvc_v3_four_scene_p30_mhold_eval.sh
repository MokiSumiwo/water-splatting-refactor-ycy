#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
PYTHON="${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}"

SCENE="${SCENE:-ALL}"
GPU="${GPU:-6}"
SEED="${SEED:-42}"
STAMP_BASE="${STAMP:-20260806_gmvc_four_scene_p30_mhold_15k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
RENDER_ROOT="${RENDER_ROOT:-${REPO_DIR}/renders}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RENDER_ROOT}/gmvc_four_scene_p30_mhold_15k}"
MAX_IMAGES="${MAX_IMAGES:--1}"
FORCE_REBUILD_BANKS="${FORCE_REBUILD_BANKS:-0}"
TEST_MODE="${TEST_MODE:-test}"

scene_list() {
  if [[ "${SCENE}" == "ALL" || "${SCENE}" == "all" ]]; then
    echo "Curasao JapaneseGradens IUI3 Panama"
  else
    echo "${SCENE}"
  fi
}

resolve_scene() {
  local scene="$1"
  case "${scene}" in
    Curasao|curasao)
      SCENE_NAME="Curasao"
      SCENE_SLUG="curasao"
      DATA_PATH="${REPO_DIR}/undistorted_data/undistorted_Curasao"
      M1_CONFIG="${REPO_DIR}/outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
      A0_CONFIG="${A0_CONFIG_OVERRIDE:-${REPO_DIR}/outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml}"
      P30_CONFIG="${P30_CONFIG_OVERRIDE:-${REPO_DIR}/outputs/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_mhold/config.yml}"
      EVALF_BANK="${EVALF_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"
      EVALG_BANK="${EVALG_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt}"
      MHOLD_LOG="${MHOLD_LOG_OVERRIDE:-${REPO_DIR}/logs/gmvc_v3_p30_release_mhold_20260805_gmvc_v3_p30_profile_release_13k_to_15k.jsonl}"
      ;;
    JapaneseGradens|japanesegradens|japanesegradens_redsea)
      SCENE_NAME="JapaneseGradens"
      SCENE_SLUG="japanesegradens_redsea"
      DATA_PATH="${REPO_DIR}/undistorted_data/undistorted_JapaneseGradens-RedSea"
      M1_CONFIG="${REPO_DIR}/outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
      A0_CONFIG="${A0_CONFIG_OVERRIDE:-${REPO_DIR}/outputs/gmvc_v3_japanesegradens_p30_release_a0_seed42_step13000_to_15000/water-splatting/gmvc_v3_japanesegradens_p30_release_a0_seed42_step13000_to_15000_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k_a0/config.yml}"
      P30_CONFIG="${P30_CONFIG_OVERRIDE:-${REPO_DIR}/outputs/gmvc_v3_japanesegradens_p30_release_mhold_seed42_step13000_to_15000/water-splatting/gmvc_v3_japanesegradens_p30_release_mhold_seed42_step13000_to_15000_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k_mhold/config.yml}"
      EVALF_BANK="${EVALF_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v3_geometry_track_banks/japanesegradens_redsea_m1_step10000_train_s4096_seed123/gmvc_track_bank.pt}"
      EVALG_BANK="${EVALG_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v3_geometry_track_banks/japanesegradens_redsea_m1_step10000_train_s4096/gmvc_track_bank.pt}"
      MHOLD_LOG="${MHOLD_LOG_OVERRIDE:-${REPO_DIR}/logs/gmvc_v3_japanesegradens_p30_release_mhold_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k.jsonl}"
      ;;
    IUI3|iui3|iui3_redsea)
      SCENE_NAME="IUI3"
      SCENE_SLUG="iui3_redsea"
      DATA_PATH="${REPO_DIR}/undistorted_data/undistorted_IUI3-RedSea"
      M1_EXP="gmvc_v3_four_scene_iui3_m1_seed${SEED}_15000"
      M1_STAMP="${STAMP_BASE}_m1_bootstrap"
      M1_CONFIG="${M1_CONFIG_OVERRIDE:-${OUTPUT_DIR}/${M1_EXP}/water-splatting/${M1_EXP}_${M1_STAMP}/config.yml}"
      A0_EXP="gmvc_v3_four_scene_a0_${SCENE_SLUG}_seed${SEED}_step10000_to_15000"
      A0_STAMP="${STAMP_BASE}_a0"
      P30_EXP="gmvc_v3_four_scene_p30_mhold_${SCENE_SLUG}_seed${SEED}_step13000_to_15000"
      P30_STAMP="${STAMP_BASE}_mhold"
      A0_CONFIG="${A0_CONFIG_OVERRIDE:-${OUTPUT_DIR}/${A0_EXP}/water-splatting/${A0_EXP}_${A0_STAMP}/config.yml}"
      P30_CONFIG="${P30_CONFIG_OVERRIDE:-${OUTPUT_DIR}/${P30_EXP}/water-splatting/${P30_EXP}_${P30_STAMP}/config.yml}"
      EVALF_BANK="${EVALF_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v2_track_banks/${SCENE_SLUG}_m1_step10000_train_s4096/gmvc_track_bank.pt}"
      EVALG_BANK="${EVALG_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v3_geometry_track_banks/${SCENE_SLUG}_m1_step10000_train_s4096/gmvc_track_bank.pt}"
      MHOLD_LOG="${MHOLD_LOG_OVERRIDE:-${REPO_DIR}/logs/gmvc_v3_four_scene_p30_mhold_${SCENE_SLUG}_${STAMP_BASE}.jsonl}"
      ;;
    Panama|panama)
      SCENE_NAME="Panama"
      SCENE_SLUG="panama"
      DATA_PATH="${REPO_DIR}/undistorted_data/undistorted_Panama"
      M1_CONFIG="${REPO_DIR}/outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
      A0_EXP="gmvc_v3_four_scene_a0_${SCENE_SLUG}_seed${SEED}_step10000_to_15000"
      A0_STAMP="${STAMP_BASE}_a0"
      P30_EXP="gmvc_v3_four_scene_p30_mhold_${SCENE_SLUG}_seed${SEED}_step13000_to_15000"
      P30_STAMP="${STAMP_BASE}_mhold"
      A0_CONFIG="${A0_CONFIG_OVERRIDE:-${OUTPUT_DIR}/${A0_EXP}/water-splatting/${A0_EXP}_${A0_STAMP}/config.yml}"
      P30_CONFIG="${P30_CONFIG_OVERRIDE:-${OUTPUT_DIR}/${P30_EXP}/water-splatting/${P30_EXP}_${P30_STAMP}/config.yml}"
      EVALF_BANK="${EVALF_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v2_track_banks/panama_m1_step10000_train_s4096/gmvc_track_bank.pt}"
      EVALG_BANK="${EVALG_BANK_OVERRIDE:-${RENDER_ROOT}/gmvc_v3_geometry_track_banks/panama_m1_step10000_train_s4096/gmvc_track_bank.pt}"
      MHOLD_LOG="${MHOLD_LOG_OVERRIDE:-${REPO_DIR}/logs/gmvc_v3_four_scene_p30_mhold_${SCENE_SLUG}_${STAMP_BASE}.jsonl}"
      ;;
    *)
      echo "Unknown SCENE=${scene}. Use Curasao, JapaneseGradens, IUI3, Panama, or ALL." >&2
      exit 2
      ;;
  esac
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

build_bank_if_missing() {
  local bank_path="$1"
  local mode="$2"
  if [[ -f "${bank_path}" && "${FORCE_REBUILD_BANKS}" != "1" ]]; then
    return
  fi
  mkdir -p "$(dirname "${bank_path}")"
  local args=(
    --load-config "${M1_CONFIG}"
    --load-step 10000
    --split train
    --samples-per-view 4096
    --max-observations-per-camera 20000
    --output-path "${bank_path}"
  )
  if [[ "${mode}" == "geometry" ]]; then
    args+=(
      --geometry-only-v2-bank
      --signal-min 0.02
      --signal-max 0.98
      --signal-softness 0.05
    )
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/build_gmvc_tracks.py" "${args[@]}"
}

eval_scene() {
  local scene="$1"
  resolve_scene "${scene}"
  local scene_dir="${SUMMARY_ROOT}/${SCENE_NAME}"
  local metrics_dir="${scene_dir}/metrics"
  local fixed_dir="${scene_dir}/fixed_bank"
  local vis_dir="${scene_dir}/visualization"

  require_file "${A0_CONFIG}" "${SCENE_NAME} A0 config"
  require_file "${P30_CONFIG}" "${SCENE_NAME} P30-MHOLD config"
  require_file "${M1_CONFIG}" "${SCENE_NAME} M1 config"

  build_bank_if_missing "${EVALF_BANK}" "filtered"
  build_bank_if_missing "${EVALG_BANK}" "geometry"

  mkdir -p "${metrics_dir}" "${fixed_dir}" "${vis_dir}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/evaluate_checkpoint_metrics.py" \
    --load-config "${A0_CONFIG}" \
    --load-step 15000 \
    --test-mode "${TEST_MODE}" \
    --output-path "${metrics_dir}/a0_rgb_metrics.json"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/evaluate_checkpoint_metrics.py" \
    --load-config "${P30_CONFIG}" \
    --load-step 15000 \
    --test-mode "${TEST_MODE}" \
    --output-path "${metrics_dir}/p30_mhold_rgb_metrics.json"

  for bank in evalf evalg; do
    if [[ "${bank}" == "evalf" ]]; then
      bank_path="${EVALF_BANK}"
    else
      bank_path="${EVALG_BANK}"
    fi
    for run in a0 p30_mhold; do
      if [[ "${run}" == "a0" ]]; then
        config_path="${A0_CONFIG}"
      else
        config_path="${P30_CONFIG}"
      fi
      out_dir="${fixed_dir}/${bank}/${run}"
      mkdir -p "${out_dir}"
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/diagnose_gmvc_fixed_bank.py" \
        --load-config "${config_path}" \
        --load-step 15000 \
        --test-mode inference \
        --track-bank "${bank_path}" \
        --max-tracks 30000 \
        --train-fraction 0.80 \
        --seed 42 \
        --closure-signal-floor 0.03 \
        --irls-delta 0.03 \
        --irls-max-weight 1.0 \
        --min-hessian 1e-5 \
        --min-transmission-span 0.01 \
        --min-depth-span-rel 0.05 \
        --object-source J_proxy_raw \
        --force-dc-proxy \
        --output-dir "${out_dir}"
    done
  done

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/render_gmvc_underwater_dewatered_comparison.py" \
    --scene-name "${SCENE_NAME}" \
    --a0-config "${A0_CONFIG}" \
    --a0-step 15000 \
    --p30-config "${P30_CONFIG}" \
    --p30-step 15000 \
    --test-mode "${TEST_MODE}" \
    --max-images "${MAX_IMAGES}" \
    --output-dir "${vis_dir}"
}

for scene in $(scene_list); do
  eval_scene "${scene}"
done

summary_args=(
  --root "${SUMMARY_ROOT}"
  --mhold-log "Curasao=${REPO_DIR}/logs/gmvc_v3_p30_release_mhold_20260805_gmvc_v3_p30_profile_release_13k_to_15k.jsonl"
  --mhold-log "JapaneseGradens=${REPO_DIR}/logs/gmvc_v3_japanesegradens_p30_release_mhold_20260806_gmvc_v3_japanesegradens_p30_profile_release_13k_to_15k.jsonl"
  --mhold-log "IUI3=${REPO_DIR}/logs/gmvc_v3_four_scene_p30_mhold_iui3_redsea_${STAMP_BASE}.jsonl"
  --mhold-log "Panama=${REPO_DIR}/logs/gmvc_v3_four_scene_p30_mhold_panama_${STAMP_BASE}.jsonl"
)
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${REPO_DIR}/scripts/diagnostics/summarize_gmvc_four_scene_15k.py" "${summary_args[@]}"
