#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/new/home_old/ycy/water-splatting-refactor}
PYTHON=${PYTHON:-/opt/anaconda3/envs/water_splatting/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/renders/gmvc_oracle_pareto_20260803}
SAMPLES_PER_VIEW=${SAMPLES_PER_VIEW:-4096}
MAX_TRACKS=${MAX_TRACKS:-30000}
ITERS=${ITERS:-500}
LR=${LR:-0.03}
SCENE=${SCENE:-japanesegradens_panama}
CLOSURE_SIGNAL_FLOOR=${CLOSURE_SIGNAL_FLOOR:-0.03}

O1_VARIANTS=${O1_VARIANTS:-"O1_S1:0.05:0.05:0:0:0;O1_S2:0.10:0.075:0:0:0;O1_S3:0.15:0.10:0:0:0;O1_S4:0.20:0.15:0:0:0;O1_R1:0.15:0.10:0.0001:0.0001:0;O1_R2:0.15:0.10:0.0005:0.0001:0;O1_R3:0.15:0.10:0.001:0.0005:0;O1_C1:0.15:0.10:0.0005:0.0001:0.01;O1_C2:0.15:0.10:0.0005:0.0001:0.05;O1_C3:0.15:0.10:0.0005:0.0001:0.10"}

cd "${REPO_ROOT}"

run_pareto() {
  local scene="$1"
  local gpu="$2"
  local config="$3"
  local step="$4"
  local output_name="$5"

  echo "[GMVC Phase O Pareto] scene=${scene} gpu=${gpu} step=${step}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/diagnostics/fit_gmvc_lowdim_oracle.py \
    --load-config "${config}" \
    --load-step "${step}" \
    --test-mode inference \
    --split train \
    --max-images 0 \
    --samples-per-view "${SAMPLES_PER_VIEW}" \
    --target-neighbor-window 0 \
    --max-tracks "${MAX_TRACKS}" \
    --iters "${ITERS}" \
    --lr "${LR}" \
    --models O0,O1 \
    --o1-variants "${O1_VARIANTS}" \
    --closure-signal-floor "${CLOSURE_SIGNAL_FLOOR}" \
    --output-dir "${OUTPUT_ROOT}/${output_name}" \
    --fit-device cuda
}

case "${SCENE}" in
  japanesegradens|JapaneseGradens|iui3|IUI3|curasao|Curasao|panama|Panama|japanesegradens_panama|all) ;;
  *)
    echo "Unknown SCENE=${SCENE}. Use japanesegradens, iui3, curasao, panama, japanesegradens_panama, or all." >&2
    exit 2
    ;;
esac

if [[ "${SCENE}" == "all" || "${SCENE}" == "japanesegradens_panama" || "${SCENE}" == "japanesegradens" || "${SCENE}" == "JapaneseGradens" ]]; then
  run_pareto \
    "JapaneseGradens" "${GPU_JAPANESEGRADENS:-6}" \
    "outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml" \
    "10000" \
    "japanesegradens_m1_step10000_train_s4096"
fi

if [[ "${SCENE}" == "all" || "${SCENE}" == "iui3" || "${SCENE}" == "IUI3" ]]; then
  run_pareto \
    "IUI3" "${GPU_IUI3:-7}" \
    "outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml" \
    "14999" \
    "iui3_m1_step14999_train_s4096"
fi

if [[ "${SCENE}" == "all" || "${SCENE}" == "curasao" || "${SCENE}" == "Curasao" ]]; then
  run_pareto \
    "Curasao" "${GPU_CURASAO:-8}" \
    "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml" \
    "10000" \
    "curasao_m1_step10000_train_s4096"
fi

if [[ "${SCENE}" == "all" || "${SCENE}" == "japanesegradens_panama" || "${SCENE}" == "panama" || "${SCENE}" == "Panama" ]]; then
  run_pareto \
    "Panama" "${GPU_PANAMA:-9}" \
    "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml" \
    "10000" \
    "panama_m1_step10000_train_s4096"
fi
