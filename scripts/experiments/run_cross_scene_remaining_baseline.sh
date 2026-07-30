#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
STAMP="${STAMP:-20260730_cross_scene_baseline}"
SEED="${SEED:-42}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${MAX_NUM_ITERATIONS}}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LAUNCH_LOG_DIR="${LOG_ROOT}/cross_scene_baseline_launcher_${STAMP}"

mkdir -p "${LAUNCH_LOG_DIR}"

run_bg() {
  local name="$1"
  local gpu="$2"
  local script="$3"
  echo "launch ${name} gpu=${gpu} script=${script}"
  env \
    GPU="${gpu}" \
    STAMP="${STAMP}" \
    M1_STAMP="${M1_STAMP:-20260730_cross_scene}" \
    SEED="${SEED}" \
    MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS}" \
    MODEL_NUM_STEPS="${MODEL_NUM_STEPS}" \
    STEPS_PER_SAVE="${STEPS_PER_SAVE}" \
    SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT}" \
    "${REPO_DIR}/${script}" \
    >"${LAUNCH_LOG_DIR}/${name}.launcher.log" 2>&1 &
}

echo "cross-scene baseline run stamp=${STAMP} seed=${SEED} iterations=${MAX_NUM_ITERATIONS}" | tee "${LAUNCH_LOG_DIR}/manifest.txt"

run_bg "curasao_baseline" "6" "scripts/experiments/cross_scene_curasao_baseline_seed42_15000.sh"; p1=$!
run_bg "japanesegradens_baseline" "7" "scripts/experiments/cross_scene_japanesegradens_baseline_seed42_15000.sh"; p2=$!
run_bg "panama_baseline" "8" "scripts/experiments/cross_scene_panama_baseline_seed42_15000.sh"; p3=$!

failed=0
for pid in "${p1}" "${p2}" "${p3}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "baseline group failed; see ${LAUNCH_LOG_DIR}" >&2
  exit 1
fi

echo "cross-scene baseline run complete: ${STAMP}"

