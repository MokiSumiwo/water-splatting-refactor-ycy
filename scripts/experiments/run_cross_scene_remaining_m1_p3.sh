#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/new/home_old/ycy/water-splatting-refactor"
STAMP="${STAMP:-20260730_cross_scene}"
SEED="${SEED:-42}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-15000}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-${MAX_NUM_ITERATIONS}}"
STEPS_PER_SAVE="${STEPS_PER_SAVE:-5000}"
SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT:-False}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LAUNCH_LOG_DIR="${LOG_ROOT}/cross_scene_launcher_${STAMP}"

mkdir -p "${LAUNCH_LOG_DIR}"

run_bg() {
  local name="$1"
  local gpu="$2"
  local script="$3"
  echo "launch ${name} gpu=${gpu} script=${script}"
  env \
    GPU="${gpu}" \
    STAMP="${STAMP}" \
    SEED="${SEED}" \
    MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS}" \
    MODEL_NUM_STEPS="${MODEL_NUM_STEPS}" \
    STEPS_PER_SAVE="${STEPS_PER_SAVE}" \
    SAVE_ONLY_LATEST_CHECKPOINT="${SAVE_ONLY_LATEST_CHECKPOINT}" \
    "${REPO_DIR}/${script}" \
    >"${LAUNCH_LOG_DIR}/${name}.launcher.log" 2>&1 &
}

wait_group() {
  local group_name="$1"
  shift
  local failed=0
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "${group_name} failed; see ${LAUNCH_LOG_DIR}" >&2
    exit 1
  fi
}

echo "cross-scene run stamp=${STAMP} seed=${SEED} iterations=${MAX_NUM_ITERATIONS}" | tee "${LAUNCH_LOG_DIR}/manifest.txt"

run_bg "curasao_m1" "6" "scripts/experiments/cross_scene_curasao_m1_seed42_15000.sh"; p1=$!
run_bg "japanesegradens_m1" "7" "scripts/experiments/cross_scene_japanesegradens_m1_seed42_15000.sh"; p2=$!
run_bg "panama_m1" "8" "scripts/experiments/cross_scene_panama_m1_seed42_15000.sh"; p3=$!
wait_group "M1 group" "${p1}" "${p2}" "${p3}"

run_bg "curasao_p3" "6" "scripts/experiments/cross_scene_curasao_p3_seed42_15000.sh"; p4=$!
run_bg "japanesegradens_p3" "7" "scripts/experiments/cross_scene_japanesegradens_p3_seed42_15000.sh"; p5=$!
run_bg "panama_p3" "8" "scripts/experiments/cross_scene_panama_p3_seed42_15000.sh"; p6=$!
wait_group "P3 group" "${p4}" "${p5}" "${p6}"

echo "cross-scene M1/P3 run complete: ${STAMP}"

