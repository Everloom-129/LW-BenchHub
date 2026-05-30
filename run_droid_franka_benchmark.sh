#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-results/droid_franka_eval}"

python ./lw_benchhub/scripts/policy/benchmark_rollouts.py \
  --output_dir "${OUTPUT_DIR}" \
  --robot Panda \
  --task LiftObj \
  --layout robocasakitchen-9-8 \
  "${@:2}"
