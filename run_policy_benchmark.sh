#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-policy/GR00T/deploy_policy_lerobot.yml}"
OUTPUT_DIR="${2:-results/policy_eval}"

python ./lw_benchhub/scripts/policy/eval_policy.py \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "${@:3}"
