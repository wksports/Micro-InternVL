#!/usr/bin/env bash
# Train Micro-InternVL with the H20-optimized configuration.

set -e

cd "$(dirname "$0")/.." || exit 1

python micro_internvl/train.py --config micro_internvl/config_h20.yaml "$@"
