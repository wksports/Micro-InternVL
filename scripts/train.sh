#!/usr/bin/env bash
# Train Micro-InternVL.

set -e

cd "$(dirname "$0")/.." || exit 1

python micro_internvl/train.py --config micro_internvl/config.yaml "$@"
