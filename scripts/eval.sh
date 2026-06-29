#!/usr/bin/env bash
# Evaluate Micro-InternVL.

set -e

cd "$(dirname "$0")/.." || exit 1

CHECKPOINT=${1:-checkpoints/final}
SPLIT=${2:-test}

python micro_internvl/evaluate.py --config micro_internvl/config.yaml --checkpoint "${CHECKPOINT}" --split "${SPLIT}"
