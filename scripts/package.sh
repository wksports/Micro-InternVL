#!/usr/bin/env bash
# Package Micro-InternVL for deployment on another server.

set -e

cd "$(dirname "$0")/.." || exit 1

VERSION=${1:-$(date +%Y%m%d)}
ARCHIVE="micro-internvl-deploy-${VERSION}.tar.gz"

tar czvf "${ARCHIVE}" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="*.pyo" \
    internvl \
    micro_internvl \
    scripts \
    tests \
    data/emds7 \
    checkpoints \
    outputs \
    README.md

echo "Packaged deployment archive: ${ARCHIVE}"
