#!/usr/bin/env bash
# Push MicroDetect to GitHub.
# Run this manually after ensuring you have GitHub write access.

set -e

cd "$(dirname "$0")/.." || exit 1

REMOTE_URL="https://github.com/wksports/Micro-InternVL.git"

if ! git remote | grep -q origin; then
    git remote add origin "${REMOTE_URL}"
else
    git remote set-url origin "${REMOTE_URL}"
fi

git add -A
git commit -m "Initial Micro-InternVL implementation based on official InternVL3.5-4B" \
    -m "- Add high-resolution patch detection head on uncompressed InternViT tokens" \
    -m "- Add cross-scale contrastive alignment (patch-text + box-text InfoNCE)" \
    -m "- Add hierarchical query support" \
    -m "- Add EMDS-7 training/evaluation pipeline" \
    -m "- Include H20 config, deployment scripts and smoke tests"

git push -u origin master

echo "Pushed to ${REMOTE_URL}"
