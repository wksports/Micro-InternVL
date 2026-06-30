#!/usr/bin/env bash
# Install Micro-InternVL dependencies for deployment.

set -e

echo "Installing Micro-InternVL dependencies..."

if [ "$1" == "--venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
fi

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo "Installation complete."
echo "Next steps:"
echo "  1. Download images:   python scripts/download_emds7.py"
echo "  2. Generate queries:  python scripts/generate_queries.py --config micro_internvl/config.yaml"
echo "  3. Start training:    bash scripts/train.sh"
