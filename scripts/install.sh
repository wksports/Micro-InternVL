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
pip install transformers accelerate peft timm
pip install pycocotools scipy Pillow PyYAML tqdm
pip install wandb bitsandbytes

echo "Installation complete."
echo "Next steps:"
echo "  1. Generate queries: python scripts/generate_queries.py --config micro_internvl/config.yaml"
echo "  2. Start training:    bash scripts/train.sh"
