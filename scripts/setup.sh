#!/usr/bin/env bash
# ============================================================
# Custom AI Model — Environment Setup Script
# ============================================================
# Usage: bash scripts/setup.sh
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs PyTorch with CUDA support
#   3. Installs all project dependencies
#   4. Verifies GPU availability
#   5. Downloads a starter model (TinyLlama 1.1B)

set -euo pipefail

echo "================================================"
echo " Custom AI Model — Setup"
echo "================================================"

# --- Virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/5] Virtual environment already exists."
fi

source .venv/bin/activate

# --- PyTorch with CUDA ---
echo "[2/5] Installing PyTorch with CUDA 12.1..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Project dependencies ---
echo "[3/5] Installing project dependencies..."
pip install -r requirements.txt

# --- GPU check ---
echo "[4/5] Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU found: {gpu} ({mem:.1f} GB)')
else:
    print('  WARNING: No GPU detected. Training will be very slow.')
    print('  Make sure NVIDIA drivers and CUDA are installed.')
"

# --- Download starter model ---
echo "[5/5] Pre-downloading TinyLlama 1.1B (for fast first run)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='./checkpoints/tinyllama-1.1b')
print('  Model downloaded to ./checkpoints/tinyllama-1.1b')
" 2>/dev/null || echo "  (Skipped — will download on first use)"

echo ""
echo "================================================"
echo " Setup complete!"
echo ""
echo " Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo " Start the API server:"
echo "   uvicorn src.api.server:app --host 0.0.0.0 --port 8000"
echo ""
echo " Fine-tune the LLM:"
echo "   python -m src.text_to_text.train --help"
echo "================================================"
