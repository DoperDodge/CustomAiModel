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

# --- Detect Python ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null && "$cmd" --version &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "ERROR: Python not found!"
    echo ""
    echo "Please install Python 3.10+ first:"
    echo "  Windows : https://www.python.org/downloads/"
    echo "            (check 'Add python.exe to PATH' during install)"
    echo "  macOS   : brew install python"
    echo "  Ubuntu  : sudo apt install python3 python3-venv"
    echo ""
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1)
echo "Using $PYTHON_VERSION ($PYTHON)"

# --- Virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating virtual environment..."
    "$PYTHON" -m venv .venv
else
    echo "[1/5] Virtual environment already exists."
fi

# Locate venv binaries (Windows puts them in Scripts/, Unix in bin/)
if [ -f ".venv/Scripts/pip.exe" ]; then
    PIP=".venv/Scripts/pip.exe"
    VPYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/pip" ]; then
    PIP=".venv/bin/pip"
    VPYTHON=".venv/bin/python"
else
    echo "ERROR: Virtual environment was created but pip was not found inside it."
    echo "Try deleting .venv and running this script again:"
    echo "  rm -rf .venv && bash scripts/setup.sh"
    exit 1
fi

echo "  venv python: $VPYTHON"

# --- PyTorch with CUDA ---
echo "[2/5] Installing PyTorch with CUDA support..."
"$VPYTHON" -m pip install --upgrade pip

# Try CUDA indexes from newest to oldest to find one with wheels for this Python version
TORCH_INSTALLED=false
for cuda_ver in cu126 cu124 cu121; do
    echo "  Trying PyTorch index: $cuda_ver ..."
    if "$VPYTHON" -m pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/$cuda_ver" 2>/dev/null; then
        echo "  Installed from $cuda_ver index."
        TORCH_INSTALLED=true
        break
    fi
done

if [ "$TORCH_INSTALLED" = false ]; then
    echo "  CUDA indexes failed — installing PyTorch from default PyPI..."
    "$VPYTHON" -m pip install torch torchvision torchaudio
fi

# --- Project dependencies ---
echo "[3/5] Installing project dependencies..."
"$VPYTHON" -m pip install -r requirements.txt

# --- GPU check ---
echo "[4/5] Checking GPU availability..."
"$VPYTHON" -c "
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
"$VPYTHON" -c "
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
