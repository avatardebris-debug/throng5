#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# setup_cloud.sh — One-shot cloud setup for throng5 Lolo training
#
# Usage:  bash setup_cloud.sh
#
# Designed for vast.ai (Ubuntu + CUDA image) or any fresh Linux box.
# Installs: Python deps, stable-retro, NES ROM tools, and throng5.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════"
echo "  throng5 Cloud Setup"
echo "═══════════════════════════════════════════════════"

# ── 1. System packages ──────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv unzip wget curl \
    libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
echo "  ✓ System packages installed"

# ── 2. Clone repo ───────────────────────────────────────────────────
echo "[2/6] Cloning throng5..."
if [ ! -d "throng5" ]; then
    # CHANGE THIS to your actual repo URL:
    git clone https://github.com/YOUR_USERNAME/throng5.git
    echo "  ✓ Cloned throng5"
else
    cd throng5 && git pull && cd ..
    echo "  ✓ throng5 already exists, pulled latest"
fi
cd throng5

# ── 3. Python environment ──────────────────────────────────────────
echo "[3/6] Setting up Python environment..."
python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip -q

# Core dependencies
pip install -q \
    numpy \
    torch torchvision \
    gymnasium \
    stable-retro \
    scipy \
    matplotlib \
    tqdm

echo "  ✓ Python dependencies installed"

# ── 4. stable-retro ROM setup ──────────────────────────────────────
echo "[4/6] Setting up stable-retro..."

# Import the Lolo ROM (you need to provide the ROM file)
RETRO_DATA_DIR=$(python3 -c "import retro; print(retro.data.path())")
echo "  Retro data dir: ${RETRO_DATA_DIR}"

# Create Lolo game directory if needed
LOLO_DIR="${RETRO_DATA_DIR}/AdventuresOfLolo-Nes"
mkdir -p "${LOLO_DIR}"

# Check if ROM exists
if [ -f "roms/Adventures of Lolo (USA).nes" ]; then
    cp "roms/Adventures of Lolo (USA).nes" "${LOLO_DIR}/rom.nes"
    echo "  ✓ Lolo ROM installed"
else
    echo "  ⚠ ROM not found at roms/Adventures of Lolo (USA).nes"
    echo "    You need to manually place the ROM file."
    echo "    After placing it, run:"
    echo "      python3 -m retro.import roms/"
fi

# Copy game metadata (rom.sha, data.json, scenario.json)
if [ -d "brain/games/lolo/retro_data" ]; then
    cp brain/games/lolo/retro_data/* "${LOLO_DIR}/" 2>/dev/null || true
    echo "  ✓ Game metadata copied"
fi

# ── 5. Verify installation ────────────────────────────────────────
echo "[5/6] Verifying installation..."
python3 -c "
import numpy as np; print(f'  numpy {np.__version__}')
import torch; print(f'  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
try:
    import retro; print(f'  stable-retro {retro.__version__}')
except: print('  stable-retro: not verified (ROM may be needed)')
from brain.games.lolo.lolo_simulator import LoloSimulator; print('  lolo_simulator ✓')
from brain.games.lolo.lolo_gan import LoloGAN; print('  lolo_gan ✓')
from brain.games.lolo.lolo_compressed_state import LoloCompressedState; print('  compressed_state ✓')
print('  All imports OK')
"

# ── 6. Create output directories ──────────────────────────────────
echo "[6/6] Creating output directories..."
mkdir -p outputs/weights outputs/checkpoints outputs/logs
echo "  ✓ Output dirs created"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete! Next steps:"
echo ""
echo "  1. Activate environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run tiered GAN training:"
echo "     python -u brain/games/lolo/run_tiered_gan.py"
echo ""
echo "  3. After training, export weights:"
echo "     python cloud/export_weights.py"
echo ""
echo "  4. Download weights to local machine:"
echo "     scp -P PORT user@HOST:throng5/outputs/weights/*.npz ."
echo "═══════════════════════════════════════════════════"
