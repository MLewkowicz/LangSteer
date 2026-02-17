#!/bin/bash
# Setup script for CALVIN environment dependencies
# This script is SELF-CONTAINED and does not require the calvin/ reference repo

set -e

echo "========================================="
echo "Setting up CALVIN environment for LangSteer"
echo "========================================="

# 1. Install CALVIN package from pip
echo ""
echo "[1/4] Installing calvin-env package..."
uv pip install git+https://github.com/mees/calvin_env.git

# 2. Get Python environment paths
echo ""
echo "[2/4] Locating installed packages..."
VENV_PATH=$(uv run python -c "import sys; print(sys.prefix)")
PYTHON_VERSION=$(uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
CALVIN_PKG="$VENV_PATH/lib/python$PYTHON_VERSION/site-packages/calvin_env"
PYBULLET_DATA="$VENV_PATH/lib/python$PYTHON_VERSION/site-packages/pybullet_data"

echo "  Python version: $PYTHON_VERSION"
echo "  Virtual env: $VENV_PATH"
echo "  CALVIN package: $CALVIN_PKG"
echo "  PyBullet data: $PYBULLET_DATA"

# 3. Check if URDF files exist in installed package
echo ""
echo "[3/4] Checking CALVIN data files..."
if [ ! -d "$CALVIN_PKG/data/franka_panda" ]; then
    echo "  Data files not bundled in pip package. Cloning repo to get them..."
    TMP_DIR=$(mktemp -d)
    git clone --depth=1 https://github.com/mees/calvin_env.git "$TMP_DIR"
    mkdir -p "$CALVIN_PKG/data"
    cp -r "$TMP_DIR/data/"* "$CALVIN_PKG/data/"
    rm -rf "$TMP_DIR"
    echo "  ✓ CALVIN data files installed"
else
    echo "  ✓ CALVIN data files found"
fi

# 4. Copy franka_panda URDF to PyBullet data directory
echo ""
echo "[4/4] Setting up PyBullet URDF paths..."
if [ -d "$PYBULLET_DATA" ]; then
    echo "  Copying Franka Panda URDFs to PyBullet data directory..."
    rm -rf "$PYBULLET_DATA/franka_panda"
    cp -r "$CALVIN_PKG/data/franka_panda" "$PYBULLET_DATA/"
    echo "  ✓ URDFs copied to: $PYBULLET_DATA/franka_panda"
else
    echo "  WARNING: PyBullet data directory not found at $PYBULLET_DATA"
    echo "  URDFs will be loaded via monkey-patch (see gym_wrapper.py)"
fi

echo ""
echo "========================================="
echo "✓ CALVIN setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Download CALVIN dataset:"
echo "     wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
echo "     unzip task_D_D.zip -d /path/to/datasets/"
echo ""
echo "  2. Set environment variables:"
echo "     export CALVIN_DATASET_PATH=/path/to/datasets/task_D_D"
echo "     export DP3_CHECKPOINT_PATH=/path/to/your/checkpoint.ckpt"
echo ""
echo "  3. Run inference:"
echo "     uv run python scripts/run_experiment.py experiment=dp3_calvin_inference"
echo ""
echo "  4. (Optional) Test with visualization:"
echo "     uv run python scripts/run_experiment.py experiment=dp3_calvin_inference \\"
echo "       env.use_gui=true experiment.num_episodes=1"
echo ""
