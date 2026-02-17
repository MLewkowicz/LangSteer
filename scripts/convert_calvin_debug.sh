#!/bin/bash
# Convert CALVIN debug dataset to Zarr format for DP3 training
# Adapt paths for your setup

# Configuration
ROOT_DIR="${CALVIN_ROOT:-/home/mlewkowicz/calvin/dataset/calvin_debug_dataset}"
SAVE_PATH="${ZARR_OUTPUT:-data/calvin_debug.zarr}"

echo "================================================"
echo "Converting CALVIN Dataset to Zarr Format"
echo "================================================"
echo "Root directory: $ROOT_DIR"
echo "Save path: $SAVE_PATH"
echo ""

# Check if root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: CALVIN dataset not found at $ROOT_DIR"
    echo "Please set CALVIN_ROOT environment variable or update ROOT_DIR in this script"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$SAVE_PATH")
mkdir -p "$OUTPUT_DIR"

# Run conversion
python scripts/convert_calvin_to_zarr.py \
    --root_dir "$ROOT_DIR" \
    --save_path "$SAVE_PATH" \
    --overwrite

echo ""
echo "================================================"
echo "Conversion complete!"
echo "Zarr dataset saved to: $SAVE_PATH"
echo "================================================"
echo ""
echo "To use for training, set:"
echo "  export CALVIN_ZARR_PATH=$SAVE_PATH"
echo "  python scripts/train_dp3.py training=dp3_debug"
