#!/bin/bash
# Quick test script for reconstruction pipeline
# Usage: ./test_reconstruction.sh

set -e

echo "=========================================="
echo "Testing 3D Reconstruction Pipeline"
echo "=========================================="

# Configuration
CONFIG="data/configs/base.yaml"
DATA_PATH="data/samples/retail_item/"
OUTPUT_PATH="data/output/test_reconstruction_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_PATH"
echo ""

# Check if Docker image exists
IMAGE_NAME=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "3d-object-reconstruction" | head -n 1)

if [ -z "$IMAGE_NAME" ]; then
    echo "Docker image not found. Building..."
    bash deploy/compose/deploy.sh setup
    IMAGE_NAME=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "3d-object-reconstruction" | head -n 1)
fi

echo "Using Docker image: $IMAGE_NAME"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run reconstruction
echo "Starting reconstruction..."
echo "=========================================="
docker run --rm \
    --gpus all \
    -v "$(pwd):/workspace/3d-object-reconstruction" \
    -w /workspace/3d-object-reconstruction \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=0 \
    "$IMAGE_NAME" \
    nvidia-3d-reconstruct \
        --config "/workspace/3d-object-reconstruction/$CONFIG" \
        --data-path "/workspace/3d-object-reconstruction/$DATA_PATH" \
        --output-path "/workspace/3d-object-reconstruction/$OUTPUT_PATH" \
        --verbose

echo ""
echo "=========================================="
echo "Reconstruction completed!"
echo "Output directory: $OUTPUT_PATH"
echo ""
echo "Output files:"
ls -lah "$OUTPUT_PATH" || true
echo "=========================================="

