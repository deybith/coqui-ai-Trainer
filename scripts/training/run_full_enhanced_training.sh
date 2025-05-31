#!/bin/bash
"""
Full Enhanced XTTS Training Launcher

This script demonstrates how to train the full enhanced XTTS model
with both Phase 1 and Phase 2 enhancements.
"""

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p data/audio
mkdir -p data/metadata
mkdir -p output/checkpoints
mkdir -p output/logs

echo "🚀 Starting Full Enhanced XTTS Training"
echo "========================================"

# Check if data exists
if [ ! -f "data/metadata.txt" ]; then
    echo "❌ Error: Training data not found!"
    echo "Please prepare your training data with the following structure:"
    echo ""
    echo "data/"
    echo "├── audio/           # Audio files (.wav)"
    echo "│   ├── audio1.wav"
    echo "│   ├── audio2.wav"
    echo "│   └── ..."
    echo "└── metadata.txt     # Metadata file"
    echo ""
    echo "Metadata format (one line per audio file):"
    echo "audio1.wav|This is the transcription text|speaker_id"
    echo "audio2.wav|Another transcription text|speaker_id"
    echo ""
    exit 1
fi

# Phase 1 + Phase 2 Training (Full Enhanced)
echo "🎯 Training with Phase 1 + Phase 2 enhancements..."
python train_full_enhanced_xtts.py \
    --config config/full_enhanced_config.json \
    --use_phase1 \
    --use_phase2 \
    --output_dir ./output \
    --data_path ./data

# Alternative training configurations:

# Phase 2 only (Advanced Neural Architecture)
# echo "🧠 Training with Phase 2 only (Mamba, MoE, Flash Attention, RoPE)..."
# python train_full_enhanced_xtts.py \
#     --config config/full_enhanced_config.json \
#     --use_phase2 \
#     --output_dir ./output_phase2 \
#     --data_path ./data

# Phase 1 only (Neural Codec + Streaming)
# echo "🎵 Training with Phase 1 only (Neural Codec, Streaming)..."
# python train_full_enhanced_xtts.py \
#     --config config/full_enhanced_config.json \
#     --use_phase1 \
#     --output_dir ./output_phase1 \
#     --data_path ./data

echo "✅ Training completed! Check output directory for results."
