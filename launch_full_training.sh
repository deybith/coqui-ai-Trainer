#!/bin/bash
# Full Enhanced XTTS Training Launcher
# This script trains XTTS with all Phase 1 + Phase 2 enhancements

echo "🚀 Starting Full Enhanced XTTS Training"
echo "======================================="

# Check if data exists
if [ ! -f "./data/demo_training/train.csv" ]; then
    echo "❌ Training data not found. Please prepare your data first."
    exit 1
fi

# Run full enhanced training
python train_full_enhanced_xtts.py \
    --config config/full_enhanced_config.json \
    --output_dir ./output/full_enhanced_training \
    --train_csv ./data/demo_training/train.csv \
    --eval_csv ./data/demo_training/eval.csv \
    --use_phase1 \
    --use_phase2 \
    --epochs 10 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_checkpointing \
    --mixed_precision \
    --tensorboard_logging

echo "✅ Training launched! Check ./output/full_enhanced_training for results."
