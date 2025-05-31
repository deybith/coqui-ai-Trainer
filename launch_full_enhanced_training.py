#!/usr/bin/env python3
"""
Launch Full Enhanced XTTS Training

This script launches training with all Phase 1 + Phase 2 enhancements
"""

import subprocess
import sys
from pathlib import Path

def run_full_training():
    """Run full enhanced training with all features."""
    print("🚀 Launching Full Enhanced XTTS Training")
    print("=" * 50)
    
    # Training command
    cmd = [
        sys.executable, "train_full_enhanced_xtts.py",
        "--config", "config/full_enhanced_config.json",
        "--output_dir", "./output/full_enhanced_training",
        "--train_csv", "./data/demo_training/train.csv", 
        "--eval_csv", "./data/demo_training/eval.csv",
        "--use_phase1",
        "--use_phase2", 
        "--epochs", "5",
        "--batch_size", "2",
        "--learning_rate", "1e-4",
        "--gradient_checkpointing",
        "--mixed_precision",
        "--tensorboard_logging"
    ]
    
    print("🏃 Running command:")
    print(" ".join(cmd))
    print()
    
    # Run training
    result = subprocess.run(cmd, cwd="/home/ubuntu/projects/coqui-ai-Trainer")
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        print("📁 Check ./output/full_enhanced_training for results")
    else:
        print("❌ Training failed. Check logs above.")
    
    return result.returncode

if __name__ == "__main__":
    run_full_training()
