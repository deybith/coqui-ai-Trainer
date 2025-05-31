#!/usr/bin/env python3
"""
Train XTTS with Your Own Data

This script guides you through training XTTS with your own audio files.
"""

import os
import sys
from pathlib import Path

def setup_custom_training():
    """Guide user through custom training setup."""
    print("🎙️  XTTS Custom Training Setup")
    print("=" * 40)
    print()
    
    print("📋 Step 1: Prepare your data")
    print("Your audio files should be:")
    print("  • WAV format, 22050Hz sample rate")
    print("  • Clear speech, minimal background noise") 
    print("  • 5-30 seconds per file")
    print("  • At least 10-50 files for good results")
    print()
    
    print("📝 Step 2: Organize your files")
    print("Put all WAV files in a single directory, then run:")
    print("python prepare_custom_data.py \\")
    print("    --audio_dir /path/to/your/wav/files \\")
    print("    --output_dir ./data/my_training \\") 
    print("    --speaker_name \"my_speaker\"")
    print()
    
    print("🚀 Step 3: Start training")
    print("python examples/train_xtts_enhanced.py \\")
    print("    --output_path ./output/my_training \\")
    print("    --train_csv ./data/my_training/train.csv \\")
    print("    --eval_csv ./data/my_training/eval.csv \\")
    print("    --language en \\")
    print("    --batch_size 2 \\")
    print("    --epochs 10")
    print()
    
    print("💡 Tips for best results:")
    print("  • Use consistent audio quality")
    print("  • Include variety in speech content")
    print("  • Monitor training with TensorBoard")
    print("  • Start with fewer epochs, increase if needed")
    print()
    
    print("🔧 Advanced options:")
    print("  • Use --max_audio_length to control memory usage")
    print("  • Use --speaker_reference for voice cloning")
    print("  • Check ./config/ for advanced configurations")

if __name__ == "__main__":
    setup_custom_training()
