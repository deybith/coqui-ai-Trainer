#!/usr/bin/env python3
"""
Fixed XTTS Training Script with CSV Support
===========================================

This script fixes the path concatenation issue and uses the corrected dataset configuration
to train XTTS models with CSV metadata files.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trainer.xtts.gpt_trainer import train_gpt

def main():
    parser = argparse.ArgumentParser(description="Fixed XTTS Training with CSV Support")
    
    # Data arguments
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV file")
    parser.add_argument("--eval_csv", type=str, required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for trained model")
    
    # Training arguments
    parser.add_argument("--language", type=str, default="en",
                        help="Language code (en, es, fr, de, etc.)")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=30,
                        help="Maximum audio length in seconds")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.train_csv):
        print(f"❌ Training CSV not found: {args.train_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.eval_csv):
        print(f"❌ Evaluation CSV not found: {args.eval_csv}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    print("🚀 Starting XTTS Training with Fixed Path Configuration")
    print(f"📂 Training data: {args.train_csv}")
    print(f"📂 Evaluation data: {args.eval_csv}")
    print(f"📂 Output path: {args.output_path}")
    print(f"🌍 Language: {args.language}")
    print(f"📊 Epochs: {args.num_epochs}, Batch size: {args.batch_size}")
    
    # Convert max_audio_length from seconds to frames
    max_audio_length_frames = int(args.max_audio_length * 22050)
    
    try:
        # Call the fixed train_gpt function
        result = train_gpt(
            language=args.language,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            grad_acumm=args.grad_acumm,
            train_csv=args.train_csv,
            eval_csv=args.eval_csv,
            output_path=args.output_path,
            max_audio_length=max_audio_length_frames
        )
        
        print("✅ Training completed successfully!")
        print(f"📋 Results:")
        print(f"   - Config: {result[0]}")
        print(f"   - Checkpoint: {result[1]}")
        print(f"   - Vocab: {result[2]}")
        print(f"   - Output path: {result[3]}")
        print(f"   - Speaker reference: {result[4]}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
