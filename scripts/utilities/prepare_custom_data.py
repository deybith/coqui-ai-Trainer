#!/usr/bin/env python3
"""
Prepare Your Own XTTS Training Data

This script helps you prepare your own audio data for XTTS training.
"""

import os
import argparse
from pathlib import Path
import pandas as pd

def prepare_custom_data(audio_dir, output_dir, speaker_name="custom_speaker"):
    """
    Prepare custom audio data for XTTS training.
    
    Args:
        audio_dir: Directory containing your WAV files
        output_dir: Directory to save prepared data
        speaker_name: Name for your speaker
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(audio_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"❌ No WAV files found in {audio_dir}")
        return False
    
    print(f"📁 Found {len(wav_files)} WAV files")
    
    # Create metadata
    metadata_lines = []
    train_data = []
    eval_data = []
    
    for i, wav_file in enumerate(wav_files):
        # For demo, we'll use the filename as text
        # In real training, you'd have transcriptions
        text = f"Audio sample number {i+1} from {speaker_name}"
        
        relative_path = f"wavs/{wav_file.name}"
        metadata_line = f"{speaker_name}|{relative_path}|{text}"
        metadata_lines.append(metadata_line)
        
        # Split 80% train, 20% eval
        if i < len(wav_files) * 0.8:
            train_data.append({
                "audio_file": str(wav_file),
                "text": text,
                "speaker_name": speaker_name
            })
        else:
            eval_data.append({
                "audio_file": str(wav_file),
                "text": text,
                "speaker_name": speaker_name
            })
    
    # Copy WAV files to output directory
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    for wav_file in wav_files:
        import shutil
        shutil.copy2(wav_file, wavs_dir / wav_file.name)
    
    # Save metadata.txt
    with open(output_dir / "metadata.txt", "w") as f:
        f.write("\n".join(metadata_lines))
    
    # Save CSV files
    pd.DataFrame(train_data).to_csv(output_dir / "train.csv", index=False)
    pd.DataFrame(eval_data).to_csv(output_dir / "eval.csv", index=False)
    
    print(f"✅ Prepared {len(train_data)} training samples")
    print(f"✅ Prepared {len(eval_data)} evaluation samples")
    print(f"✅ Data saved to {output_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare custom XTTS training data")
    parser.add_argument("--audio_dir", required=True, help="Directory containing WAV files")
    parser.add_argument("--output_dir", required=True, help="Output directory for prepared data")
    parser.add_argument("--speaker_name", default="custom_speaker", help="Speaker name")
    
    args = parser.parse_args()
    
    print("🎙️  Preparing Custom XTTS Training Data")
    print("=====================================")
    
    success = prepare_custom_data(args.audio_dir, args.output_dir, args.speaker_name)
    
    if success:
        print("\n🚀 Ready to train! Use this command:")
        print(f"python examples/train_xtts_enhanced.py \\")
        print(f"    --output_path ./output/custom_training \\")
        print(f"    --train_csv {args.output_dir}/train.csv \\")
        print(f"    --eval_csv {args.output_dir}/eval.csv \\")
        print(f"    --language en \\")
        print(f"    --batch_size 2 \\")
        print(f"    --epochs 10")

if __name__ == "__main__":
    main()
