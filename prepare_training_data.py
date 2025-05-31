#!/usr/bin/env python3
"""
Data Preparation Helper for Full Enhanced XTTS Training
"""

import os
import sys
import argparse
from pathlib import Path
import torchaudio
import torch

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Full Enhanced XTTS training")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw audio files")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for prepared data")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--create_metadata", action="store_true", help="Create basic metadata from filenames")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directories
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Preparing XTTS training data")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    
    # Find audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
        audio_files.extend(input_dir.glob(ext))
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Process audio files
    for i, audio_file in enumerate(audio_files):
        try:
            # Load and process audio
            audio, sr = torchaudio.load(audio_file)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != args.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, args.sample_rate)
                audio = resampler(audio)
            
            # Save processed audio
            output_file = audio_output_dir / f"{audio_file.stem}.wav"
            torchaudio.save(output_file, audio, args.sample_rate)
            
            print(f"✅ Processed ({i+1}/{len(audio_files)}): {audio_file.name}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
    
    # Create metadata if requested
    if args.create_metadata:
        metadata_file = output_dir / "metadata.txt"
        with open(metadata_file, 'w') as f:
            for audio_file in audio_output_dir.glob('*.wav'):
                speaker = audio_file.stem.split('_')[0] if '_' in audio_file.stem else "default"
                text_hint = audio_file.stem.replace('_', ' ').replace('-', ' ')
                f.write(f"{audio_file.name}|{text_hint}|{speaker}\n")
        
        print(f"✅ Created metadata file: {metadata_file}")
        print("⚠️  Please edit the metadata file to add proper transcriptions!")

if __name__ == "__main__":
    main()
