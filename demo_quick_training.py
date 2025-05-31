#!/usr/bin/env python3
"""
Quick Demo: Enhanced XTTS Training Test
Creates minimal demo data and runs a quick training test
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path

def create_demo_data():
    """Create minimal demo data for testing"""
    print("🎯 Creating demo training data...")
    
    # Create demo data directory
    demo_dir = Path("./data/demo_training")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    wavs_dir = demo_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    # Create synthetic audio samples
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    num_samples = int(sample_rate * duration)
    
    # Generate 5 demo audio files with different synthetic voices
    audio_files = []
    transcripts = []
    
    for i in range(5):
        # Create synthetic audio (simple sine waves with different frequencies)
        freq = 200 + i * 50  # Different frequencies for different "speakers"
        t = torch.linspace(0, duration, num_samples)
        audio = 0.5 * torch.sin(2 * torch.pi * freq * t)
        
        # Add some noise for realism
        noise = 0.1 * torch.randn(num_samples)
        audio = audio + noise
        
        # Save audio file
        audio_path = wavs_dir / f"demo_{i:03d}.wav"
        torchaudio.save(str(audio_path), audio.unsqueeze(0), sample_rate)
        
        # Create transcript
        transcript = f"This is demo audio sample number {i+1} for testing enhanced XTTS training."
        
        audio_files.append(f"wavs/demo_{i:03d}.wav")
        transcripts.append(transcript)
    
    # Create metadata file
    metadata_path = demo_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        for i, (audio_file, transcript) in enumerate(zip(audio_files, transcripts)):
            speaker_id = f"demo_speaker_{i % 2}"  # 2 speakers
            f.write(f"{speaker_id}|{audio_file}|{transcript}\n")
    
    print(f"✅ Created {len(audio_files)} demo audio files in {demo_dir}")
    print(f"✅ Created metadata file: {metadata_path}")
    
    return str(demo_dir)

def create_demo_config(data_path):
    """Create optimized demo configuration"""
    print("🎯 Creating demo configuration...")
    
    demo_config = {
        "data_path": data_path,
        "model_name": "demo_enhanced_xtts",
        "use_phase1": True,
        "use_phase2": True,
        "use_neural_codec": True,
        "use_streaming": True,
        "use_quality_monitoring": True,
        "use_mamba": True,
        "use_flash_attention": True,
        "use_rope": True,
        "use_moe": True,
        
        # Smaller model for demo
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "max_text_tokens": 200,
        "max_mel_tokens": 300,
        "max_prompt_tokens": 50,
        
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "mamba_d_state": 8,
        
        # Demo training settings
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "validation_split": 0.2,
        "save_every_n_epochs": 1,
        
        "gradient_clip": 1.0,
        "weight_decay": 1e-5,
        "warmup_steps": 10,
        
        "optimizer": "adamw",
        "scheduler": "cosine_with_warmup",
        "mixed_precision": True,
        "gradient_checkpointing": True,
        
        "log_every_n_steps": 5,
        "validate_every_n_epochs": 1,
        "tensorboard_dir": "./output/demo_logs",
        
        "label_smoothing": 0.1,
        "dropout": 0.1,
        "use_deepspeed": False,
        "deepspeed_config": None
    }
    
    config_path = Path("./config/demo_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(demo_config, f, indent=2)
    
    print(f"✅ Created demo configuration: {config_path}")
    return str(config_path)

def run_demo_training():
    """Run complete demo training"""
    print("🚀 Starting Enhanced XTTS Training Demo")
    print("=" * 50)
    
    # Create demo data
    data_path = create_demo_data()
    
    # Create demo config
    config_path = create_demo_config(data_path)
    
    # Run training
    print("\n🎯 Starting training...")
    import subprocess
    
    cmd = [
        "python", "train_full_enhanced_xtts.py",
        "--config", config_path,
        "--use_phase1",
        "--use_phase2", 
        "--output_dir", "./output/demo_training"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Demo training completed successfully!")
        print("\n📊 Training outputs available in:")
        print("  - ./output/demo_training/checkpoints/")
        print("  - ./output/demo_training/logs/")
    else:
        print("❌ Demo training failed:")
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_demo_training()
    if success:
        print("\n🎉 Enhanced XTTS demo completed successfully!")
        print("🚀 Your system is ready for full training!")
    else:
        print("\n⚠️  Demo encountered issues. Check error messages above.")
