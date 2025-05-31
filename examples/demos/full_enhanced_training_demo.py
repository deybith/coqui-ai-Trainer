#!/usr/bin/env python3
"""
Full Enhanced XTTS Training Demo

This script demonstrates the complete training pipeline for the full enhanced XTTS model
with both Phase 1 and Phase 2 enhancements.
"""

import os
import sys
import json
import torch
from pathlib import Path

def check_system_requirements():
    """Check if system meets requirements for training."""
    print("🔍 Checking System Requirements...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA GPU: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("⚠️  Warning: Less than 8GB GPU memory. Consider reducing batch size.")
    else:
        print("❌ CUDA not available. GPU training recommended for enhanced XTTS.")
        return False
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"✅ PyTorch: {torch_version}")
    
    if not torch_version.startswith('2.'):
        print("⚠️  Warning: PyTorch 2.0+ recommended for optimal performance.")
    
    return True

def check_enhanced_components():
    """Check if enhanced components are available."""
    print("\n🔍 Checking Enhanced Components...")
    
    phase1_available = False
    phase2_available = False
    
    # Check Phase 1 components
    try:
        from trainer.xtts.models.enhanced_xtts import EnhancedXtts
        from trainer.xtts.layers.encodec import create_encodec_for_xtts
        phase1_available = True
        print("✅ Phase 1 components available")
    except ImportError as e:
        print(f"❌ Phase 1 components not available: {e}")
    
    # Check Phase 2 components
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedXTTSGPT
        from trainer.xtts.layers.attention.phase2_integration import Phase2Config
        phase2_available = True
        print("✅ Phase 2 components available")
    except ImportError as e:
        print(f"❌ Phase 2 components not available: {e}")
    
    return phase1_available, phase2_available

def create_sample_dataset():
    """Create a minimal sample dataset for demonstration."""
    print("\n📊 Creating Sample Dataset...")
    
    data_dir = Path("./demo_data")
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy audio files (for demo purposes)
    sample_rate = 22050
    duration = 3  # 3 seconds
    
    for i in range(5):
        # Generate simple sine wave as dummy audio
        import numpy as np
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440 + i * 100  # Different frequencies
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Save as wav file
        audio_file = audio_dir / f"demo_sample_{i:03d}.wav"
        import torchaudio
        torchaudio.save(audio_file, torch.tensor(audio).unsqueeze(0), sample_rate)
    
    # Create metadata file
    metadata_file = data_dir / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("demo_sample_000.wav|Hello, this is a demonstration of enhanced XTTS training.|demo_speaker\n")
        f.write("demo_sample_001.wav|The weather is nice today for machine learning.|demo_speaker\n")
        f.write("demo_sample_002.wav|Artificial intelligence is advancing rapidly.|demo_speaker\n")
        f.write("demo_sample_003.wav|Text to speech synthesis is fascinating.|demo_speaker\n")
        f.write("demo_sample_004.wav|This is the final sample for our demo.|demo_speaker\n")
    
    print(f"✅ Sample dataset created at: {data_dir}")
    print(f"   - 5 audio samples")
    print(f"   - Metadata file with transcriptions")
    
    return str(data_dir)

def create_demo_config(data_path: str, phase1_available: bool, phase2_available: bool):
    """Create a demo configuration for training."""
    print("\n⚙️  Creating Demo Configuration...")
    
    config = {
        "data_path": data_path,
        "model_name": "demo_enhanced_xtts",
        "use_phase1": phase1_available,
        "use_phase2": phase2_available,
        "use_neural_codec": phase1_available,
        "use_streaming": phase1_available,
        "use_quality_monitoring": phase1_available,
        "use_mamba": phase2_available,
        "use_flash_attention": phase2_available,
        "use_rope": phase2_available,
        "use_moe": phase2_available,
        
        # Small model for demo
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "max_text_tokens": 100,
        "max_mel_tokens": 200,
        "max_prompt_tokens": 50,
        
        # MoE settings (if enabled)
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "mamba_d_state": 8,
        
        # Training settings for demo
        "batch_size": 1,
        "learning_rate": 1e-3,
        "num_epochs": 5,
        "validation_split": 0.2,
        "save_every_n_epochs": 2,
        
        "gradient_clip": 1.0,
        "weight_decay": 1e-5,
        "warmup_steps": 10,
        
        "optimizer": "adamw",
        "scheduler": "cosine_with_warmup",
        "mixed_precision": True,
        "gradient_checkpointing": True,
        
        "log_every_n_steps": 5,
        "validate_every_n_epochs": 2,
        "tensorboard_dir": "./demo_logs",
        
        "label_smoothing": 0.1,
        "dropout": 0.1,
        "use_deepspeed": False
    }
    
    # Save demo config
    config_path = "demo_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Demo configuration created: {config_path}")
    print(f"   - Phase 1 enabled: {phase1_available}")
    print(f"   - Phase 2 enabled: {phase2_available}")
    print(f"   - Small model size for quick demo")
    
    return config_path

def run_training_demo(config_path: str):
    """Run the training demonstration."""
    print("\n🚀 Starting Training Demonstration...")
    
    try:
        # Import the training script
        sys.path.append('.')
        
        # Run training with the demo config
        cmd = f"python train_full_enhanced_xtts.py --config {config_path} --output_dir ./demo_output"
        
        print(f"📝 Running command: {cmd}")
        print("⏰ This may take a few minutes...")
        
        # Execute training
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training demo completed successfully!")
            print("📁 Check ./demo_output for results")
        else:
            print("❌ Training demo failed:")
            print(result.stderr)
            return False
        
    except Exception as e:
        print(f"❌ Training demo error: {e}")
        return False
    
    return True

def cleanup_demo():
    """Clean up demo files."""
    import shutil
    
    print("\n🧹 Cleaning up demo files...")
    
    # Remove demo data
    if Path("./demo_data").exists():
        shutil.rmtree("./demo_data")
        print("✅ Removed demo_data/")
    
    # Remove demo config
    if Path("demo_config.json").exists():
        os.remove("demo_config.json")
        print("✅ Removed demo_config.json")
    
    # Remove demo output (optional - you might want to keep this)
    # if Path("./demo_output").exists():
    #     shutil.rmtree("./demo_output")
    #     print("✅ Removed demo_output/")

def show_next_steps():
    """Show next steps for actual training."""
    print("\n🎯 Next Steps for Actual Training:")
    print("=" * 50)
    print()
    print("1. 📊 Prepare Your Data:")
    print("   python prepare_training_data.py --input_dir /path/to/your/audio --output_dir ./data")
    print()
    print("2. ⚙️  Edit Configuration:")
    print("   - Edit config/full_enhanced_config.json")
    print("   - Adjust model size based on your GPU memory")
    print("   - Set appropriate batch_size and learning_rate")
    print()
    print("3. 🚀 Start Training:")
    print("   python train_full_enhanced_xtts.py \\")
    print("       --config config/full_enhanced_config.json \\")
    print("       --use_phase1 --use_phase2 \\")
    print("       --data_path ./data \\")
    print("       --output_dir ./output")
    print()
    print("4. 📈 Monitor Progress:")
    print("   tensorboard --logdir ./output/logs")
    print()
    print("5. 📚 Documentation:")
    print("   - Read FULL_ENHANCED_TRAINING_GUIDE.md for detailed instructions")
    print("   - Check PHASE2_COMPLETION_REPORT.md for feature details")
    print()

def main():
    """Main demonstration function."""
    print("🎯 Full Enhanced XTTS Training Demonstration")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Exiting.")
        return
    
    # Check enhanced components
    phase1_available, phase2_available = check_enhanced_components()
    
    if not phase1_available and not phase2_available:
        print("❌ No enhanced components available. Please install Phase 1 or Phase 2 components.")
        return
    
    # Create sample dataset
    data_path = create_sample_dataset()
    
    # Create demo configuration
    config_path = create_demo_config(data_path, phase1_available, phase2_available)
    
    # Ask user if they want to run the demo
    print("\n❓ Do you want to run the training demonstration? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        # Run training demo
        success = run_training_demo(config_path)
        
        if success:
            print("\n🎉 Training demonstration completed successfully!")
            print("📁 Results saved in ./demo_output/")
        else:
            print("\n❌ Training demonstration failed.")
    else:
        print("\n⏭️  Skipping training demonstration.")
    
    # Clean up
    print("\n❓ Clean up demo files? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        cleanup_demo()
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Demonstration complete!")

if __name__ == "__main__":
    main()
