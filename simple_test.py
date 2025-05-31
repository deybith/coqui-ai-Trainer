#!/usr/bin/env python3
"""
Simple test to check if we can load the trained model.
"""

import os
import sys
import torch
import json

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🎯 Simple Model Loading Test")
    print("=" * 30)
    
    # Check if training directory exists
    training_dir = "output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d"
    
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    print(f"✓ Training directory found: {training_dir}")
    
    # Check config file
    config_path = os.path.join(training_dir, "config.json")
    if os.path.exists(config_path):
        print(f"✓ Config file found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Config loaded successfully")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return False
    else:
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Check checkpoint files
    checkpoints = []
    for file in os.listdir(training_dir):
        if file.endswith('.pth'):
            checkpoints.append(file)
    
    if checkpoints:
        print(f"✓ Found {len(checkpoints)} checkpoint(s): {', '.join(checkpoints)}")
    else:
        print("❌ No checkpoint files found")
        return False
    
    # Test importing TTS
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        print("✓ TTS modules imported successfully")
    except ImportError as e:
        print(f"❌ Error importing TTS: {e}")
        return False
    
    # Test loading config
    try:
        xtts_config = XttsConfig()
        xtts_config.load_json(config_path)
        print("✓ XTTS config loaded successfully")
    except Exception as e:
        print(f"❌ Error loading XTTS config: {e}")
        return False
    
    # Test model initialization
    try:
        model = Xtts.init_from_config(xtts_config)
        print("✓ XTTS model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing XTTS model: {e}")
        return False
    
    # Test loading checkpoint
    try:
        best_checkpoint = os.path.join(training_dir, "best_model.pth")
        if not os.path.exists(best_checkpoint):
            # Try other checkpoints
            for checkpoint in checkpoints:
                if "best_model" in checkpoint:
                    best_checkpoint = os.path.join(training_dir, checkpoint)
                    break
        
        if os.path.exists(best_checkpoint):
            print(f"Loading checkpoint: {best_checkpoint}")
            checkpoint = torch.load(best_checkpoint, map_location="cpu")
            
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                print(f"✓ Model loaded from checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                model.load_state_dict(checkpoint)
                print("✓ Model weights loaded from checkpoint")
        else:
            print("❌ No suitable checkpoint found")
            return False
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, will use CPU")
    
    print("\n🎉 All basic tests passed! Model is ready for inference.")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nNext steps:")
        print("1. Run synthesis test: python test_trained_model.py")
        print("2. Try different texts and reference audios")
        print("3. Consider training for more epochs if needed")
    else:
        print("\n❌ Basic tests failed. Please check the issues above.")
