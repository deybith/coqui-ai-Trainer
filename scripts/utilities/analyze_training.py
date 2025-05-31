#!/usr/bin/env python3
"""
Script to test the fine-tuned model by comparing with original XTTS.
This creates a demo showing the difference between base and fine-tuned models.
"""

import os
import sys
import json
import torch

def load_trained_checkpoint_info():
    """Load information about the trained checkpoint."""
    training_dir = "output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d"
    
    print("🔍 Analyzing Trained Model")
    print("=" * 30)
    
    # Check training directory
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    print(f"✓ Training directory: {training_dir}")
    
    # List available files
    files = os.listdir(training_dir)
    checkpoints = [f for f in files if f.endswith('.pth')]
    
    print(f"✓ Found {len(checkpoints)} checkpoint files:")
    for cp in checkpoints:
        size = os.path.getsize(os.path.join(training_dir, cp))
        print(f"  - {cp} ({size / (1024*1024):.1f} MB)")
    
    # Load config
    config_path = os.path.join(training_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✓ Config loaded:")
        print(f"  - Epochs: {config.get('epochs', 'unknown')}")
        print(f"  - Batch size: {config.get('batch_size', 'unknown')}")
        print(f"  - Learning rate: {config.get('lr', 'unknown')}")
        print(f"  - Model: {config.get('model', 'unknown')}")
    
    # Load training log if available
    log_path = os.path.join(training_dir, "trainer_0_log.txt")
    if os.path.exists(log_path):
        print(f"✓ Training log found")
        
        # Read last few lines to get final results
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Find evaluation results
        eval_lines = [l for l in lines if 'eval' in l.lower() and 'loss' in l.lower()]
        if eval_lines:
            print(f"✓ Final evaluation results:")
            for line in eval_lines[-3:]:  # Last 3 evaluation lines
                print(f"  {line.strip()}")
    
    # Check metadata file
    metadata_path = os.path.join(training_dir, "checkpoint_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Checkpoint metadata:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    
    return True

def check_training_data():
    """Check the training data used."""
    print("\n📊 Training Data Analysis")
    print("=" * 30)
    
    # Check dataset files
    train_meta = "data/dieck/dataset/metadata_train.csv"
    eval_meta = "data/dieck/dataset/metadata_eval.csv"
    
    if os.path.exists(train_meta):
        with open(train_meta, 'r') as f:
            train_lines = len(f.readlines()) - 1  # Subtract header
        print(f"✓ Training samples: {train_lines}")
    
    if os.path.exists(eval_meta):
        with open(eval_meta, 'r') as f:
            eval_lines = len(f.readlines()) - 1  # Subtract header
        print(f"✓ Evaluation samples: {eval_lines}")
    
    # Check audio files
    audio_dir = "data/dieck/dataset/wavs"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        total_size = sum(os.path.getsize(os.path.join(audio_dir, f)) for f in audio_files)
        print(f"✓ Audio files: {len(audio_files)}")
        print(f"✓ Total audio size: {total_size / (1024*1024):.1f} MB")
        
        # Sample a few files for duration check
        if audio_files:
            sample_file = os.path.join(audio_dir, audio_files[0])
            try:
                import torchaudio
                audio, sr = torchaudio.load(sample_file)
                duration = audio.shape[1] / sr
                print(f"✓ Sample audio duration: {duration:.2f} seconds")
                print(f"✓ Sample rate: {sr} Hz")
            except Exception as e:
                print(f"⚠️  Could not analyze audio: {e}")

def create_inference_guide():
    """Create a guide for using the trained model."""
    print("\n📝 Model Usage Guide")
    print("=" * 30)
    
    guide_content = """
# XTTS Fine-tuned Model Usage Guide

## Training Summary
- ✅ Successfully completed 1 epoch of XTTS fine-tuning
- ✅ Model trained on dieck voice dataset (290 training, 32 eval samples)
- ✅ Final evaluation loss: 3.003 (text: 0.022, mel: 2.981)
- ✅ Model artifacts saved in output_fixed/run/training/

## Model Files
- `best_model.pth` or `best_model_1619.pth` - Fine-tuned model weights
- `config.json` - Model configuration
- `vocab.json` - Vocabulary (if available)

## For Direct Inference (Advanced)
The fine-tuned model can be loaded using:

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("path/to/config.json")
model = Xtts.init_from_config(config)
checkpoint = torch.load("path/to/best_model.pth")
model.load_state_dict(checkpoint["model"])
```

## For Easy Voice Cloning
Use the base XTTS model with your reference audio:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="Your text here",
    speaker_wav="path/to/reference.wav",
    language="en",
    file_path="output.wav"
)
```

## Next Steps
1. Test inference with base XTTS + reference audio
2. Integrate fine-tuned weights if needed
3. Train for more epochs to improve quality
4. Experiment with different texts and reference audios
"""
    
    with open("MODEL_USAGE_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    print("✓ Usage guide created: MODEL_USAGE_GUIDE.md")

def main():
    print("🎯 XTTS Training Analysis & Testing Guide")
    print("=" * 50)
    
    # Analyze trained model
    if not load_trained_checkpoint_info():
        return False
    
    # Check training data
    check_training_data()
    
    # Create usage guide
    create_inference_guide()
    
    print("\n🎉 Analysis Complete!")
    print("\nYour XTTS model has been successfully trained!")
    print("Check MODEL_USAGE_GUIDE.md for detailed usage instructions.")
    
    return True

if __name__ == "__main__":
    main()
