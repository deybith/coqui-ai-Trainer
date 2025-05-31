#!/usr/bin/env python3
"""
Test script for the trained XTTS model.
This script loads the trained model and generates sample audio.
"""

import os
import sys
import torch
import torchaudio
import json
import argparse
from pathlib import Path
import traceback

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    print("✓ Successfully imported TTS modules")
except ImportError as e:
    print(f"❌ Error importing TTS modules: {e}")
    print("Please ensure TTS is properly installed")
    sys.exit(1)

def find_latest_checkpoint(training_dir):
    """Find the latest/best checkpoint in the training directory."""
    checkpoint_files = []
    
    # Look for best_model.pth first
    best_model_path = os.path.join(training_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        return best_model_path
    
    # Look for best_model_*.pth
    for file in os.listdir(training_dir):
        if file.startswith("best_model_") and file.endswith(".pth"):
            checkpoint_files.append((file, os.path.join(training_dir, file)))
    
    if checkpoint_files:
        # Sort by modification time and return the latest
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        return checkpoint_files[0][1]
    
    # Look for regular checkpoints
    for file in os.listdir(training_dir):
        if file.startswith("checkpoint_") and file.endswith(".pth"):
            step = int(file.split("_")[1].split(".")[0])
            checkpoint_files.append((step, os.path.join(training_dir, file)))
    
    if checkpoint_files:
        # Sort by step number and return the highest
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_files[0][1]
    
    return None

def load_model(config_path, checkpoint_path, vocab_path=None):
    """Load the trained XTTS model."""
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load config
    config = XttsConfig()
    config.load_json(config_path)
    
    # Initialize model
    model = Xtts.init_from_config(config)
    
    # Load checkpoint
    print("Loading model weights...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    # Load vocabulary if available
    if vocab_path and os.path.exists(vocab_path):
        print(f"Loading vocabulary from: {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        # Set vocabulary in model if it has this attribute
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'vocab'):
            model.tokenizer.vocab = vocab
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    return model, config, device

def find_reference_audio(data_dir):
    """Find a reference audio file for voice cloning."""
    audio_dir = os.path.join(data_dir, "dieck", "dataset")
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return None
    
    # Look for wav files
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            return os.path.join(audio_dir, file)
    
    return None

def synthesize_speech(model, config, device, text, reference_audio_path, output_path, language="en"):
    """Synthesize speech using the trained model."""
    print(f"Synthesizing text: '{text}'")
    print(f"Using reference audio: {reference_audio_path}")
    print(f"Language: {language}")
    
    try:
        # Load reference audio
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        
        # Load and preprocess reference audio
        reference_audio, sr = torchaudio.load(reference_audio_path)
        
        # Ensure mono
        if reference_audio.shape[0] > 1:
            reference_audio = torch.mean(reference_audio, dim=0, keepdim=True)
        
        # Resample if necessary (XTTS typically uses 22050 Hz)
        target_sr = getattr(config.audio, 'sample_rate', 22050)
        if sr != target_sr:
            print(f"Resampling reference audio from {sr} Hz to {target_sr} Hz")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            reference_audio = resampler(reference_audio)
        
        # Move to device
        reference_audio = reference_audio.to(device)
        
        print("Generating speech...")
        
        # Generate speech
        with torch.no_grad():
            # Different models might have different inference methods
            if hasattr(model, 'synthesize'):
                # Method 1: Direct synthesize method
                outputs = model.synthesize(
                    text=text,
                    config=config,
                    speaker_wav=reference_audio,
                    language=language
                )
            elif hasattr(model, 'inference'):
                # Method 2: Inference method
                outputs = model.inference(
                    text=text,
                    reference_wav=reference_audio,
                    language=language
                )
            else:
                # Method 3: Forward pass (might need different parameters)
                print("Using forward pass method...")
                # This might need adjustment based on the model's forward method
                outputs = model(
                    text=text,
                    speaker_wav=reference_audio,
                    language=language
                )
        
        # Extract audio from outputs
        if isinstance(outputs, dict):
            if 'wav' in outputs:
                audio = outputs['wav']
            elif 'audio' in outputs:
                audio = outputs['audio']
            else:
                # Take the first tensor value
                audio = list(outputs.values())[0]
        else:
            audio = outputs
        
        # Ensure audio is a tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        
        # Ensure correct shape for saving
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
        elif audio.dim() == 3:
            audio = audio.squeeze(0)   # Remove batch dimension if present
        
        # Save audio
        print(f"Saving audio to: {output_path}")
        torchaudio.save(output_path, audio.cpu(), target_sr)
        
        print("✓ Speech synthesis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test trained XTTS model")
    parser.add_argument("--training_dir", type=str, 
                       default="output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d",
                       help="Path to training output directory")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Path to data directory")
    parser.add_argument("--text", type=str, 
                       default="Hello, this is a test of the trained voice cloning model.",
                       help="Text to synthesize")
    parser.add_argument("--output", type=str, default="test_output.wav",
                       help="Output audio file path")
    parser.add_argument("--language", type=str, default="en",
                       help="Language code")
    parser.add_argument("--reference_audio", type=str, default=None,
                       help="Path to reference audio file (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    print("🎯 Testing Trained XTTS Model")
    print("=" * 50)
    
    # Check if training directory exists
    if not os.path.exists(args.training_dir):
        print(f"❌ Training directory not found: {args.training_dir}")
        return False
    
    # Find config file
    config_path = os.path.join(args.training_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(args.training_dir)
    if not checkpoint_path:
        print(f"❌ No checkpoint found in: {args.training_dir}")
        return False
    
    print(f"Found checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Find vocabulary file
    vocab_path = os.path.join(args.training_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        # Look in original model files
        original_model_dir = "output_fixed/run/training/XTTS_v2.0_original_model_files"
        vocab_path = os.path.join(original_model_dir, "vocab.json")
        if not os.path.exists(vocab_path):
            vocab_path = None
    
    if vocab_path:
        print(f"Found vocabulary: {vocab_path}")
    else:
        print("⚠️  No vocabulary file found, using default")
    
    try:
        # Load model
        print("\n📦 Loading model...")
        model, config, device = load_model(config_path, checkpoint_path, vocab_path)
        print("✓ Model loaded successfully!")
        
        # Find reference audio
        if args.reference_audio:
            reference_audio_path = args.reference_audio
        else:
            reference_audio_path = find_reference_audio(args.data_dir)
        
        if not reference_audio_path:
            print("❌ No reference audio found. Please provide --reference_audio")
            return False
        
        print(f"Using reference audio: {reference_audio_path}")
        
        # Create output directory
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Synthesize speech
        print("\n🎤 Synthesizing speech...")
        success = synthesize_speech(
            model=model,
            config=config,
            device=device,
            text=args.text,
            reference_audio_path=reference_audio_path,
            output_path=args.output,
            language=args.language
        )
        
        if success:
            print(f"\n🎉 Success! Generated audio saved to: {args.output}")
            print("\nTo play the audio, you can use:")
            print(f"  ffplay {args.output}")
            print(f"  or open the file in your audio player")
            return True
        else:
            print("\n❌ Synthesis failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
