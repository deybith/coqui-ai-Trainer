#!/usr/bin/env python3
"""
Quick inference test for the trained XTTS model.
"""

import os
import sys
import torch
import torchaudio
import json

def main():
    print("🎯 Quick Inference Test")
    print("=" * 30)
    
    # Paths
    training_dir = "output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d"
    config_path = os.path.join(training_dir, "config.json")
    
    # Import TTS
    print("Importing TTS modules...")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    print("✓ TTS modules imported")
    
    # Load config
    print("Loading config...")
    config = XttsConfig()
    config.load_json(config_path)
    print("✓ Config loaded")
    
    # Initialize model
    print("Initializing model...")
    model = Xtts.init_from_config(config)
    print("✓ Model initialized")
    
    # Find best checkpoint
    best_model_path = os.path.join(training_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(training_dir, "best_model_1619.pth")
    
    if not os.path.exists(best_model_path):
        print(f"❌ No checkpoint found")
        return False
    
    print(f"Loading checkpoint: {os.path.basename(best_model_path)}")
    
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ Checkpoint loaded")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    print("✓ Model ready for inference")
    
    # Find reference audio
    ref_audio_path = "data/dieck/dataset/wavs/audio10_00000000.wav"
    if not os.path.exists(ref_audio_path):
        print(f"❌ Reference audio not found: {ref_audio_path}")
        return False
    
    print(f"Using reference: {ref_audio_path}")
    
    # Load reference audio
    ref_audio, sr = torchaudio.load(ref_audio_path)
    
    # Ensure mono
    if ref_audio.shape[0] > 1:
        ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
    
    # Resample to 22050 if needed
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(sr, 22050)
        ref_audio = resampler(ref_audio)
    
    ref_audio = ref_audio.to(device)
    print("✓ Reference audio loaded")
    
    # Test synthesis
    text = "Hello, this is a test of the trained voice model."
    print(f"Synthesizing: '{text}'")
    
    try:
        with torch.no_grad():
            # Try inference method
            if hasattr(model, 'inference'):
                print("Using inference method...")
                outputs = model.inference(
                    text=text,
                    reference_wav=ref_audio,
                    language="en"
                )
            else:
                print("Using synthesize method...")
                outputs = model.synthesize(
                    text=text,
                    config=config,
                    speaker_wav=ref_audio,
                    language="en"
                )
        
        # Handle outputs
        if isinstance(outputs, dict):
            if 'wav' in outputs:
                audio = outputs['wav']
            elif 'audio' in outputs:
                audio = outputs['audio']
            else:
                audio = list(outputs.values())[0]
        else:
            audio = outputs
        
        # Save audio
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(0)
        
        output_path = "quick_test_output.wav"
        torchaudio.save(output_path, audio.cpu(), 22050)
        
        print(f"✓ Audio generated and saved to: {output_path}")
        print(f"✓ Audio shape: {audio.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Inference test successful!")
        else:
            print("\n❌ Inference test failed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
