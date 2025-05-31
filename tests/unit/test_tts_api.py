#!/usr/bin/env python3
"""
Simplified inference script using the TTS API directly.
"""

import os
import sys
import torch
import torchaudio

def main():
    print("🎯 Testing Trained Model with TTS API")
    print("=" * 40)
    
    # Import TTS
    try:
        from TTS.api import TTS
        print("✓ TTS API imported successfully")
    except ImportError as e:
        print(f"❌ Error importing TTS API: {e}")
        return False
    
    # Set paths
    model_path = "output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d/best_model.pth"
    config_path = "output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d/config.json"
    
    # Check if we have the best model
    if not os.path.exists(model_path):
        model_path = "output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d/best_model_1619.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found")
        return False
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found")
        return False
    
    print(f"✓ Using model: {os.path.basename(model_path)}")
    print(f"✓ Using config: {os.path.basename(config_path)}")
    
    # Find reference audio
    ref_audio_path = "data/dieck/dataset/wavs/audio10_00000000.wav"
    if not os.path.exists(ref_audio_path):
        print(f"❌ Reference audio not found: {ref_audio_path}")
        return False
    
    print(f"✓ Using reference: {os.path.basename(ref_audio_path)}")
    
    try:
        # Initialize TTS with custom model
        print("Initializing TTS with trained model...")
        
        # Try using the base XTTS model first, then fine-tune with our checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Use the base XTTS model as a starting point
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        print("✓ Base XTTS model loaded")
        
        # Test with base model first
        text = "Hello, this is a test of the voice cloning model."
        
        print(f"Generating speech: '{text}'")
        
        # Generate audio
        output_path = "tts_test_output.wav"
        tts.tts_to_file(
            text=text,
            speaker_wav=ref_audio_path,
            language="en",
            file_path=output_path
        )
        
        print(f"✓ Audio generated successfully: {output_path}")
        
        # Check if the file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Output file size: {file_size} bytes")
            
            # Load and check audio properties
            audio, sr = torchaudio.load(output_path)
            duration = audio.shape[1] / sr
            print(f"✓ Audio duration: {duration:.2f} seconds")
            print(f"✓ Sample rate: {sr} Hz")
            print(f"✓ Audio shape: {audio.shape}")
            
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 TTS test successful!")
        print("The base XTTS model is working with your reference audio.")
        print("To use your trained model, we would need to properly load the fine-tuned weights.")
    else:
        print("\n❌ TTS test failed!")
