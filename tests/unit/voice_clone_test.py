#!/usr/bin/env python3
"""
Working inference script using TTS API.
"""

import os
import sys

def main():
    print("🎯 Voice Cloning Test")
    print("=" * 30)
    
    try:
        from TTS.api import TTS
        print("✓ TTS API imported")
        
        # Load XTTS model
        print("Loading XTTS model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        print("✓ XTTS model loaded")
        
        # Set paths
        ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
        output_file = "voice_clone_test.wav"
        
        if not os.path.exists(ref_audio):
            print(f"❌ Reference audio not found: {ref_audio}")
            return False
        
        # Test text
        text = "Hello! This is a test of voice cloning using the dieck voice dataset. The model should clone the voice characteristics from the reference audio."
        
        print(f"Reference audio: {ref_audio}")
        print(f"Text: {text}")
        print(f"Output: {output_file}")
        
        # Generate speech
        print("Generating speech...")
        tts.tts_to_file(
            text=text,
            speaker_wav=ref_audio,
            language="en",
            file_path=output_file
        )
        
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"✓ Audio generated: {output_file} ({size} bytes)")
            return True
        else:
            print("❌ No output file created")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Voice cloning test successful!")
        print("Note: This uses the base XTTS model with your reference audio.")
        print("To use your fine-tuned model, additional integration would be needed.")
    else:
        print("\n❌ Voice cloning test failed!")
