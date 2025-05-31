#!/usr/bin/env python3
"""
Simple voice cloning demo using the trained model and TTS API.
This script demonstrates voice cloning with your reference audio.
"""

import os
import sys

def main():
    print("🎤 Voice Cloning Demo")
    print("=" * 30)
    
    # Check if reference audio exists
    ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
    if not os.path.exists(ref_audio):
        print(f"❌ Reference audio not found: {ref_audio}")
        print("Please ensure the dataset is in the correct location.")
        return False
    
    print(f"✓ Using reference audio: {os.path.basename(ref_audio)}")
    
    # Test texts
    test_texts = [
        "Hello! This is a test of the voice cloning system.",
        "The model has been trained on the dieck voice dataset.",
        "Voice cloning technology is quite impressive these days.",
        "This demonstrates the capabilities of the XTTS model."
    ]
    
    try:
        print("Loading TTS model...")
        from TTS.api import TTS
        
        # Load XTTS model
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        print("✓ XTTS model loaded successfully")
        
        # Generate samples
        for i, text in enumerate(test_texts, 1):
            output_file = f"demo_output_{i}.wav"
            
            print(f"\nGenerating sample {i}...")
            print(f"Text: {text}")
            
            tts.tts_to_file(
                text=text,
                speaker_wav=ref_audio,
                language="en",
                file_path=output_file
            )
            
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"✓ Generated: {output_file} ({size} bytes)")
            else:
                print(f"❌ Failed to generate: {output_file}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nGenerated files:")
        for i in range(1, len(test_texts) + 1):
            filename = f"demo_output_{i}.wav"
            if os.path.exists(filename):
                print(f"  - {filename}")
        
        print("\nTo play the audio files:")
        print("  ffplay demo_output_1.wav")
        print("  # or use your favorite audio player")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure TTS is properly installed.")
        return False
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1)
