#!/usr/bin/env python3
"""
Interactive voice cloning script - allows you to input custom text for voice cloning.
"""

import os
import sys
from datetime import datetime

def interactive_voice_cloning():
    """Interactive voice cloning with custom text input."""
    
    print("🎤 Interactive Voice Cloning")
    print("=" * 35)
    print("Enter your text and generate custom voice clones!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    try:
        from TTS.api import TTS
        
        # Load TTS model
        print("Loading TTS model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        print("✓ Model loaded successfully!\n")
        
        # Reference audio
        ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
        if not os.path.exists(ref_audio):
            print(f"❌ Reference audio not found: {ref_audio}")
            return False
        
        counter = 1
        
        while True:
            try:
                # Get user input
                user_text = input("Enter text to clone (or 'quit' to exit): ").strip()
                
                if user_text.lower() in ['quit', 'exit', '']:
                    break
                
                if len(user_text) < 3:
                    print("⚠️  Please enter longer text (at least 3 characters)\n")
                    continue
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%H%M%S")
                output_file = f"custom_voice_{counter}_{timestamp}.wav"
                
                print(f"Generating: {user_text}")
                print(f"Output file: {output_file}")
                
                # Generate speech
                tts.tts_to_file(
                    text=user_text,
                    speaker_wav=ref_audio,
                    language="en",
                    file_path=output_file
                )
                
                if os.path.exists(output_file):
                    size = os.path.getsize(output_file)
                    print(f"✓ Generated: {output_file} ({size:,} bytes)")
                    print(f"🎵 Play with: ffplay {output_file}\n")
                    counter += 1
                else:
                    print("❌ Failed to generate audio file\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
        
        print(f"\n🎉 Session completed! Generated {counter-1} custom voice clips.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure TTS is properly installed.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def batch_voice_cloning():
    """Batch voice cloning from a predefined list."""
    
    print("\n🔄 Batch Voice Cloning")
    print("=" * 30)
    
    # Predefined interesting texts
    batch_texts = [
        "Welcome to the future of voice technology!",
        "This voice cloning system can speak any text you provide.",
        "Artificial intelligence has made remarkable progress in recent years.",
        "The quality of synthetic speech continues to improve dramatically.",
        "Voice cloning opens up exciting possibilities for content creation."
    ]
    
    try:
        from TTS.api import TTS
        
        print("Loading TTS model for batch processing...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        
        ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
        
        print(f"Generating {len(batch_texts)} audio samples...\n")
        
        for i, text in enumerate(batch_texts, 1):
            output_file = f"batch_output_{i}.wav"
            
            print(f"[{i}/{len(batch_texts)}] Generating: {text[:50]}...")
            
            tts.tts_to_file(
                text=text,
                speaker_wav=ref_audio,
                language="en",
                file_path=output_file
            )
            
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"✓ Created: {output_file} ({size:,} bytes)")
            else:
                print(f"❌ Failed: {output_file}")
        
        print(f"\n🎉 Batch processing completed!")
        print("Generated files: batch_output_1.wav through batch_output_5.wav")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False

def main():
    print("🎯 Advanced Voice Cloning Demonstration")
    print("=" * 45)
    print("\nChoose an option:")
    print("1. Interactive voice cloning (enter custom text)")
    print("2. Batch voice cloning (predefined texts)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            interactive_voice_cloning()
            break
        elif choice == '2':
            batch_voice_cloning()
            break
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
