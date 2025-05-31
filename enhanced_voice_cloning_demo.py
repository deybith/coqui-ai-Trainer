#!/usr/bin/env python3
"""
Enhanced Voice Cloning Demo with Quality Improvements
This script demonstrates advanced voice cloning capabilities with quality analysis.
"""

import os
import sys
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_audio_quality(audio_path):
    """Analyze audio quality metrics."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Basic audio analysis
        duration = waveform.shape[1] / sample_rate
        peak_amplitude = torch.max(torch.abs(waveform)).item()
        rms_level = torch.sqrt(torch.mean(waveform**2)).item()
        
        # Simple quality score
        quality_score = min(1.0, rms_level * 10)
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'peak_amplitude': peak_amplitude,
            'rms_level': rms_level,
            'quality_score': quality_score,
            'file_size': os.path.getsize(audio_path)
        }
    except Exception as e:
        print(f"❌ Error analyzing {audio_path}: {e}")
        return None

def enhanced_voice_cloning_demo():
    """Enhanced voice cloning demonstration."""
    
    print("🎉 Enhanced Voice Cloning Demo")
    print("=" * 40)
    
    try:
        from TTS.api import TTS
        
        print("🔄 Loading TTS model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        print("✓ Model loaded successfully")
        
        # Reference audio
        ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
        if not os.path.exists(ref_audio):
            print(f"❌ Reference audio not found: {ref_audio}")
            return False
        
        # Enhanced test texts
        test_scenarios = [
            {
                'category': 'Simple',
                'text': "Hello! This is an enhanced voice cloning test.",
            },
            {
                'category': 'Technical',
                'text': "The neural network utilizes advanced transformer architectures for voice synthesis.",
            },
            {
                'category': 'Emotional',
                'text': "I'm absolutely thrilled about this amazing voice technology breakthrough!",
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            output_file = f"enhanced_demo_{scenario['category'].lower()}.wav"
            
            print(f"\n[{i}/{len(test_scenarios)}] Generating {scenario['category']} sample...")
            
            start_time = time.time()
            
            try:
                tts.tts_to_file(
                    text=scenario['text'],
                    speaker_wav=ref_audio,
                    language="en",
                    file_path=output_file
                )
                
                generation_time = time.time() - start_time
                
                if os.path.exists(output_file):
                    analysis = analyze_audio_quality(output_file)
                    
                    if analysis:
                        print(f"✓ Generated in {generation_time:.2f}s")
                        print(f"  Duration: {analysis['duration']:.2f}s")
                        print(f"  Quality: {analysis['quality_score']:.3f}")
                        print(f"  Size: {analysis['file_size']:,} bytes")
                        
                        results.append({
                            'category': scenario['category'],
                            'generation_time': generation_time,
                            'analysis': analysis
                        })
                    else:
                        print(f"✓ Generated (analysis failed)")
                else:
                    print(f"❌ Generation failed")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Summary
        if results:
            print(f"\n📊 Enhanced Demo Summary")
            print("=" * 30)
            avg_quality = np.mean([r['analysis']['quality_score'] for r in results])
            avg_time = np.mean([r['generation_time'] for r in results])
            total_audio = sum([r['analysis']['duration'] for r in results])
            
            print(f"✓ Generated {len(results)} samples")
            print(f"✓ Average quality: {avg_quality:.3f}")
            print(f"✓ Average time: {avg_time:.2f}s")
            print(f"✓ Total audio: {total_audio:.1f}s")
            
            print(f"\n🎵 Generated files:")
            for result in results:
                print(f"  • enhanced_demo_{result['category'].lower()}.wav")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def quick_single_test():
    """Quick single test."""
    
    print("🚀 Quick Enhanced Test")
    print("=" * 25)
    
    try:
        from TTS.api import TTS
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        
        ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
        if not os.path.exists(ref_audio):
            print(f"❌ Reference audio not found")
            return False
        
        test_text = "This is an enhanced voice cloning test with quality analysis."
        output_file = "enhanced_quick_test.wav"
        
        print(f"🔄 Generating test sample...")
        
        tts.tts_to_file(
            text=test_text,
            speaker_wav=ref_audio,
            language="en",
            file_path=output_file
        )
        
        if os.path.exists(output_file):
            analysis = analyze_audio_quality(output_file)
            if analysis:
                print(f"✓ Generated: {output_file}")
                print(f"✓ Duration: {analysis['duration']:.2f}s")
                print(f"✓ Quality: {analysis['quality_score']:.3f}")
            return True
        else:
            print(f"❌ Generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    
    print("🎯 Enhanced Voice Cloning System")
    print("=" * 40)
    print("Choose an option:")
    print("1. Full enhanced demo")
    print("2. Quick enhanced test")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            enhanced_voice_cloning_demo()
            break
        elif choice == '2':
            quick_single_test()
            break
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
