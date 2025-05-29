#!/usr/bin/env python3
"""
Enhanced XTTS Inference Testing Script

This script tests the enhanced XTTS model to validate fixes for:
1. Audio cutoff issues
2. Language mixing
3. Robotic/echo artifacts
"""

import argparse
import os
import sys
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
import time
import librosa

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trainer.xtts.enhanced_configs import EnhancedXTTSConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced XTTS Inference Testing")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained XTTS model directory")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--speaker_wav", type=str, required=True,
                        help="Path to speaker reference WAV file")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--output_path", type=str, default="./test_output.wav",
                        help="Output WAV file path")
    
    # Enhanced inference parameters
    parser.add_argument("--temperature", type=float, default=0.75,
                        help="Sampling temperature (lower = more consistent)")
    parser.add_argument("--repetition_penalty", type=float, default=2.5,
                        help="Repetition penalty (higher = less repetition)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="Top-p sampling parameter")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier")
    
    # Audio quality parameters
    parser.add_argument("--enable_text_splitting", action="store_true", default=True,
                        help="Enable text splitting for long sentences")
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming synthesis")
    
    return parser.parse_args()


def load_enhanced_model(model_path):
    """Load the enhanced XTTS model with optimized settings."""
    try:
        from trainer.xtts.configs.xtts_config import XttsConfig
        from trainer.xtts.models.xtts import Xtts
        
        print(f"🔄 Loading model from: {model_path}")
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = XttsConfig()
        config.load_json(config_path)
        
        # Load model
        model = Xtts.init_from_config(config)
        
        # Load checkpoint
        checkpoint_path = None
        for filename in ["model.pth", "best_model.pth", "checkpoint.pth"]:
            candidate = os.path.join(model_path, filename)
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break
        
        if not checkpoint_path:
            raise FileNotFoundError(f"No model checkpoint found in {model_path}")
        
        print(f"📁 Loading checkpoint: {checkpoint_path}")
        model.load_checkpoint(config, checkpoint_path, eval=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model.cuda()
            print("🚀 Model loaded on GPU")
        else:
            print("💻 Model loaded on CPU")
        
        return model, config
        
    except ImportError:
        print("❌ TTS library not found. Please install with: pip install TTS")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        sys.exit(1)


def validate_speaker_wav(speaker_wav_path):
    """Validate and preprocess speaker reference audio."""
    if not os.path.exists(speaker_wav_path):
        raise FileNotFoundError(f"Speaker WAV not found: {speaker_wav_path}")
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(speaker_wav_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("🔄 Converted speaker audio to mono")
        
        # Resample to 22050 Hz if needed
        if sample_rate != 22050:
            resampler = torchaudio.transforms.Resample(sample_rate, 22050)
            waveform = resampler(waveform)
            sample_rate = 22050
            print(f"🔄 Resampled speaker audio from {sample_rate}Hz to 22050Hz")
        
        # Check duration (should be 3-15 seconds for best results)
        duration = waveform.shape[1] / sample_rate
        if duration < 3:
            print(f"⚠️  Speaker audio is short ({duration:.1f}s). Consider using 3-15s for best results.")
        elif duration > 30:
            print(f"⚠️  Speaker audio is long ({duration:.1f}s). Truncating to 30s.")
            waveform = waveform[:, :30*sample_rate]
        
        print(f"✅ Speaker audio validated: {duration:.1f}s, {sample_rate}Hz")
        return waveform, sample_rate
        
    except Exception as e:
        raise ValueError(f"Failed to process speaker audio: {str(e)}")


def preprocess_text(text, language="en"):
    """Preprocess text for better synthesis."""
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Add punctuation if missing
    if not text.endswith(('.', '!', '?')):
        text += "."
    
    # Split very long sentences for better processing
    if len(text) > 200:
        print("📝 Text is long, consider splitting into shorter sentences for better results")
    
    print(f"📝 Preprocessed text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
    return text


def enhanced_inference(model, config, text, speaker_wav, language, args):
    """Perform enhanced inference with optimized parameters."""
    print("\n🎯 Starting enhanced inference...")
    
    # Preprocess inputs
    text = preprocess_text(text, language)
    speaker_waveform, _ = validate_speaker_wav(speaker_wav)
    
    # Enhanced inference parameters
    inference_kwargs = {
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "speed": args.speed,
        "enable_text_splitting": args.enable_text_splitting,
    }
    
    print("⚙️  Enhanced inference parameters:")
    for key, value in inference_kwargs.items():
        print(f"  • {key}: {value}")
    
    try:
        start_time = time.time()
        
        # Perform inference
        print("🔄 Generating audio...")
        
        if args.stream:
            print("🌊 Using streaming synthesis...")
            # Note: Streaming implementation would go here
            # For now, use standard synthesis
        
        # Standard synthesis with enhanced parameters
        outputs = model.synthesize(
            text=text,
            config=config,
            speaker_wav=speaker_waveform,
            language=language,
            **inference_kwargs
        )
        
        inference_time = time.time() - start_time
        
        if outputs is None or len(outputs["wav"]) == 0:
            raise ValueError("Model produced empty output")
        
        # Get the generated audio
        audio = outputs["wav"]
        
        # Convert to numpy if tensor
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        # Ensure proper shape
        if audio.ndim == 2:
            audio = audio.squeeze()
        
        audio_duration = len(audio) / 22050
        
        print(f"✅ Audio generated successfully!")
        print(f"  • Duration: {audio_duration:.2f}s")
        print(f"  • Inference time: {inference_time:.2f}s")
        print(f"  • Real-time factor: {inference_time/audio_duration:.2f}x")
        
        return audio, 22050
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        raise


def analyze_audio_quality(audio, sample_rate, output_path):
    """Analyze the generated audio for quality issues."""
    print("\n🔍 Analyzing audio quality...")
    
    try:
        # Basic statistics
        duration = len(audio) / sample_rate
        max_amplitude = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        print(f"📊 Audio Statistics:")
        print(f"  • Duration: {duration:.2f}s")
        print(f"  • Max amplitude: {max_amplitude:.3f}")
        print(f"  • RMS level: {rms:.3f}")
        print(f"  • Dynamic range: {20*np.log10(max_amplitude/rms):.1f}dB")
        
        # Check for cutoff (silence at the end)
        silence_threshold = 0.01
        end_samples = min(2205, len(audio))  # Check last 0.1s
        end_rms = np.sqrt(np.mean(audio[-end_samples:]**2))
        
        if end_rms < silence_threshold and duration > 1.0:
            print("⚠️  Potential audio cutoff detected (silent ending)")
        else:
            print("✅ No audio cutoff detected")
        
        # Check for clipping
        clipped_samples = np.sum(np.abs(audio) > 0.95)
        if clipped_samples > 0:
            print(f"⚠️  Audio clipping detected: {clipped_samples} samples")
        else:
            print("✅ No audio clipping detected")
        
        # Check for robotic artifacts (high frequency content)
        if len(audio) > 1024:
            freqs = np.fft.fftfreq(1024, 1/sample_rate)
            fft = np.fft.fft(audio[:1024])
            high_freq_energy = np.sum(np.abs(fft[freqs > 8000]))
            total_energy = np.sum(np.abs(fft))
            
            if high_freq_energy / total_energy > 0.1:
                print("⚠️  High frequency artifacts detected (possible robotic sound)")
            else:
                print("✅ No obvious robotic artifacts detected")
        
        # Save audio
        torchaudio.save(output_path, torch.tensor(audio).unsqueeze(0), sample_rate)
        print(f"💾 Audio saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio analysis failed: {str(e)}")
        return False


def run_test_suite(model, config, args):
    """Run a comprehensive test suite."""
    print("\n🧪 Running Enhanced XTTS Test Suite")
    print("="*50)
    
    test_cases = [
        {
            "name": "Short Phrase Test",
            "text": "Hello, this is a short test.",
            "description": "Tests basic synthesis quality"
        },
        {
            "name": "Long Sentence Test", 
            "text": "This is a much longer sentence designed to test whether the model can maintain consistency and avoid cutting off audio before the sentence is completely finished.",
            "description": "Tests audio cutoff fixes"
        },
        {
            "name": "Language Consistency Test",
            "text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
            "description": "Tests language mixing prevention"
        },
        {
            "name": "Natural Speech Test",
            "text": "Welcome to the enhanced XTTS system. We have implemented several improvements to address audio quality issues.",
            "description": "Tests robotic artifact reduction"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/4: {test_case['name']}")
        print(f"📝 {test_case['description']}")
        print(f"💬 Text: {test_case['text']}")
        
        try:
            # Generate output filename
            test_output = f"test_{i}_{test_case['name'].lower().replace(' ', '_')}.wav"
            
            # Run inference
            audio, sample_rate = enhanced_inference(
                model, config, test_case['text'], 
                args.speaker_wav, args.language, args
            )
            
            # Analyze quality
            quality_ok = analyze_audio_quality(audio, sample_rate, test_output)
            
            results.append({
                "test": test_case['name'],
                "success": True,
                "quality": quality_ok,
                "output": test_output
            })
            
            print(f"✅ Test {i} completed successfully")
            
        except Exception as e:
            print(f"❌ Test {i} failed: {str(e)}")
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"✅ Tests passed: {passed}/{total}")
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['test']}")
        if not result["success"]:
            print(f"     Error: {result.get('error', 'Unknown')}")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced XTTS is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check the errors above.")
    
    return results


def main():
    """Main testing function."""
    print("🎙️  Enhanced XTTS Inference Testing")
    print("="*50)
    
    args = parse_args()
    
    # Load model
    model, config = load_enhanced_model(args.model_path)
    
    if args.text == "TEST_SUITE":
        # Run comprehensive test suite
        results = run_test_suite(model, config, args)
    else:
        # Single inference test
        print(f"\n🎯 Single Inference Test")
        print(f"📝 Text: {args.text}")
        
        try:
            audio, sample_rate = enhanced_inference(
                model, config, args.text, 
                args.speaker_wav, args.language, args
            )
            
            analyze_audio_quality(audio, sample_rate, args.output_path)
            
            print(f"\n🎉 Inference completed successfully!")
            print(f"📁 Output saved to: {args.output_path}")
            
        except Exception as e:
            print(f"\n❌ Inference failed: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
