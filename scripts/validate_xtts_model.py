#!/usr/bin/env python3
"""
XTTS Model Validation Script

This script validates that the enhanced XTTS fixes are working correctly:
1. Tests for audio cutoff issues
2. Checks for language mixing
3. Analyzes robotic/echo artifacts
4. Provides quality metrics and recommendations
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
from typing import Dict, List, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="XTTS Model Validation")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained XTTS model directory")
    parser.add_argument("--speaker_wav", type=str, required=True,
                        help="Path to speaker reference WAV file")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                        help="Directory to save validation results")
    
    return parser.parse_args()


class XTTSValidator:
    """XTTS Model Validator for checking common issues."""
    
    def __init__(self, model_path: str, speaker_wav: str, language: str = "en"):
        self.model_path = model_path
        self.speaker_wav = speaker_wav
        self.language = language
        self.model = None
        self.config = None
        self.validation_results = {}
        
    def load_model(self):
        """Load the XTTS model."""
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            print(f"🔄 Loading model from: {self.model_path}")
            
            # Load config
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            self.config = XttsConfig()
            self.config.load_json(config_path)
            
            # Load model
            self.model = Xtts.init_from_config(self.config)
            
            # Load checkpoint
            checkpoint_path = None
            for filename in ["model.pth", "best_model.pth", "checkpoint.pth"]:
                candidate = os.path.join(self.model_path, filename)
                if os.path.exists(candidate):
                    checkpoint_path = candidate
                    break
            
            if not checkpoint_path:
                raise FileNotFoundError(f"No model checkpoint found in {self.model_path}")
            
            print(f"📁 Loading checkpoint: {checkpoint_path}")
            self.model.load_checkpoint(self.config, checkpoint_path, eval=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model.cuda()
                print("🚀 Model loaded on GPU")
            else:
                print("💻 Model loaded on CPU")
            
            return True
            
        except ImportError:
            print("❌ TTS library not found. Please install with: pip install TTS")
            return False
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            return False
    
    def validate_speaker_wav(self) -> bool:
        """Validate speaker reference audio."""
        if not os.path.exists(self.speaker_wav):
            print(f"❌ Speaker WAV not found: {self.speaker_wav}")
            return False
        
        try:
            waveform, sample_rate = torchaudio.load(self.speaker_wav)
            duration = waveform.shape[1] / sample_rate
            
            print(f"✅ Speaker audio: {duration:.1f}s, {sample_rate}Hz")
            
            if duration < 3:
                print("⚠️  Speaker audio is short. Consider using 3-15s for best results.")
            elif duration > 30:
                print("⚠️  Speaker audio is long. Consider trimming to 15-30s.")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to validate speaker audio: {str(e)}")
            return False
    
    def test_audio_cutoff(self) -> Dict:
        """Test for audio cutoff issues."""
        print("\n🔍 Testing for audio cutoff issues...")
        
        test_texts = [
            "This is a short test sentence to check basic functionality.",
            "This is a much longer sentence designed to test whether the model can maintain consistency and avoid cutting off audio before the sentence is completely finished with all words being properly pronounced.",
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and should be synthesized completely without any premature cutoffs.",
        ]
        
        results = {
            "test_name": "Audio Cutoff Test",
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, text in enumerate(test_texts):
            try:
                print(f"  Testing text {i+1}: {text[:50]}...")
                
                # Synthesize audio
                audio, _ = self._synthesize_text(text)
                duration = len(audio) / 22050
                
                # Check for cutoff (analyze ending)
                cutoff_detected = self._detect_cutoff(audio, text)
                
                test_result = {
                    "text": text,
                    "duration": duration,
                    "cutoff_detected": cutoff_detected,
                    "status": "PASS" if not cutoff_detected else "FAIL"
                }
                
                results["details"].append(test_result)
                
                if cutoff_detected:
                    results["failed"] += 1
                    print(f"    ❌ Cutoff detected in {duration:.1f}s audio")
                else:
                    results["passed"] += 1
                    print(f"    ✅ No cutoff detected in {duration:.1f}s audio")
                    
            except Exception as e:
                print(f"    ❌ Test failed: {str(e)}")
                results["failed"] += 1
                results["details"].append({
                    "text": text,
                    "error": str(e),
                    "status": "ERROR"
                })
        
        return results
    
    def test_language_consistency(self) -> Dict:
        """Test for language mixing issues."""
        print("\n🌍 Testing for language consistency...")
        
        test_texts = [
            "Hello world, this is a test of the text to speech system.",
            "The weather is beautiful today and perfect for a walk in the park.",
            "Technology has advanced significantly in recent years with artificial intelligence.",
            "Please make sure to speak clearly and maintain consistent pronunciation throughout.",
        ]
        
        results = {
            "test_name": "Language Consistency Test",
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, text in enumerate(test_texts):
            try:
                print(f"  Testing consistency {i+1}: {text[:50]}...")
                
                # Synthesize same text multiple times
                audios = []
                for _ in range(3):
                    audio, _ = self._synthesize_text(text)
                    audios.append(audio)
                
                # Check consistency between generations
                consistency_score = self._measure_consistency(audios)
                consistent = consistency_score > 0.7  # Threshold for consistency
                
                test_result = {
                    "text": text,
                    "consistency_score": consistency_score,
                    "consistent": consistent,
                    "status": "PASS" if consistent else "FAIL"
                }
                
                results["details"].append(test_result)
                
                if consistent:
                    results["passed"] += 1
                    print(f"    ✅ Good consistency (score: {consistency_score:.2f})")
                else:
                    results["failed"] += 1
                    print(f"    ❌ Poor consistency (score: {consistency_score:.2f})")
                    
            except Exception as e:
                print(f"    ❌ Test failed: {str(e)}")
                results["failed"] += 1
                results["details"].append({
                    "text": text,
                    "error": str(e),
                    "status": "ERROR"
                })
        
        return results
    
    def test_audio_quality(self) -> Dict:
        """Test for robotic/echo artifacts."""
        print("\n🎵 Testing for audio quality issues...")
        
        test_texts = [
            "Welcome to the enhanced XTTS system with improved audio quality.",
            "The new configuration reduces robotic artifacts and improves naturalness.",
            "Listen carefully for any echo, distortion, or metallic sounds in this synthesis.",
        ]
        
        results = {
            "test_name": "Audio Quality Test", 
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, text in enumerate(test_texts):
            try:
                print(f"  Testing quality {i+1}: {text[:50]}...")
                
                # Synthesize audio
                audio, sample_rate = self._synthesize_text(text)
                
                # Analyze audio quality
                quality_metrics = self._analyze_audio_quality(audio, sample_rate)
                quality_good = quality_metrics["overall_score"] > 0.7
                
                test_result = {
                    "text": text,
                    "quality_metrics": quality_metrics,
                    "quality_good": quality_good,
                    "status": "PASS" if quality_good else "FAIL"
                }
                
                results["details"].append(test_result)
                
                if quality_good:
                    results["passed"] += 1
                    print(f"    ✅ Good quality (score: {quality_metrics['overall_score']:.2f})")
                else:
                    results["failed"] += 1
                    print(f"    ❌ Poor quality (score: {quality_metrics['overall_score']:.2f})")
                    
            except Exception as e:
                print(f"    ❌ Test failed: {str(e)}")
                results["failed"] += 1
                results["details"].append({
                    "text": text,
                    "error": str(e),
                    "status": "ERROR"
                })
        
        return results
    
    def _synthesize_text(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio."""
        # Load speaker reference
        speaker_waveform, _ = torchaudio.load(self.speaker_wav)
        if speaker_waveform.shape[0] > 1:
            speaker_waveform = torch.mean(speaker_waveform, dim=0, keepdim=True)
        
        # Enhanced inference parameters
        outputs = self.model.synthesize(
            text=text,
            config=self.config,
            speaker_wav=speaker_waveform,
            language=self.language,
            temperature=0.75,
            repetition_penalty=2.5,
            top_k=40,
            top_p=0.8,
        )
        
        audio = outputs["wav"]
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        if audio.ndim == 2:
            audio = audio.squeeze()
            
        return audio, 22050
    
    def _detect_cutoff(self, audio: np.ndarray, text: str) -> bool:
        """Detect if audio is cut off prematurely."""
        # Check if audio ends abruptly (high energy at the end)
        end_samples = min(2205, len(audio))  # Last 0.1 seconds
        end_energy = np.mean(audio[-end_samples:]**2)
        
        # Expected duration based on text length (rough estimate)
        expected_duration = len(text.split()) * 0.6  # ~0.6s per word
        actual_duration = len(audio) / 22050
        
        # Cutoff detected if audio is much shorter than expected or ends with high energy
        duration_ratio = actual_duration / expected_duration
        abrupt_ending = end_energy > 0.01
        
        return duration_ratio < 0.7 or abrupt_ending
    
    def _measure_consistency(self, audios: List[np.ndarray]) -> float:
        """Measure consistency between multiple audio generations."""
        if len(audios) < 2:
            return 1.0
        
        # Compare spectral features
        spectrograms = []
        for audio in audios:
            if len(audio) > 1024:
                spec = np.abs(np.fft.fft(audio[:1024]))
                spectrograms.append(spec)
        
        if len(spectrograms) < 2:
            return 0.5
        
        # Calculate correlation between spectrograms
        correlations = []
        for i in range(len(spectrograms)):
            for j in range(i+1, len(spectrograms)):
                corr = np.corrcoef(spectrograms[i], spectrograms[j])[0,1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.5
    
    def _analyze_audio_quality(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Analyze audio for quality metrics."""
        # Basic quality metrics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        snr = 20 * np.log10(peak / (rms + 1e-8))
        
        # Spectral analysis
        if len(audio) > 1024:
            freqs = np.fft.fftfreq(1024, 1/sample_rate)
            fft = np.fft.fft(audio[:1024])
            
            # High frequency content (potential artifacts)
            high_freq_energy = np.sum(np.abs(fft[freqs > 8000]))
            total_energy = np.sum(np.abs(fft))
            high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
            
            # Spectral centroid (brightness)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs_pos = freqs[:len(freqs)//2]
            spectral_centroid = np.sum(freqs_pos * magnitude) / (np.sum(magnitude) + 1e-8)
        else:
            high_freq_ratio = 0.0
            spectral_centroid = 1000.0
        
        # Quality scoring
        snr_score = min(1.0, max(0.0, (snr - 10) / 30))  # 10-40 dB range
        artifact_score = max(0.0, 1.0 - high_freq_ratio * 10)  # Lower is better
        brightness_score = max(0.0, 1.0 - abs(spectral_centroid - 2000) / 3000)
        
        overall_score = (snr_score + artifact_score + brightness_score) / 3
        
        return {
            "rms": rms,
            "peak": peak,
            "snr": snr,
            "high_freq_ratio": high_freq_ratio,
            "spectral_centroid": spectral_centroid,
            "snr_score": snr_score,
            "artifact_score": artifact_score,
            "brightness_score": brightness_score,
            "overall_score": overall_score
        }
    
    def run_validation(self, output_dir: str) -> Dict:
        """Run complete validation suite."""
        print("🧪 Starting XTTS Model Validation")
        print("="*50)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        if not self.load_model():
            return {"status": "error", "message": "Failed to load model"}
        
        # Validate speaker audio
        if not self.validate_speaker_wav():
            return {"status": "error", "message": "Invalid speaker audio"}
        
        # Run tests
        results = {
            "model_path": self.model_path,
            "speaker_wav": self.speaker_wav,
            "language": self.language,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {}
        }
        
        # Test 1: Audio cutoff
        results["tests"]["cutoff"] = self.test_audio_cutoff()
        
        # Test 2: Language consistency
        results["tests"]["consistency"] = self.test_language_consistency()
        
        # Test 3: Audio quality
        results["tests"]["quality"] = self.test_audio_quality()
        
        # Generate summary
        total_passed = sum(test["passed"] for test in results["tests"].values())
        total_failed = sum(test["failed"] for test in results["tests"].values())
        total_tests = total_passed + total_failed
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": total_passed / total_tests if total_tests > 0 else 0
        }
        
        # Save results
        results_file = os.path.join(output_dir, "validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print validation summary."""
        print("\n" + "="*50)
        print("📊 VALIDATION RESULTS SUMMARY")
        print("="*50)
        
        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        print("\n📋 Test Details:")
        for test_name, test_results in results["tests"].items():
            status = "✅" if test_results["failed"] == 0 else "❌"
            print(f"{status} {test_results['test_name']}: {test_results['passed']}/{test_results['passed'] + test_results['failed']}")
        
        if summary["success_rate"] >= 0.8:
            print("\n🎉 Excellent! Your enhanced XTTS model is working well.")
        elif summary["success_rate"] >= 0.6:
            print("\n👍 Good! Minor issues detected. Check the detailed results.")
        else:
            print("\n⚠️  Issues detected. Review the fixes and retrain if necessary.")
        
        print("="*50)


def main():
    """Main validation function."""
    args = parse_args()
    
    validator = XTTSValidator(args.model_path, args.speaker_wav, args.language)
    results = validator.run_validation(args.output_dir)
    
    if results.get("status") == "error":
        print(f"❌ Validation failed: {results['message']}")
        sys.exit(1)
    
    print(f"\n📁 Detailed results saved to: {args.output_dir}/validation_results.json")


if __name__ == "__main__":
    main()
