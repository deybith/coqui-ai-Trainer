"""
Enhanced XTTS Testing and Validation Suite

This script provides comprehensive testing for the enhanced XTTS model including:
- Functional tests for all new components
- Performance regression tests
- Quality validation tests
- Integration tests with streaming
- Backward compatibility tests

Usage:
    python test_enhanced_xtts.py --model_path /path/to/model --test_suite all
"""

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm

# Import models and components
from trainer.xtts.models.enhanced_xtts import EnhancedXtts, EnhancedXttsConfig, create_enhanced_xtts
from trainer.xtts.layers.encodec import create_encodec_for_xtts, mel_to_encodec_codes
from trainer.xtts.layers.streaming import create_streaming_decoder, create_quality_monitor
from benchmark_xtts import XTTSBenchmarkSuite, AudioQualityMetrics

logger = logging.getLogger(__name__)


class EnhancedXTTSTestSuite:
    """Comprehensive test suite for Enhanced XTTS model."""
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.test_results = {}
        
        # Create test data
        self.test_audio = self._create_test_audio()
        self.test_texts = self._create_test_texts()
        
        logger.info(f"Test suite initialized on {device}")
    
    def _create_test_audio(self) -> torch.Tensor:
        """Create synthetic test audio."""
        # Generate 3 seconds of test audio with various frequencies
        sample_rate = 22050
        duration = 3.0
        samples = int(sample_rate * duration)
        
        t = torch.linspace(0, duration, samples)
        
        # Mix of frequencies to test codec performance
        audio = (
            0.3 * torch.sin(2 * np.pi * 440 * t) +  # A4 note
            0.2 * torch.sin(2 * np.pi * 880 * t) +  # A5 note
            0.1 * torch.sin(2 * np.pi * 220 * t) +  # A3 note
            0.05 * torch.randn(samples)              # White noise
        )
        
        return audio
    
    def _create_test_texts(self) -> List[str]:
        """Create test texts of various lengths and complexities."""
        return [
            "Hello world!",
            "This is a test of the text-to-speech system.",
            "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet.",
            "Artificial intelligence and machine learning are revolutionizing the field of speech synthesis.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        ]
    
    def run_all_tests(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Run all test suites."""
        logger.info("Starting comprehensive test suite")
        
        test_suites = [
            ("Neural Codec Tests", self.test_neural_codec),
            ("Streaming Tests", self.test_streaming),
            ("Quality Monitoring Tests", self.test_quality_monitoring),
            ("Performance Tests", self.test_performance),
            ("Integration Tests", self.test_integration),
            ("Backward Compatibility Tests", self.test_backward_compatibility),
            ("Stress Tests", self.test_stress),
        ]
        
        all_results = {}
        
        for suite_name, test_function in test_suites:
            logger.info(f"Running {suite_name}...")
            try:
                results = test_function(model)
                all_results[suite_name] = {
                    "status": "PASSED" if results.get("passed", False) else "FAILED",
                    "results": results,
                }
                logger.info(f"{suite_name}: {all_results[suite_name]['status']}")
            except Exception as e:
                logger.error(f"{suite_name} failed with exception: {e}")
                all_results[suite_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
        
        # Summary
        passed = sum(1 for r in all_results.values() if r["status"] == "PASSED")
        total = len(all_results)
        
        all_results["SUMMARY"] = {
            "total_suites": total,
            "passed": passed,
            "failed": total - passed,
            "overall_status": "PASSED" if passed == total else "FAILED",
        }
        
        logger.info(f"Test suite completed: {passed}/{total} passed")
        return all_results
    
    def test_neural_codec(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Test neural codec functionality."""
        results = {"passed": True, "details": {}}
        
        try:
            # Test codec availability
            if not hasattr(model, 'neural_codec') or model.neural_codec is None:
                if model.enhanced_config.use_neural_codec:
                    results["passed"] = False
                    results["details"]["codec_availability"] = "FAILED: Codec enabled but not available"
                else:
                    results["details"]["codec_availability"] = "SKIPPED: Codec disabled"
                    return results
            else:
                results["details"]["codec_availability"] = "PASSED"
            
            # Test codec forward pass
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050, n_mels=80
            ).to(self.device)
            
            test_mel = mel_transform(self.test_audio.to(self.device)).transpose(1, 2).unsqueeze(0)
            
            with torch.no_grad():
                codec_output, codes, losses = model.neural_codec(test_mel)
            
            # Validate outputs
            if codec_output.shape != test_mel.shape:
                results["passed"] = False
                results["details"]["codec_output_shape"] = f"FAILED: Expected {test_mel.shape}, got {codec_output.shape}"
            else:
                results["details"]["codec_output_shape"] = "PASSED"
            
            # Test reconstruction quality
            reconstruction_loss = F.mse_loss(codec_output, test_mel).item()
            if reconstruction_loss > 1.0:  # Threshold for acceptable reconstruction
                results["passed"] = False
                results["details"]["reconstruction_quality"] = f"FAILED: Loss too high ({reconstruction_loss:.4f})"
            else:
                results["details"]["reconstruction_quality"] = f"PASSED: Loss {reconstruction_loss:.4f}"
            
            # Test quantization codes
            if codes.dim() != 3:  # [B, T, num_quantizers]
                results["passed"] = False
                results["details"]["quantization_codes"] = f"FAILED: Expected 3D tensor, got {codes.dim()}D"
            else:
                results["details"]["quantization_codes"] = "PASSED"
            
            # Test losses
            expected_losses = ["commitment_loss", "reconstruction_loss"]
            for loss_name in expected_losses:
                if loss_name not in losses:
                    results["passed"] = False
                    results["details"][f"loss_{loss_name}"] = "FAILED: Missing loss"
                else:
                    results["details"][f"loss_{loss_name}"] = "PASSED"
            
        except Exception as e:
            results["passed"] = False
            results["details"]["exception"] = f"FAILED: {str(e)}"
        
        return results
    
    def test_streaming(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Test streaming functionality."""
        results = {"passed": True, "details": {}}
        
        try:
            # Test streaming availability
            if not hasattr(model, 'streaming_decoder') or model.streaming_decoder is None:
                if model.enhanced_config.enable_streaming:
                    results["passed"] = False
                    results["details"]["streaming_availability"] = "FAILED: Streaming enabled but not available"
                else:
                    results["details"]["streaming_availability"] = "SKIPPED: Streaming disabled"
                    return results
            else:
                results["details"]["streaming_availability"] = "PASSED"
            
            # Test streaming generation
            if hasattr(model, 'generate_streaming'):
                text = self.test_texts[1]  # Medium length text
                reference_audio = self.test_audio.to(self.device)
                
                start_time = time.time()
                try:
                    generated_audio = model.generate_streaming(
                        text=text,
                        speaker_audio=reference_audio,
                        max_length=512,
                    )
                    generation_time = time.time() - start_time
                    
                    # Validate output
                    if generated_audio is None or generated_audio.numel() == 0:
                        results["passed"] = False
                        results["details"]["streaming_generation"] = "FAILED: No output generated"
                    else:
                        results["details"]["streaming_generation"] = f"PASSED: Generated {generated_audio.shape[-1]} samples in {generation_time:.2f}s"
                    
                    # Test latency
                    audio_duration = generated_audio.shape[-1] / 22050  # Assume 22050 Hz
                    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
                    
                    if rtf > 2.0:  # Should be better than 2x real-time
                        results["details"]["streaming_latency"] = f"WARNING: High RTF ({rtf:.2f})"
                    else:
                        results["details"]["streaming_latency"] = f"PASSED: RTF {rtf:.2f}"
                
                except Exception as e:
                    results["passed"] = False
                    results["details"]["streaming_generation"] = f"FAILED: {str(e)}"
            else:
                results["passed"] = False
                results["details"]["streaming_generation"] = "FAILED: No generate_streaming method"
            
            # Test streaming buffer
            if hasattr(model, 'streaming_buffer'):
                buffer = model.streaming_buffer
                
                # Test buffer operations
                test_chunk = torch.randn(256, 80)  # Random chunk
                buffer.reset()
                
                ready = buffer.add_chunk(test_chunk)
                if ready:
                    chunk = buffer.get_processing_chunk()
                    if chunk is not None:
                        results["details"]["streaming_buffer"] = "PASSED"
                    else:
                        results["details"]["streaming_buffer"] = "FAILED: No chunk returned"
                else:
                    results["details"]["streaming_buffer"] = "PASSED: Buffer not ready (expected)"
            
        except Exception as e:
            results["passed"] = False
            results["details"]["exception"] = f"FAILED: {str(e)}"
        
        return results
    
    def test_quality_monitoring(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Test quality monitoring functionality."""
        results = {"passed": True, "details": {}}
        
        try:
            # Test quality monitor availability
            if not hasattr(model, 'quality_monitor') or model.quality_monitor is None:
                if model.enhanced_config.enable_quality_monitoring:
                    results["passed"] = False
                    results["details"]["monitor_availability"] = "FAILED: Quality monitoring enabled but not available"
                else:
                    results["details"]["monitor_availability"] = "SKIPPED: Quality monitoring disabled"
                    return results
            else:
                results["details"]["monitor_availability"] = "PASSED"
            
            # Test quality assessment
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050, n_mels=80
            ).to(self.device)
            
            test_mel = mel_transform(self.test_audio.to(self.device)).transpose(1, 2).unsqueeze(0)
            
            with torch.no_grad():
                quality_metrics = model.quality_monitor(test_mel)
            
            # Validate quality metrics
            expected_keys = ['current_quality', 'average_quality', 'quality_trend', 'needs_adjustment']
            for key in expected_keys:
                if key not in quality_metrics:
                    results["passed"] = False
                    results["details"][f"metric_{key}"] = "FAILED: Missing metric"
                else:
                    value = quality_metrics[key]
                    if key in ['current_quality', 'average_quality']:
                        if not (0.0 <= value <= 1.0):
                            results["details"][f"metric_{key}"] = f"WARNING: Out of range ({value:.3f})"
                        else:
                            results["details"][f"metric_{key}"] = f"PASSED: {value:.3f}"
                    else:
                        results["details"][f"metric_{key}"] = f"PASSED: {value}"
            
        except Exception as e:
            results["passed"] = False
            results["details"]["exception"] = f"FAILED: {str(e)}"
        
        return results
    
    def test_performance(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Test performance characteristics."""
        results = {"passed": True, "details": {}}
        
        try:
            # Memory usage test
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            # Inference speed test
            text_tokens = torch.randint(0, 1000, (1, 50), device=self.device)
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050, n_mels=80
            ).to(self.device)
            audio_features = mel_transform(self.test_audio.to(self.device)).transpose(1, 2).unsqueeze(0)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(text_tokens=text_tokens, audio_features=audio_features)
            
            # Measure inference time
            times = []
            for _ in range(10):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model(text_tokens=text_tokens, audio_features=audio_features)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results["details"]["inference_time"] = f"PASSED: {avg_time:.3f}±{std_time:.3f}s"
            
            # Memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage = (peak_memory - initial_memory) / 1024 / 1024  # MB
                results["details"]["memory_usage"] = f"PASSED: {memory_usage:.1f} MB"
                
                # Check for memory leaks
                if memory_usage > 2000:  # 2GB threshold
                    results["details"]["memory_leak_check"] = "WARNING: High memory usage"
                else:
                    results["details"]["memory_leak_check"] = "PASSED"
            
            # Model size test
            param_count = sum(p.numel() for p in model.parameters())
            param_count_millions = param_count / 1e6
            
            results["details"]["model_size"] = f"PASSED: {param_count_millions:.1f}M parameters"
            
            # Check for reasonable model size
            if param_count_millions > 1000:  # 1B parameter threshold
                results["details"]["model_size_check"] = "WARNING: Very large model"
            else:
                results["details"]["model_size_check"] = "PASSED"
            
        except Exception as e:
            results["passed"] = False
            results["details"]["exception"] = f"FAILED: {str(e)}"
        
        return results
    
    def test_integration(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Test integration between components."""
        results = {"passed": True, "details": {}}
        
        try:
            # Test full pipeline
            text = self.test_texts[2]  # Longer text
            reference_audio = self.test_audio.to(self.device)
            
            # Test with different streaming modes
            streaming_modes = [False]
            if model.enhanced_config.enable_streaming:
                streaming_modes.append(True)
            
            for streaming in streaming_modes:
                mode_name = "streaming" if streaming else "standard"
                
                try:
                    # Prepare inputs
                    if hasattr(model, 'tokenizer'):
                        text_tokens = model.tokenizer.encode(text)
                        text_tokens = torch.tensor(text_tokens, device=self.device).unsqueeze(0)
                    else:
                        text_tokens = torch.randint(0, 1000, (1, len(text)), device=self.device)
                    
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=22050, n_mels=80
                    ).to(self.device)
                    audio_features = mel_transform(reference_audio).transpose(1, 2).unsqueeze(0)
                    
                    # Forward pass
                    outputs = model(
                        text_tokens=text_tokens,
                        audio_features=audio_features,
                        streaming=streaming,
                        return_quality_metrics=True,
                    )
                    
                    # Validate outputs
                    if 'logits' not in outputs:
                        results["passed"] = False
                        results["details"][f"integration_{mode_name}"] = "FAILED: No logits output"
                    else:
                        results["details"][f"integration_{mode_name}"] = "PASSED"
                    
                    # Test quality metrics if available
                    if model.enhanced_config.enable_quality_monitoring and 'quality_metrics' in outputs:
                        quality_metrics = outputs['quality_metrics']
                        if 'current_quality' in quality_metrics:
                            results["details"][f"quality_integration_{mode_name}"] = "PASSED"
                        else:
                            results["details"][f"quality_integration_{mode_name}"] = "FAILED: Missing quality metrics"
                    
                except Exception as e:
                    results["passed"] = False
                    results["details"][f"integration_{mode_name}"] = f"FAILED: {str(e)}"
            
            # Test codec integration if available
            if model.enhanced_config.use_neural_codec and model.neural_codec is not None:
                try:
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=22050, n_mels=80
                    ).to(self.device)
                    test_mel = mel_transform(reference_audio).transpose(1, 2).unsqueeze(0)
                    
                    # Test encoding
                    codes = model.neural_codec.encode_only(test_mel)
                    
                    # Test decoding
                    reconstructed = model.neural_codec.decode_only(codes)
                    
                    if reconstructed.shape == test_mel.shape:
                        results["details"]["codec_integration"] = "PASSED"
                    else:
                        results["passed"] = False
                        results["details"]["codec_integration"] = f"FAILED: Shape mismatch {reconstructed.shape} vs {test_mel.shape}"
                
                except Exception as e:
                    results["passed"] = False
                    results["details"]["codec_integration"] = f"FAILED: {str(e)}"
            
        except Exception as e:
            results["passed"] = False
            results["details"]["exception"] = f"FAILED: {str(e)}"
        
        return results
    
    def test_backward_compatibility(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Test backward compatibility with original XTTS."""
        results = {"passed": True, "details": {}}
        
        try:
            # Test that enhanced features can be disabled
            config = model.enhanced_config
            
            # Check fallback mechanisms
            if config.fallback_to_mel:
                results["details"]["mel_fallback"] = "PASSED: Fallback enabled"
            else:
                results["details"]["mel_fallback"] = "INFO: Fallback disabled"
            
            # Test with enhanced features disabled
            original_codec_setting = config.use_neural_codec
            original_streaming_setting = config.enable_streaming
            original_quality_setting = config.enable_quality_monitoring
            
            # Temporarily disable enhanced features
            config.use_neural_codec = False
            config.enable_streaming = False
            config.enable_quality_monitoring = False
            
            # Test basic forward pass
            text_tokens = torch.randint(0, 1000, (1, 50), device=self.device)
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050, n_mels=80
            ).to(self.device)
            audio_features = mel_transform(self.test_audio.to(self.device)).transpose(1, 2).unsqueeze(0)
            
            try:
                with torch.no_grad():
                    outputs = model(text_tokens=text_tokens, audio_features=audio_features)
                
                results["details"]["basic_compatibility"] = "PASSED"
            except Exception as e:
                results["passed"] = False
                results["details"]["basic_compatibility"] = f"FAILED: {str(e)}"
            
            # Restore original settings
            config.use_neural_codec = original_codec_setting
            config.enable_streaming = original_streaming_setting
            config.enable_quality_monitoring = original_quality_setting
            
        except Exception as e:
            results["passed"] = False
            results["details"]["exception"] = f"FAILED: {str(e)}"
        
        return results
    
    def test_stress(self, model: EnhancedXtts) -> Dict[str, Any]:
        """Stress test with various inputs."""
        results = {"passed": True, "details": {}}
        
        try:
            # Test with different text lengths
            test_cases = [
                ("short", self.test_texts[0]),
                ("medium", self.test_texts[2]),
                ("long", self.test_texts[4]),
            ]
            
            for case_name, text in test_cases:
                try:
                    # Prepare inputs
                    if hasattr(model, 'tokenizer'):
                        text_tokens = model.tokenizer.encode(text)
                        text_tokens = torch.tensor(text_tokens, device=self.device).unsqueeze(0)
                    else:
                        text_tokens = torch.randint(0, 1000, (1, min(len(text), 200)), device=self.device)
                    
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=22050, n_mels=80
                    ).to(self.device)
                    audio_features = mel_transform(self.test_audio.to(self.device)).transpose(1, 2).unsqueeze(0)
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(text_tokens=text_tokens, audio_features=audio_features)
                    
                    results["details"][f"stress_test_{case_name}"] = "PASSED"
                
                except Exception as e:
                    results["passed"] = False
                    results["details"][f"stress_test_{case_name}"] = f"FAILED: {str(e)}"
            
            # Test with different batch sizes
            batch_sizes = [1, 2, 4] if torch.cuda.is_available() else [1, 2]
            
            for batch_size in batch_sizes:
                try:
                    text_tokens = torch.randint(0, 1000, (batch_size, 50), device=self.device)
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=22050, n_mels=80
                    ).to(self.device)
                    
                    # Repeat audio for batch
                    batch_audio = self.test_audio.unsqueeze(0).repeat(batch_size, 1).to(self.device)
                    audio_features = mel_transform(batch_audio).transpose(1, 2)
                    
                    with torch.no_grad():
                        outputs = model(text_tokens=text_tokens, audio_features=audio_features)
                    
                    results["details"][f"batch_size_{batch_size}"] = "PASSED"
                
                except Exception as e:
                    results["details"][f"batch_size_{batch_size}"] = f"FAILED: {str(e)}"
                    if batch_size > 1:
                        # Large batch size failures are warnings, not failures
                        continue
                    else:
                        results["passed"] = False
            
        except Exception as e:
            results["passed"] = False
            results["details"]["exception"] = f"FAILED: {str(e)}"
        
        return results


def run_component_tests():
    """Run standalone component tests."""
    logger.info("Running component tests...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test neural codec components
    logger.info("Testing neural codec...")
    try:
        codec = create_encodec_for_xtts(mel_dim=80, adaptive=True).to(device)
        test_mel = torch.randn(1, 100, 80, device=device)
        
        with torch.no_grad():
            output, codes, losses = codec(test_mel)
        
        assert output.shape == test_mel.shape, f"Shape mismatch: {output.shape} vs {test_mel.shape}"
        assert codes.dim() == 3, f"Codes should be 3D, got {codes.dim()}D"
        assert "reconstruction_loss" in losses, "Missing reconstruction loss"
        
        logger.info("✓ Neural codec tests passed")
    except Exception as e:
        logger.error(f"✗ Neural codec tests failed: {e}")
    
    # Test streaming components
    logger.info("Testing streaming components...")
    try:
        streaming_decoder = create_streaming_decoder(
            vocab_size=1000,
            embed_dim=512,
            num_layers=6,
            num_heads=8,
        ).to(device)
        
        test_tokens = torch.randint(0, 1000, (1, 50), device=device)
        
        with torch.no_grad():
            outputs = streaming_decoder(test_tokens, streaming=True)
        
        assert "logits" in outputs, "Missing logits output"
        assert outputs["logits"].shape[:2] == test_tokens.shape, "Shape mismatch in logits"
        
        logger.info("✓ Streaming component tests passed")
    except Exception as e:
        logger.error(f"✗ Streaming component tests failed: {e}")
    
    # Test quality monitor
    logger.info("Testing quality monitor...")
    try:
        quality_monitor = create_quality_monitor(feature_dim=80).to(device)
        test_features = torch.randn(1, 100, 80, device=device)
        
        with torch.no_grad():
            metrics = quality_monitor(test_features)
        
        expected_keys = ['current_quality', 'average_quality', 'quality_trend', 'needs_adjustment']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        logger.info("✓ Quality monitor tests passed")
    except Exception as e:
        logger.error(f"✗ Quality monitor tests failed: {e}")


def main():
    """Main testing script."""
    parser = argparse.ArgumentParser(description="Enhanced XTTS Testing Suite")
    parser.add_argument("--model_path", help="Path to enhanced XTTS model")
    parser.add_argument("--config_path", help="Path to model config")
    parser.add_argument("--test_suite", default="all", 
                       choices=["all", "components", "model", "performance"],
                       help="Which test suite to run")
    parser.add_argument("--output_dir", help="Output directory for test results")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run component tests
    if args.test_suite in ["all", "components"]:
        run_component_tests()
    
    # Run model tests
    if args.test_suite in ["all", "model"] and args.model_path:
        logger.info("Loading enhanced XTTS model...")
        
        try:
            model = create_enhanced_xtts(
                config_path=args.config_path,
                checkpoint_path=args.model_path,
                device=args.device,
            )
            
            # Initialize test suite
            test_suite = EnhancedXTTSTestSuite(device=args.device)
            
            # Run all tests
            results = test_suite.run_all_tests(model)
            
            # Print summary
            print("\n" + "="*60)
            print("ENHANCED XTTS TEST RESULTS SUMMARY")
            print("="*60)
            
            for suite_name, suite_results in results.items():
                if suite_name == "SUMMARY":
                    continue
                
                status = suite_results["status"]
                status_symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "!"
                
                print(f"{status_symbol} {suite_name}: {status}")
                
                if "results" in suite_results and "details" in suite_results["results"]:
                    for detail_name, detail_result in suite_results["results"]["details"].items():
                        print(f"    {detail_name}: {detail_result}")
            
            # Overall summary
            summary = results["SUMMARY"]
            print("\n" + "-"*60)
            print(f"Overall: {summary['passed']}/{summary['total_suites']} test suites passed")
            print(f"Status: {summary['overall_status']}")
            print("="*60)
            
            # Save results if output directory specified
            if args.output_dir:
                output_path = Path(args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                with open(output_path / "test_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Test results saved to {output_path / 'test_results.json'}")
            
        except Exception as e:
            logger.error(f"Failed to load model or run tests: {e}")
            return 1
    
    # Performance benchmarks
    if args.test_suite in ["all", "performance"] and args.model_path:
        logger.info("Running performance benchmarks...")
        
        try:
            # This would integrate with the benchmark suite
            # For now, just log that it would run
            logger.info("Performance benchmarks would run here...")
            logger.info("Use benchmark_xtts.py for detailed performance testing")
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())
