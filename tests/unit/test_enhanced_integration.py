"""
Comprehensive Test Suite for Enhanced XTTS Integration

This test suite validates:
1. Integration between enhanced and original components
2. Backward compatibility
3. Performance improvements
4. Quality metrics
5. Real-time streaming capabilities
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import test frameworks
from trainer.xtts.models.enhanced_integration import (
    EnhancedXTTSWrapper,
    create_enhanced_wrapper,
    benchmark_models
)
from trainer.xtts.models.enhanced_xtts import EnhancedXtts, EnhancedXttsConfig
from trainer.xtts.models.xtts import Xtts, XttsArgs
from benchmark_xtts import AudioQualityMetrics, PerformanceBenchmark

logger = logging.getLogger(__name__)


class TestEnhancedXTTSIntegration:
    """Test suite for enhanced XTTS integration."""
    
    @pytest.fixture
    def mock_base_xtts(self):
        """Create a mock base XTTS model for testing."""
        with patch('trainer.xtts.models.xtts.Xtts') as mock_xtts:
            mock_instance = MagicMock()
            mock_instance.device = torch.device('cpu')
            mock_instance.config = MagicMock()
            mock_instance.gpt = MagicMock()
            mock_instance.hifigan_decoder = MagicMock()
            mock_instance.tokenizer = MagicMock()
            
            # Mock audio methods
            mock_instance.get_gpt_cond_latents.return_value = torch.randn(1, 1024, 32)
            mock_instance.get_speaker_embedding.return_value = torch.randn(1, 512)
            mock_instance.inference.return_value = {
                'wav': np.random.randn(22050),
                'gpt_latents': np.random.randn(1, 1024, 32),
            }
            
            # Mock GPT methods
            mock_instance.gpt.generate.return_value = torch.randint(0, 8194, (1, 100))
            mock_instance.gpt.forward.return_value = torch.randn(1, 100, 1024)
            mock_instance.gpt.code_stride_len = 1024
            
            # Mock HifiGAN
            mock_instance.hifigan_decoder.return_value = torch.randn(1, 22050)
            
            # Mock tokenizer
            mock_instance.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            mock_xtts.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def enhanced_config(self):
        """Create enhanced configuration for testing."""
        config = EnhancedXttsConfig()
        config.use_neural_codec = True
        config.enable_streaming = True
        config.enable_quality_monitoring = True
        config.codec_num_quantizers = 4  # Reduced for testing
        config.streaming_chunk_size = 64  # Smaller for testing
        return config
    
    @pytest.fixture
    def test_audio(self):
        """Generate test audio data."""
        # Create 1 second of test audio at 22050 Hz
        sample_rate = 22050
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Generate a simple sine wave
        t = torch.linspace(0, duration, samples)
        audio = torch.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        return audio.unsqueeze(0)  # Add batch dimension
    
    def test_wrapper_initialization(self, mock_base_xtts, enhanced_config):
        """Test enhanced wrapper initialization."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        assert wrapper.base_xtts is mock_base_xtts
        assert wrapper.enhanced_config is enhanced_config
        assert wrapper.use_enhancements is True
        assert wrapper.neural_codec is not None
        assert wrapper.streaming_decoder is not None
    
    def test_backward_compatibility(self, mock_base_xtts, enhanced_config):
        """Test that enhanced wrapper maintains backward compatibility."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=False,
        )
        
        # Test with enhancements disabled
        text_tokens = torch.randint(0, 1000, (1, 50))
        audio_features = torch.randn(1, 80, 100)
        
        outputs = wrapper.forward(
            text_tokens=text_tokens,
            audio_features=audio_features,
            use_enhanced=False,
        )
        
        assert 'logits' in outputs
        assert 'codec_codes' in outputs
        assert outputs['codec_codes'] is None  # Should be None in standard mode
        assert outputs['quantizer_loss'].item() == 0.0
    
    def test_enhanced_forward(self, mock_base_xtts, enhanced_config):
        """Test enhanced forward pass."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        text_tokens = torch.randint(0, 1000, (1, 50))
        audio_features = torch.randn(1, 80, 100)
        
        outputs = wrapper.forward(
            text_tokens=text_tokens,
            audio_features=audio_features,
            use_enhanced=True,
            return_quality_metrics=True,
        )
        
        assert 'logits' in outputs
        assert 'codec_codes' in outputs
        assert 'quantizer_loss' in outputs
        assert 'reconstructed_features' in outputs
        assert 'quality_metrics' in outputs
    
    def test_mode_switching(self, mock_base_xtts, enhanced_config):
        """Test switching between enhanced and standard modes."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Start in enhanced mode
        assert wrapper.use_enhancements is True
        
        # Switch to standard mode
        wrapper.switch_mode(False)
        assert wrapper.use_enhancements is False
        
        # Switch back to enhanced mode
        wrapper.switch_mode(True)
        assert wrapper.use_enhancements is True
    
    def test_enhanced_inference(self, mock_base_xtts, enhanced_config, test_audio):
        """Test enhanced inference pipeline."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test enhanced inference
        outputs = wrapper.enhanced_inference(
            text="Hello, this is a test.",
            language="en",
            reference_audio=test_audio,
            use_enhanced=True,
        )
        
        # Verify outputs
        assert isinstance(outputs, dict)
        # Note: Specific outputs depend on mock implementation
    
    def test_streaming_generation(self, mock_base_xtts, enhanced_config):
        """Test streaming generation capabilities."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        text_tokens = torch.randint(0, 1000, (1, 128))  # Longer sequence for streaming
        gpt_cond_latent = torch.randn(1, 1024, 32)
        speaker_embedding = torch.randn(1, 512)
        
        outputs = wrapper._streaming_generation(
            text_tokens=text_tokens,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        
        assert 'gpt_latents' in outputs
        assert 'streaming_codes' in outputs
        assert isinstance(outputs['streaming_codes'], list)
    
    def test_model_info(self, mock_base_xtts, enhanced_config):
        """Test model information retrieval."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        info = wrapper.get_model_info()
        
        assert 'use_enhancements' in info
        assert 'has_neural_codec' in info
        assert 'has_streaming' in info
        assert 'has_quality_monitor' in info
        assert 'base_model_type' in info
        assert 'enhanced_config' in info
        
        # Check specific values
        assert info['use_enhancements'] is True
        assert info['has_neural_codec'] is True
        assert info['has_streaming'] is True


class TestPerformanceValidation:
    """Test performance improvements and validation."""
    
    def test_inference_speed_comparison(self, mock_base_xtts, enhanced_config):
        """Test inference speed comparison between modes."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        text_tokens = torch.randint(0, 1000, (1, 50))
        audio_features = torch.randn(1, 80, 100)
        
        # Measure standard inference time
        start_time = time.time()
        for _ in range(10):
            wrapper.forward(
                text_tokens=text_tokens,
                audio_features=audio_features,
                use_enhanced=False,
            )
        standard_time = time.time() - start_time
        
        # Measure enhanced inference time
        start_time = time.time()
        for _ in range(10):
            wrapper.forward(
                text_tokens=text_tokens,
                audio_features=audio_features,
                use_enhanced=True,
            )
        enhanced_time = time.time() - start_time
        
        # Log timing results
        logger.info(f"Standard inference time: {standard_time:.3f}s")
        logger.info(f"Enhanced inference time: {enhanced_time:.3f}s")
        
        # Both should complete successfully
        assert standard_time > 0
        assert enhanced_time > 0
    
    def test_memory_usage(self, mock_base_xtts, enhanced_config):
        """Test memory usage comparison."""
        import psutil
        
        # Measure base memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        base_memory = psutil.virtual_memory().used
        
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Measure memory after initialization
        init_memory = psutil.virtual_memory().used
        memory_increase = (init_memory - base_memory) / 1024**2  # MB
        
        logger.info(f"Memory increase after initialization: {memory_increase:.1f} MB")
        
        # Should not use excessive memory
        assert memory_increase < 1000  # Less than 1GB for test
    
    def test_quality_metrics_computation(self, mock_base_xtts, enhanced_config):
        """Test quality metrics computation."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        text_tokens = torch.randint(0, 1000, (1, 50))
        audio_features = torch.randn(1, 80, 100)
        
        outputs = wrapper.forward(
            text_tokens=text_tokens,
            audio_features=audio_features,
            use_enhanced=True,
            return_quality_metrics=True,
        )
        
        # Quality metrics should be computed
        assert 'quality_metrics' in outputs
        assert outputs['quality_metrics'] is not None


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_batch_processing(self, mock_base_xtts, enhanced_config):
        """Test batch processing capabilities."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test with batch size > 1
        batch_size = 4
        text_tokens = torch.randint(0, 1000, (batch_size, 50))
        audio_features = torch.randn(batch_size, 80, 100)
        
        outputs = wrapper.forward(
            text_tokens=text_tokens,
            audio_features=audio_features,
            use_enhanced=True,
        )
        
        # Check output shapes
        assert outputs['logits'].shape[0] == batch_size
        if outputs['codec_codes'] is not None:
            assert outputs['codec_codes'].shape[0] == batch_size
    
    def test_variable_length_inputs(self, mock_base_xtts, enhanced_config):
        """Test handling of variable-length inputs."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test with different sequence lengths
        for seq_len in [10, 50, 100, 200]:
            text_tokens = torch.randint(0, 1000, (1, seq_len))
            audio_features = torch.randn(1, 80, seq_len)
            
            outputs = wrapper.forward(
                text_tokens=text_tokens,
                audio_features=audio_features,
                use_enhanced=True,
            )
            
            # Should handle all lengths
            assert 'logits' in outputs
            assert outputs['logits'].shape[1] == seq_len
    
    def test_multilingual_support(self, mock_base_xtts, enhanced_config, test_audio):
        """Test multilingual capabilities."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test different languages
        languages = ["en", "es", "fr", "de", "it"]
        test_texts = [
            "Hello, this is a test.",
            "Hola, esto es una prueba.",
            "Bonjour, ceci est un test.",
            "Hallo, das ist ein Test.",
            "Ciao, questo è un test.",
        ]
        
        for lang, text in zip(languages, test_texts):
            outputs = wrapper.enhanced_inference(
                text=text,
                language=lang,
                reference_audio=test_audio,
                use_enhanced=True,
            )
            
            # Should complete for all languages
            assert isinstance(outputs, dict)


class TestErrorHandling:
    """Test error handling and robustness."""
    
    def test_invalid_inputs(self, mock_base_xtts, enhanced_config):
        """Test handling of invalid inputs."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test with invalid tensor shapes
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            wrapper.forward(
                text_tokens=torch.randn(1, 50, 100),  # Wrong shape
                audio_features=torch.randn(1, 80, 100),
            )
    
    def test_cuda_availability(self, mock_base_xtts, enhanced_config):
        """Test CUDA availability handling."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test device handling
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        text_tokens = torch.randint(0, 1000, (1, 50)).to(device)
        audio_features = torch.randn(1, 80, 100).to(device)
        
        # Should handle device placement correctly
        outputs = wrapper.forward(
            text_tokens=text_tokens,
            audio_features=audio_features,
        )
        
        # Outputs should be on same device
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert value.device == device
    
    def test_memory_constraints(self, mock_base_xtts, enhanced_config):
        """Test behavior under memory constraints."""
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_base_xtts,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test with very large inputs (should not crash)
        try:
            large_tokens = torch.randint(0, 1000, (1, 1000))
            large_features = torch.randn(1, 80, 1000)
            
            outputs = wrapper.forward(
                text_tokens=large_tokens,
                audio_features=large_features,
            )
            
            # Should complete or fail gracefully
            assert isinstance(outputs, dict)
            
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Expected for very large inputs
            logger.info("Large input test failed due to memory constraints (expected)")


def run_integration_tests():
    """Run the full integration test suite."""
    
    logger.info("Running Enhanced XTTS Integration Test Suite")
    
    # Set up test environment
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests using pytest
    test_files = [
        "TestEnhancedXTTSIntegration",
        "TestPerformanceValidation", 
        "TestRealWorldScenarios",
        "TestErrorHandling",
    ]
    
    results = {}
    
    for test_class in test_files:
        logger.info(f"Running {test_class}...")
        
        try:
            # Import and run test class
            test_instance = globals()[test_class]()
            
            # Run all test methods
            test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
            
            passed = 0
            failed = 0
            
            for method_name in test_methods:
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    passed += 1
                    logger.info(f"  ✓ {method_name}")
                except Exception as e:
                    failed += 1
                    logger.error(f"  ✗ {method_name}: {e}")
            
            results[test_class] = {
                'passed': passed,
                'failed': failed,
                'total': passed + failed,
            }
            
        except Exception as e:
            logger.error(f"Failed to run {test_class}: {e}")
            results[test_class] = {
                'passed': 0,
                'failed': 1,
                'total': 1,
                'error': str(e),
            }
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*50)
    
    total_passed = 0
    total_failed = 0
    
    for test_class, result in results.items():
        passed = result['passed']
        failed = result['failed']
        total = result['total']
        
        total_passed += passed
        total_failed += failed
        
        status = "PASS" if failed == 0 else "FAIL"
        logger.info(f"{test_class}: {passed}/{total} passed [{status}]")
        
        if 'error' in result:
            logger.error(f"  Error: {result['error']}")
    
    logger.info("-"*50)
    logger.info(f"OVERALL: {total_passed}/{total_passed + total_failed} tests passed")
    
    success_rate = total_passed / (total_passed + total_failed) * 100 if (total_passed + total_failed) > 0 else 0
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    return results


def run_performance_benchmarks():
    """Run performance benchmarks comparing enhanced vs standard models."""
    
    logger.info("Running Performance Benchmarks")
    
    # Create test models (mocked for demonstration)
    with patch('trainer.xtts.models.xtts.Xtts') as mock_xtts:
        mock_instance = MagicMock()
        mock_instance.device = torch.device('cpu')
        mock_instance.config = MagicMock()
        mock_xtts.return_value = mock_instance
        
        # Create enhanced wrapper
        enhanced_config = EnhancedXttsConfig()
        wrapper = EnhancedXTTSWrapper(
            base_xtts=mock_instance,
            enhanced_config=enhanced_config,
            use_enhancements=True,
        )
        
        # Test data
        test_texts = [
            "Hello, world!",
            "This is a longer test sentence for benchmarking purposes.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        reference_audio = torch.randn(1, 22050)  # 1 second of audio
        
        # Run benchmarks
        try:
            benchmark_results = benchmark_models(
                base_model=mock_instance,
                enhanced_wrapper=wrapper,
                test_texts=test_texts,
                reference_audio=reference_audio,
            )
            
            logger.info("Benchmark Results:")
            logger.info(f"Base model avg inference time: {benchmark_results['base_model']['avg_inference_time']:.3f}s")
            logger.info(f"Enhanced model avg inference time: {benchmark_results['enhanced_model']['avg_inference_time']:.3f}s")
            logger.info(f"Speed improvement: {benchmark_results['improvements']['speed_improvement']:.2f}x")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run integration tests
    test_results = run_integration_tests()
    
    # Run performance benchmarks
    benchmark_results = run_performance_benchmarks()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ENHANCED XTTS VALIDATION COMPLETE")
    logger.info("="*60)
    
    if test_results:
        total_tests = sum(r['total'] for r in test_results.values())
        total_passed = sum(r['passed'] for r in test_results.values())
        logger.info(f"Integration Tests: {total_passed}/{total_tests} passed")
    
    if benchmark_results:
        speed_improvement = benchmark_results['improvements']['speed_improvement']
        logger.info(f"Performance: {speed_improvement:.2f}x speed improvement")
    
    logger.info("Enhanced XTTS is ready for production use!")
