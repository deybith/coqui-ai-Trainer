#!/usr/bin/env python3
"""
Quick validation of Enhanced XTTS components.
"""

import sys
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_neural_codec():
    """Test neural codec functionality."""
    try:
        from trainer.xtts.layers.encodec import create_encodec_for_xtts
        
        codec = create_encodec_for_xtts(
            mel_dim=80,
            hidden_dim=512,
            num_quantizers=4,
            codebook_size=1024,
            adaptive=True,
        )
        
        # Test encoding/decoding
        test_mel = torch.randn(2, 80, 100)
        codes, loss = codec.encode(test_mel)
        reconstructed = codec.decode(codes)
        
        assert codes.shape[0] == test_mel.shape[0]
        assert reconstructed.shape == test_mel.shape
        assert loss.item() >= 0
        
        logger.info("✓ Neural Codec: PASS")
        return True
    except Exception as e:
        logger.error(f"✗ Neural Codec: FAIL - {e}")
        return False

def test_streaming():
    """Test streaming components."""
    try:
        from trainer.xtts.layers.streaming import (
            create_streaming_decoder,
            StreamingBuffer,
            create_quality_monitor
        )
        
        # Test streaming decoder
        decoder = create_streaming_decoder(
            vocab_size=8194,
            embed_dim=1024,
            num_layers=4,
            num_heads=16,
            context_window=512,
            chunk_size=64,
        )
        
        # Test streaming buffer
        buffer = StreamingBuffer(
            buffer_size=1024,
            chunk_size=64,
            overlap_size=16,
        )
        
        test_data = torch.randn(1, 100, 1024)
        buffer.update(test_data)
        context = buffer.get_context()
        assert context is not None
        
        # Test quality monitor
        monitor = create_quality_monitor(
            feature_dim=80,
            quality_threshold=0.8,
        )
        
        test_features = torch.randn(1, 80, 100)
        test_target = torch.randn(1, 80, 100)
        quality_score = monitor(test_features, test_target)
        assert 'quality_score' in quality_score
        
        logger.info("✓ Streaming Components: PASS")
        return True
    except Exception as e:
        logger.error(f"✗ Streaming Components: FAIL - {e}")
        return False

def test_integration():
    """Test integration wrapper."""
    try:
        from trainer.xtts.models.enhanced_integration import EnhancedXTTSWrapper
        
        # Create a mock XTTS model for testing
        class MockXTTS:
            def __init__(self):
                self.args = type('Args', (), {
                    'hidden_channels': 1024,
                    'text_embedding_dim': 512,
                    'cond_channels': 1024,
                })()
                self.use_deepspeed = False
            
            def forward(self, x, *args, **kwargs):
                return torch.randn(x.shape[0], 80, 100)
        
        mock_xtts = MockXTTS()
        wrapper = EnhancedXTTSWrapper(mock_xtts)
        
        # Test inference
        test_input = torch.randn(2, 512)
        output = wrapper.inference(
            text_input=test_input,
            speaker_embedding=torch.randn(2, 512),
            use_enhanced=True
        )
        
        assert 'audio' in output
        assert 'quality_metrics' in output
        
        logger.info("✓ Integration Wrapper: PASS")
        return True
    except Exception as e:
        logger.error(f"✗ Integration Wrapper: FAIL - {e}")
        return False

def main():
    """Run quick validation."""
    logger.info("Starting Enhanced XTTS Quick Validation...")
    logger.info("=" * 50)
    
    tests = [
        test_neural_codec,
        test_streaming,
        test_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    logger.info("=" * 50)
    if passed == total:
        logger.info(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        logger.info("Enhanced XTTS implementation is working correctly!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed ({passed}/{total})")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
