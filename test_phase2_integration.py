"""
XTTS Phase 2 Enhanced GPT Integration Test

This script validates that the Phase 2 Enhanced XTTS GPT model 
integrates correctly with all Phase 2 enhancements while maintaining
backward compatibility with the original XTTS architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
from trainer.xtts.layers.xtts.phase2_enhanced_gpt import (
    Phase2GPTConfig,
    Phase2EnhancedGPT,
    build_phase2_enhanced_gpt_transformer,
    create_phase2_enhanced_gpt,
    validate_phase2_integration,
)
from trainer.xtts.layers.attention.phase2_integration import Phase2Config


def test_configuration_alignment():
    """Test that Phase2GPTConfig and Phase2Config parameter alignment works"""
    print("🔍 Testing Configuration Parameter Alignment...")
    
    # Test automatic Phase2Config creation
    config = Phase2GPTConfig(
        layers=8,
        d_model=512,
        heads=8,
        use_phase2_enhancements=True
    )
    
    assert config.phase2_config is not None, "Phase2Config should be auto-created"
    assert config.phase2_config.d_model == config.d_model, "d_model should match"
    assert config.phase2_config.num_heads == config.heads, "num_heads should match"
    
    print("✅ Auto-configuration alignment test passed")
    
    # Test manual Phase2Config creation
    custom_phase2_config = Phase2Config(
        d_model=1024,
        num_heads=16,
        use_mamba=True,
        use_flash_attention=True,
        use_rope=True,
        use_moe=True,
    )
    
    custom_config = Phase2GPTConfig(
        layers=12,
        d_model=1024,
        heads=16,
        use_phase2_enhancements=True,
        phase2_config=custom_phase2_config
    )
    
    assert custom_config.phase2_config.d_model == 1024, "Custom d_model should be preserved"
    assert custom_config.phase2_config.num_heads == 16, "Custom num_heads should be preserved"
    
    print("✅ Manual configuration alignment test passed")
    return True


def test_factory_functions():
    """Test factory functions for creating Phase 2 enhanced models"""
    print("🔍 Testing Factory Functions...")
    
    # Test build_phase2_enhanced_gpt_transformer
    gpt, mel_pos_emb, text_pos_emb, _, _ = build_phase2_enhanced_gpt_transformer(
        layers=4,
        model_dim=256,  # Fixed parameter name
        heads=4,
        max_mel_seq_len=250,
        max_text_seq_len=120,
        checkpointing=False,
        use_phase2_enhancements=True
    )
    
    assert hasattr(gpt, 'h'), "GPT should have transformer blocks"
    assert len(gpt.h) == 4, "Should have 4 layers"
    
    print("✅ build_phase2_enhanced_gpt_transformer test passed")
    
    # Test create_phase2_enhanced_gpt
    model = create_phase2_enhanced_gpt(
        layers=6,
        d_model=384,
        heads=6,
        use_phase2_enhancements=True
    )
    
    assert model.use_phase2_enhancements == True, "Phase 2 enhancements should be enabled"
    assert model.layers == 6, "Should have 6 layers"
    assert model.d_model == 384, "Model dimension should be 384"
    
    print("✅ create_phase2_enhanced_gpt test passed")
    return True


def test_backward_compatibility():
    """Test that Phase 2 enhanced model maintains backward compatibility"""
    print("🔍 Testing Backward Compatibility...")
    
    # Test with Phase 2 disabled (should use original implementation)
    model_original = Phase2EnhancedGPT(
        layers=4,
        d_model=256,
        heads=4,
        use_phase2_enhancements=False
    )
    
    assert not model_original.use_phase2_enhancements, "Phase 2 should be disabled"
    
    # Test with Phase 2 enabled
    model_enhanced = Phase2EnhancedGPT(
        layers=4,
        d_model=256,
        heads=4,
        use_phase2_enhancements=True
    )
    
    assert model_enhanced.use_phase2_enhancements, "Phase 2 should be enabled"
    
    # Both models should have the same interface
    assert hasattr(model_original, 'get_grad_norm_parameter_groups'), "Should have parameter groups method"
    assert hasattr(model_enhanced, 'get_grad_norm_parameter_groups'), "Should have parameter groups method"
    
    print("✅ Backward compatibility test passed")
    return True


def test_embedding_compatibility():
    """Test that embeddings work with proper token ranges"""
    print("🔍 Testing Embedding Compatibility...")
    
    model = Phase2EnhancedGPT(
        layers=2,
        d_model=128,
        heads=2,
        use_phase2_enhancements=True,
        number_text_tokens=512,  # Large enough for start_text_token=261
        num_audio_tokens=8500,   # Large enough for start_audio_token=8192, stop_audio_token=8193
    )
    
    # Test text embeddings
    text_tokens = torch.randint(0, 200, (2, 10))
    text_emb = model.text_embedding(text_tokens)
    text_pos_emb = model.text_pos_embedding(text_tokens)
    
    assert text_emb.shape == (2, 10, 128), f"Text embedding shape should be (2, 10, 128), got {text_emb.shape}"
    assert text_pos_emb.shape == (2, 10, 128), f"Text pos embedding shape should be (2, 10, 128), got {text_pos_emb.shape}"
    
    # Test mel embeddings
    mel_tokens = torch.randint(0, 8000, (2, 15))
    mel_emb = model.mel_embedding(mel_tokens)
    mel_pos_emb = model.mel_pos_embedding(mel_tokens)
    
    assert mel_emb.shape == (2, 15, 128), f"Mel embedding shape should be (2, 15, 128), got {mel_emb.shape}"
    assert mel_pos_emb.shape == (2, 15, 128), f"Mel pos embedding shape should be (2, 15, 128), got {mel_pos_emb.shape}"
    
    print("✅ Embedding compatibility test passed")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("🚀 Running Phase 2 Enhanced XTTS GPT Integration Tests...")
    print("=" * 60)
    
    tests = [
        ("Configuration Alignment", test_configuration_alignment),
        ("Factory Functions", test_factory_functions), 
        ("Backward Compatibility", test_backward_compatibility),
        ("Embedding Compatibility", test_embedding_compatibility),
        ("Full Integration", validate_phase2_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 {test_name}:")
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 All Phase 2 Enhanced XTTS GPT integration tests PASSED!")
        print("✅ Configuration parameter alignment is working correctly")
        print("✅ Factory functions are working correctly")
        print("✅ Backward compatibility is maintained")
        print("✅ Embeddings are working correctly")
        print("✅ Phase 2 enhancements are properly integrated")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
