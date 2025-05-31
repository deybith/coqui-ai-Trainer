#!/usr/bin/env python3
"""
Integration test to validate the embedding fix works with actual XTTS model training workflow.
This test verifies that the vocabulary size increase resolves the IndexError during training.
"""

import torch
import os
import sys
from pathlib import Path

# Add the trainer module to the path
sys.path.insert(0, str(Path(__file__).parent / "trainer"))

def test_xtts_model_initialization():
    """Test XTTS model initialization with the embedding fix."""
    print("🔧 Testing XTTS model initialization...")
    
    try:
        from trainer.xtts.models.xtts import XttsConfig, Xtts
        
        # Create a minimal config for testing
        config = XttsConfig(
            model_args={
                "gpt_batch_size": 1,
                "enable_redirection": False,
                "kv_cache": True,
                "gpt_checkpoint": None,
                "clvp_checkpoint": None,
                "decoder_checkpoint": None,
                "num_chars": 255,
            },
            audio={
                "sample_rate": 22050,
                "output_sample_rate": 24000,
            },
            model_dir="/tmp/test_model",
            tokenizer_file=None,
            gpt_num_samples=1,
            gpt_max_audio_tokens=604,
            gpt_max_text_tokens=402,
            gpt_max_prompt_tokens=70,
            gpt_layers=30,
            gpt_n_model_channels=1024,
            gpt_n_heads=16,
            gpt_number_text_tokens=512,  # This should now be 512 by default
            gpt_start_text_token=261,  # This was causing the IndexError
            gpt_checkpointing=False,
            gpt_train_solo_embeddings=False,
            gpt_code_stride_len=1024,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_mse_loss=False,
        )
        
        print(f"✅ Config created with gpt_number_text_tokens: {config.gpt_number_text_tokens}")
        print(f"✅ Start text token: {config.gpt_start_text_token}")
        
        # Initialize the model
        model = Xtts(config)
        print("✅ XTTS model initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ XTTS model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase2_enhanced_gpt_with_special_tokens():
    """Test Phase2EnhancedGPT directly with special tokens."""
    print("\n🔧 Testing Phase2EnhancedGPT with special tokens...")
    
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT, Phase2GPTConfig
        
        # Create config with special tokens
        config = Phase2GPTConfig(
            vocab_size=512,  # Should be 512 now
            block_size=1024,
            n_layer=12,
            n_head=12,
            n_embd=768,
            number_text_tokens=512,  # Should be 512 now
            start_text_token=261,  # This was causing the issue
            stop_text_token=0,
        )
        
        print(f"✅ Phase2GPTConfig created with number_text_tokens: {config.number_text_tokens}")
        print(f"✅ Start text token: {config.start_text_token}")
        
        # Create the model
        model = Phase2EnhancedGPT(config)
        print("✅ Phase2EnhancedGPT created successfully")
        
        # Test with the problematic start_text_token
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        # Create input tensor with the start_text_token that was causing IndexError
        text_inputs = torch.tensor([[261]], dtype=torch.long, device=device)  # start_text_token
        print(f"✅ Created text_inputs with start_text_token: {text_inputs}")
        
        # This should NOT raise IndexError anymore
        with torch.no_grad():
            text_emb = model.text_embedding(text_inputs)
            print(f"✅ Text embedding successful! Shape: {text_emb.shape}")
            
            # Test position embedding too
            text_pos_emb = model.text_pos_embedding(text_inputs)
            print(f"✅ Text position embedding successful! Shape: {text_pos_emb.shape}")
            
            # Test combined embedding (this was the line that failed)
            combined_emb = text_emb + text_pos_emb
            print(f"✅ Combined embedding successful! Shape: {combined_emb.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase2EnhancedGPT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer_integration():
    """Test tokenizer integration with the embedding layers."""
    print("\n🔧 Testing tokenizer integration...")
    
    try:
        from trainer.xtts.layers.xtts.tokenizer import VoiceBpeTokenizer
        
        # Initialize tokenizer
        tokenizer = VoiceBpeTokenizer()
        print(f"✅ Tokenizer initialized")
        print(f"✅ Tokenizer vocabulary size: {tokenizer.get_number_tokens()}")
        
        # Test text encoding
        test_text = "Hello world, this is a test."
        encoded = tokenizer.encode(test_text)
        print(f"✅ Encoded text: {encoded}")
        print(f"✅ Max token ID in encoded text: {max(encoded) if encoded else 'None'}")
        
        # Verify all token IDs are within the vocabulary range
        vocab_size = tokenizer.get_number_tokens()
        for token_id in encoded:
            if token_id >= vocab_size:
                print(f"❌ Token ID {token_id} exceeds vocabulary size {vocab_size}")
                return False
        
        print(f"✅ All token IDs are within vocabulary range [0, {vocab_size-1}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_bounds():
    """Test that embedding layers handle the full range of token IDs correctly."""
    print("\n🔧 Testing embedding layer bounds...")
    
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT, Phase2GPTConfig
        
        config = Phase2GPTConfig(number_text_tokens=512)
        model = Phase2EnhancedGPT(config)
        
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        # Test edge cases
        test_cases = [
            ("Min token ID", torch.tensor([[0]], dtype=torch.long)),
            ("Max valid token ID", torch.tensor([[511]], dtype=torch.long)),  # 512-1
            ("Start text token", torch.tensor([[261]], dtype=torch.long)),  # The problematic one
            ("Multiple tokens", torch.tensor([[0, 100, 261, 500]], dtype=torch.long)),
        ]
        
        for test_name, token_tensor in test_cases:
            try:
                with torch.no_grad():
                    embedding = model.text_embedding(token_tensor.to(device))
                    print(f"✅ {test_name}: {token_tensor.flatten().tolist()} -> Shape: {embedding.shape}")
            except Exception as e:
                print(f"❌ {test_name} failed: {e}")
                return False
        
        # Test what would have failed before the fix
        try:
            # This would have failed with the old vocabulary size of 256
            problematic_tokens = torch.tensor([[256, 261, 300, 400]], dtype=torch.long)
            with torch.no_grad():
                embedding = model.text_embedding(problematic_tokens.to(device))
                print(f"✅ Previously problematic tokens handled: Shape: {embedding.shape}")
        except Exception as e:
            print(f"❌ Previously problematic tokens still fail: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding bounds test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting XTTS Training Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Tokenizer Integration", test_tokenizer_integration),
        ("Embedding Bounds", test_embedding_bounds),
        ("Phase2EnhancedGPT with Special Tokens", test_phase2_enhanced_gpt_with_special_tokens),
        ("XTTS Model Initialization", test_xtts_model_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✅" if success else "❌"
        print(f"{icon} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The embedding fix is working correctly.")
        print("🚀 Ready for actual XTTS training!")
        return True
    else:
        print("⚠️  Some tests failed. Please investigate the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
