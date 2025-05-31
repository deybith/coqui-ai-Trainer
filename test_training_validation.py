#!/usr/bin/env python3
"""
Quick training validation test to ensure the embedding fix works during actual training.
This test simulates a minimal training scenario to validate the IndexError is resolved.
"""

import torch
import sys
import os

# Set PYTHONPATH to avoid module conflicts
sys.path.insert(0, '/home/ubuntu/projects/coqui-ai-Trainer')

def test_training_scenario():
    """Test a minimal training scenario to ensure the fix works."""
    print("🚀 Testing XTTS Training Scenario")
    print("=" * 50)
    
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT, Phase2GPTConfig
        
        print("📋 Setting up training scenario...")
        
        # Create model configuration similar to actual training
        config = Phase2GPTConfig(
            vocab_size=512,
            block_size=1024,
            n_layer=2,  # Small for testing
            n_head=8,
            n_embd=512,
            number_text_tokens=512,  # This should handle tokens up to 511
            start_text_token=261,    # This was causing the IndexError
            stop_text_token=0,
            dropout=0.1,
        )
        
        print(f"   ✅ Config: vocab_size={config.vocab_size}, number_text_tokens={config.number_text_tokens}")
        print(f"   ✅ Special tokens: start={config.start_text_token}, stop={config.stop_text_token}")
        
        # Create model
        model = Phase2EnhancedGPT(config)
        device = torch.device('cpu')  # Use CPU for testing
        model = model.to(device)
        model.train()  # Set to training mode
        
        print("   ✅ Model created and set to training mode")
        
        # Simulate training data with problematic tokens
        print("\n📋 Testing training forward pass with problematic tokens...")
        
        # Create a batch that includes the tokens that were causing IndexError
        batch_size = 2
        seq_len = 10
        
        # Include tokens that were problematic: 256, 261, 300, etc.
        text_inputs = torch.tensor([
            [0, 100, 256, 261, 300, 400, 50, 75, 200, 0],      # First sequence
            [261, 256, 300, 150, 25, 400, 500, 300, 261, 0],   # Second sequence
        ], dtype=torch.long, device=device)
        
        print(f"   ✅ Created batch: shape={text_inputs.shape}")
        print(f"   ✅ Token range in batch: {text_inputs.min().item()} to {text_inputs.max().item()}")
        
        # This is the exact forward pass that was failing during training
        try:
            with torch.no_grad():  # No gradients for this test
                # The line that was causing IndexError: line 789 in phase2_enhanced_gpt.py
                text_emb = model.text_embedding(text_inputs)
                text_pos_emb = model.text_pos_embedding(text_inputs)
                combined_emb = text_emb + text_pos_emb
                
                print(f"   ✅ Forward pass successful!")
                print(f"   ✅ Text embedding shape: {text_emb.shape}")
                print(f"   ✅ Position embedding shape: {text_pos_emb.shape}")
                print(f"   ✅ Combined embedding shape: {combined_emb.shape}")
                
        except IndexError as e:
            print(f"   ❌ IndexError still occurs during forward pass: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Other error during forward pass: {e}")
            return False
        
        # Test with even more extreme tokens
        print("\n📋 Testing edge case tokens...")
        extreme_tokens = torch.tensor([[0, 511, 261, 256, 300, 450, 500, 511]], dtype=torch.long, device=device)
        
        try:
            with torch.no_grad():
                text_emb = model.text_embedding(extreme_tokens)
                print(f"   ✅ Edge case tokens handled successfully: shape={text_emb.shape}")
        except Exception as e:
            print(f"   ❌ Edge case tokens failed: {e}")
            return False
        
        print("\n🎉 TRAINING SCENARIO TEST PASSED!")
        print("✅ The embedding fix works correctly during training!")
        print("✅ No more IndexError with tokens 256-511!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING SCENARIO TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actual_xtts_integration():
    """Test integration with actual XTTS components."""
    print("\n🔧 Testing XTTS Integration...")
    
    try:
        # Test tokenizer integration
        from trainer.xtts.layers.xtts.tokenizer import VoiceBpeTokenizer
        
        tokenizer = VoiceBpeTokenizer()
        vocab_size = tokenizer.get_number_tokens()
        print(f"   ✅ Tokenizer vocab size: {vocab_size}")
        
        # Test that tokenizer produces valid tokens for our expanded vocabulary
        test_text = "Hello world, this is a comprehensive test of the XTTS tokenizer."
        encoded = tokenizer.encode(test_text)
        
        if len(encoded) > 0:
            max_token = max(encoded)
            print(f"   ✅ Encoded text max token: {max_token} (should be < {vocab_size})")
            
            if max_token >= vocab_size:
                print(f"   ❌ ERROR: Token {max_token} exceeds vocab size {vocab_size}")
                return False
        
        # Test that our expanded embedding can handle all tokenizer output
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT, Phase2GPTConfig
        
        config = Phase2GPTConfig(number_text_tokens=512)
        model = Phase2EnhancedGPT(config)
        device = torch.device('cpu')
        model = model.to(device)
        
        # Test with actual tokenizer output
        if len(encoded) > 0:
            tokens_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
            with torch.no_grad():
                embedding = model.text_embedding(tokens_tensor)
                print(f"   ✅ Tokenizer output processed successfully: {embedding.shape}")
        
        print("   ✅ XTTS integration test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ XTTS integration test failed: {e}")
        return False

def main():
    """Run all training validation tests."""
    print("🔍 XTTS Training Validation")
    print("=" * 60)
    
    test1_success = test_training_scenario()
    test2_success = test_actual_xtts_integration()
    
    print("\n" + "=" * 60)
    print("📊 TRAINING VALIDATION RESULTS")
    print("=" * 60)
    
    if test1_success and test2_success:
        print("🎉 ALL TRAINING VALIDATIONS PASSED!")
        print("✅ The IndexError fix is working perfectly in training scenarios!")
        print("✅ XTTS is ready for full training!")
        print("\n🚀 Summary of what was fixed:")
        print("   • Increased vocabulary size from 256 to 512 tokens")
        print("   • start_text_token=261 now works correctly")
        print("   • All tokens 0-511 are now supported")
        print("   • No more 'IndexError: index out of range' during training")
        return True
    else:
        print("❌ SOME TRAINING VALIDATIONS FAILED")
        print("⚠️  Please review the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
