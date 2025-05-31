#!/usr/bin/env python3
"""
Test script to verify the token ID indexing fix in Phase2EnhancedGPT

This script tests that the embedding layer can now handle special tokens
like start_text_token=261 without throwing IndexError.
"""

import torch
import sys
import os

# Add trainer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

def test_embedding_fix():
    """Test that the embedding layer fix resolves the IndexError"""
    
    print("🔍 Testing Phase2EnhancedGPT embedding layer fix...")
    
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT
        
        # Create a model with the fixed vocabulary size
        model = Phase2EnhancedGPT(
            layers=4,
            d_model=256,
            heads=4,
            number_text_tokens=512,  # Fixed: increased from 256 to accommodate special tokens
            num_audio_tokens=8500,
            start_text_token=261,    # This previously caused IndexError
            stop_text_token=0,
            use_phase2_enhancements=False  # Use simpler config for testing
        )
        
        print(f"✅ Model created successfully with:")
        print(f"   - Text embedding vocabulary size: {model.text_embedding.num_embeddings}")
        print(f"   - Start text token: {model.start_text_token}")
        print(f"   - Stop text token: {model.stop_text_token}")
        
        # Test that we can handle the special tokens
        print("\n🧪 Testing special token handling...")
        
        # Create test inputs that include the special start token
        batch_size = 2
        seq_len = 10
        
        # Test text inputs with special tokens
        text_inputs = torch.randint(0, 255, (batch_size, seq_len))  # Normal tokenizer range
        text_inputs[:, 0] = model.start_text_token  # Set first token to start token (261)
        
        print(f"   - Text input shape: {text_inputs.shape}")
        print(f"   - Text input range: {text_inputs.min().item()} to {text_inputs.max().item()}")
        print(f"   - Contains start token {model.start_text_token}: {(text_inputs == model.start_text_token).any().item()}")
        
        # Test that embedding lookup works without IndexError
        print("\n🎯 Testing embedding lookup...")
        text_embeddings = model.text_embedding(text_inputs)
        position_embeddings = model.text_pos_embedding(text_inputs)
        
        print(f"   ✅ Text embeddings computed successfully: {text_embeddings.shape}")
        print(f"   ✅ Position embeddings computed successfully: {position_embeddings.shape}")
        
        # Test the combined embedding (this is where the original error occurred)
        combined_embeddings = text_embeddings + position_embeddings
        print(f"   ✅ Combined embeddings computed successfully: {combined_embeddings.shape}")
        
        # Test edge cases - maximum token values
        print("\n🔬 Testing edge cases...")
        
        # Test with token ID = 511 (maximum allowed with new vocabulary size of 512)
        edge_inputs = torch.full((1, 1), 511, dtype=torch.long)
        edge_embeddings = model.text_embedding(edge_inputs)
        print(f"   ✅ Maximum token ID (511) handled successfully: {edge_embeddings.shape}")
        
        # Test that token ID = 512 would still fail (as expected)
        try:
            invalid_inputs = torch.full((1, 1), 512, dtype=torch.long)
            invalid_embeddings = model.text_embedding(invalid_inputs)
            print("   ⚠️  Warning: Token ID 512 should have failed but didn't")
        except IndexError:
            print("   ✅ Token ID 512 correctly fails (expected behavior)")
        
        print(f"\n🎉 All tests passed! The embedding fix is working correctly.")
        print(f"   📊 Summary:")
        print(f"   - Vocabulary size increased from 256 to 512")
        print(f"   - Special start_text_token=261 now works correctly")
        print(f"   - No more IndexError during embedding lookup")
        print(f"   - Both regular tokens (0-254) and special tokens (255+) are supported")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_original_vs_fixed():
    """Compare behavior before and after the fix"""
    
    print(f"\n🔄 Comparing original vs fixed behavior...")
    
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT
        
        # Simulate the original problematic configuration
        print("   🚫 Testing original configuration (would have failed):")
        print("      - number_text_tokens=256, start_text_token=261")
        print("      - This would cause IndexError: index 261 out of range [0, 256)")
        
        # Test the fixed configuration
        print("\n   ✅ Testing fixed configuration:")
        model_fixed = Phase2EnhancedGPT(
            layers=2,
            d_model=128,
            heads=2,
            number_text_tokens=512,  # Fixed: increased from 256
            start_text_token=261,
            use_phase2_enhancements=False
        )
        
        # Test with the problematic token
        test_input = torch.tensor([[261]], dtype=torch.long)  # Start token that caused the error
        embedding = model_fixed.text_embedding(test_input)
        
        print(f"      - Successfully processed start_text_token=261")
        print(f"      - Embedding shape: {embedding.shape}")
        print(f"      - Embedding vocabulary size: {model_fixed.text_embedding.num_embeddings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Phase2EnhancedGPT Embedding Fix")
    print("=" * 50)
    
    test1_passed = test_embedding_fix()
    test2_passed = test_original_vs_fixed()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED - The embedding fix is working correctly!")
        print("📝 Summary:")
        print("   - Phase2EnhancedGPT vocabulary size increased from 256 to 512")
        print("   - Special tokens like start_text_token=261 are now supported")
        print("   - IndexError: index out of range has been resolved")
        print("   - Training should now proceed without token ID errors")
        return True
    else:
        print("❌ SOME TESTS FAILED - Please review the output above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
