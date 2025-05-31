#!/usr/bin/env python3
"""
Final validation test for XTTS training IndexError fix.
This test validates that the token ID indexing error has been resolved.
"""

import torch
import sys
import os

# Set PYTHONPATH to avoid module conflicts
sys.path.insert(0, '/home/ubuntu/projects/coqui-ai-Trainer')

def test_embedding_fix():
    """Test that the embedding fix resolves the IndexError."""
    print("🚀 Testing XTTS Token ID Indexing Fix")
    print("=" * 50)
    
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT, Phase2GPTConfig
        
        # Test 1: Verify default configuration
        print("📋 Test 1: Verifying default configuration...")
        config = Phase2GPTConfig()
        print(f"   ✅ Default number_text_tokens: {config.number_text_tokens}")
        print(f"   ✅ Default start_text_token: {config.start_text_token}")
        
        if config.number_text_tokens < 512:
            print(f"   ❌ ERROR: number_text_tokens should be ≥512, got {config.number_text_tokens}")
            return False
        
        # Test 2: Create model with problematic tokens
        print("\n📋 Test 2: Creating model with enhanced configuration...")
        enhanced_config = Phase2GPTConfig(
            number_text_tokens=512,
            start_text_token=261,  # This was causing the IndexError
            stop_text_token=0,
        )
        
        model = Phase2EnhancedGPT(enhanced_config)
        print(f"   ✅ Phase2EnhancedGPT created successfully")
        print(f"   ✅ Text embedding vocab size: {model.text_embedding.num_embeddings}")
        
        # Test 3: Test the specific scenario that was failing
        print("\n📋 Test 3: Testing problematic token scenarios...")
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        # Test cases that would have failed before the fix
        test_cases = [
            ("start_text_token=261", torch.tensor([[261]], dtype=torch.long)),
            ("token_256", torch.tensor([[256]], dtype=torch.long)),
            ("token_300", torch.tensor([[300]], dtype=torch.long)),
            ("multiple_high_tokens", torch.tensor([[256, 261, 300, 400, 500]], dtype=torch.long)),
        ]
        
        for test_name, tokens in test_cases:
            try:
                with torch.no_grad():
                    # This is the exact line that was failing: text_emb = self.text_embedding(text_inputs)
                    text_emb = model.text_embedding(tokens.to(device))
                    text_pos_emb = model.text_pos_embedding(tokens.to(device))
                    # This is the exact line from phase2_enhanced_gpt.py line 789
                    combined_emb = text_emb + text_pos_emb
                    
                print(f"   ✅ {test_name}: tokens {tokens.flatten().tolist()} -> embedding shape {text_emb.shape}")
            except IndexError as e:
                print(f"   ❌ {test_name}: IndexError still occurs: {e}")
                return False
            except Exception as e:
                print(f"   ❌ {test_name}: Unexpected error: {e}")
                return False
        
        # Test 4: Verify bounds
        print("\n📋 Test 4: Verifying embedding bounds...")
        max_token_id = enhanced_config.number_text_tokens - 1
        print(f"   ✅ Maximum valid token ID: {max_token_id}")
        
        try:
            # Test maximum valid token
            max_token = torch.tensor([[max_token_id]], dtype=torch.long)
            with torch.no_grad():
                embedding = model.text_embedding(max_token.to(device))
            print(f"   ✅ Maximum token {max_token_id} processed successfully")
        except Exception as e:
            print(f"   ❌ Maximum token {max_token_id} failed: {e}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The IndexError has been successfully resolved!")
        print("✅ XTTS training should now work without token ID indexing errors!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_original_gpt_fix():
    """Test that the original GPT class also has the fix."""
    print("\n🔧 Testing original GPT class fix...")
    
    try:
        from trainer.xtts.layers.xtts.gpt import GPT
        
        # Create GPT with default parameters to check the default value
        model = GPT()
        print(f"   ✅ Original GPT number_text_tokens: {model.number_text_tokens}")
        
        if model.number_text_tokens < 512:
            print(f"   ⚠️  WARNING: Original GPT still uses {model.number_text_tokens}, should be ≥512")
            return False
        
        # Test that the problematic token works
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        # Test the start_text_token that was causing issues
        test_token = torch.tensor([[261]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            try:
                # This should work now
                result = model.text_embedding(test_token)
                print(f"   ✅ Original GPT handles token 261 successfully! Shape: {result.shape}")
            except Exception as e:
                print(f"   ❌ Original GPT still fails with token 261: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Original GPT test failed: {e}")
        return False

def main():
    """Run the validation tests."""
    print("🔍 XTTS IndexError Fix Validation")
    print("=" * 60)
    
    success1 = test_embedding_fix()
    success2 = test_original_gpt_fix()
    
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 60)
    
    if success1 and success2:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ The token ID indexing error has been RESOLVED")
        print("✅ XTTS training is ready to proceed")
        print("\n🚀 Next steps:")
        print("   • Run actual XTTS training to confirm")
        print("   • Monitor for any remaining embedding-related issues")
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("⚠️  Please review the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
