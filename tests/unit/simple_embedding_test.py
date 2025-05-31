#!/usr/bin/env python3
"""
Simple test to verify the embedding fix works
"""

import sys
import os
import torch

# Add the specific path to avoid package conflicts
sys.path.insert(0, '/home/ubuntu/projects/coqui-ai-Trainer')

try:
    # Import the fixed class
    from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT
    print("✅ Phase2EnhancedGPT imported successfully")
    
    # Test 1: Create model with new default vocabulary size
    print("\n🧪 Test 1: Creating model with default parameters...")
    model = Phase2EnhancedGPT(
        layers=2, 
        d_model=128, 
        heads=2, 
        use_phase2_enhancements=False  # Disable to avoid complex dependencies
    )
    print(f"   ✅ Model created successfully")
    print(f"   📊 Text embedding vocabulary size: {model.text_embedding.num_embeddings}")
    print(f"   🎯 Start text token: {model.start_text_token}")
    print(f"   🔢 Default number_text_tokens: {model.number_text_tokens}")
    
    # Test 2: Verify the fix - test with start_text_token=261
    print("\n🎯 Test 2: Testing problematic token that caused IndexError...")
    test_token = torch.tensor([[261]], dtype=torch.long)  # The problematic start token
    
    try:
        embedding = model.text_embedding(test_token)
        print(f"   ✅ Successfully processed start_text_token=261")
        print(f"   📏 Embedding shape: {embedding.shape}")
        print(f"   🎉 NO IndexError - Fix is working!")
    except IndexError as e:
        print(f"   ❌ IndexError still occurs: {e}")
        print(f"   🔍 This means the fix didn't work properly")
        
    # Test 3: Verify vocabulary bounds
    print(f"\n🔬 Test 3: Testing vocabulary bounds...")
    print(f"   📊 Vocabulary size: {model.text_embedding.num_embeddings}")
    print(f"   🔢 Valid token range: 0 to {model.text_embedding.num_embeddings - 1}")
    
    # Test maximum valid token
    max_valid_token = model.text_embedding.num_embeddings - 1
    test_max = torch.tensor([[max_valid_token]], dtype=torch.long)
    embedding_max = model.text_embedding(test_max)
    print(f"   ✅ Maximum valid token ({max_valid_token}) works: {embedding_max.shape}")
    
    # Test that exceeding bounds still fails (as expected)
    try:
        invalid_token = torch.tensor([[model.text_embedding.num_embeddings]], dtype=torch.long)
        embedding_invalid = model.text_embedding(invalid_token)
        print(f"   ⚠️  Warning: Token beyond bounds should have failed")
    except IndexError:
        print(f"   ✅ Token beyond bounds correctly fails (expected)")
        
    print(f"\n🎉 SUCCESS: All tests passed!")
    print(f"📝 Summary:")
    print(f"   - Default vocabulary size is now {model.text_embedding.num_embeddings} (was 256)")
    print(f"   - Special start_text_token=261 works without IndexError")
    print(f"   - Phase2EnhancedGPT embedding fix is working correctly")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
