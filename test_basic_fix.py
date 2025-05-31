#!/usr/bin/env python3
"""
Simple validation test for the embedding fix.
"""

import torch
import sys
from pathlib import Path

# Add the trainer module to the path
sys.path.insert(0, str(Path(__file__).parent / "trainer"))

def test_basic_fix():
    """Test that the basic fix is working."""
    print("🔧 Testing basic embedding fix...")
    
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT, Phase2GPTConfig
        
        # Create config
        config = Phase2GPTConfig()
        print(f"✅ Default number_text_tokens: {config.number_text_tokens}")
        
        # Create model
        model = Phase2EnhancedGPT(config)
        print(f"✅ Model created successfully")
        
        # Test the problematic token
        device = torch.device('cpu')
        model = model.to(device)
        
        # This was the token causing IndexError
        text_inputs = torch.tensor([[261]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            result = model.text_embedding(text_inputs)
            print(f"✅ Token 261 processed successfully! Result shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_fix()
    if success:
        print("🎉 Basic fix validation PASSED!")
    else:
        print("❌ Basic fix validation FAILED!")
    sys.exit(0 if success else 1)
