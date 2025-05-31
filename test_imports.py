#!/usr/bin/env python3
"""
Quick imports test to identify hanging issue
"""

import torch
import sys
import traceback
sys.path.append('.')

print("🔍 Testing imports step by step...")

try:
    print("1. Basic torch import - OK")
    
    print("2. Importing Phase2Config...")
    from trainer.xtts.layers.attention.phase2_integration import Phase2Config
    print("   ✅ Phase2Config imported successfully")
    
    print("3. Importing Phase2GPTConfig...")
    from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2GPTConfig
    print("   ✅ Phase2GPTConfig imported successfully")
    
    print("4. Creating basic Phase2Config...")
    basic_config = Phase2Config(d_model=128, num_heads=2)
    print(f"   ✅ Basic Phase2Config created: d_model={basic_config.d_model}")
    
    print("5. Creating Phase2GPTConfig...")
    gpt_config = Phase2GPTConfig(layers=1, d_model=128, heads=2, use_phase2_enhancements=False)
    print(f"   ✅ Phase2GPTConfig created: layers={gpt_config.layers}")
    
    print("6. Trying Phase2EnhancedGPT import...")
    from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT
    print("   ✅ Phase2EnhancedGPT imported successfully")
    
    print("7. Creating minimal model WITHOUT Phase 2 enhancements...")
    model = Phase2EnhancedGPT(
        layers=1,
        d_model=128,
        heads=2,
        max_text_tokens=20,
        max_mel_tokens=40,
        max_prompt_tokens=10,
        use_phase2_enhancements=False,  # Disable Phase 2 to test baseline
    )
    print("   ✅ Model created without Phase 2 enhancements")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    print("✅ All imports and basic model creation successful!")
    
except Exception as e:
    print(f"❌ Error at step: {e}")
    traceback.print_exc()
