#!/usr/bin/env python3
"""
Simple test to isolate the hanging issue.
Test basic imports step by step.
"""

import sys
import os
import time

# Add the project root to Python path
project_root = "/home/ubuntu/projects/coqui-ai-Trainer"
if project_root not in sys.path:
    sys.path.append(project_root)

def test_step_by_step():
    """Test imports step by step to identify where hanging occurs."""
    
    print("Step 1: Testing basic torch import...")
    try:
        import torch
        print("   ✓ torch imported successfully")
    except Exception as e:
        print(f"   ✗ torch import failed: {e}")
        return False
    
    print("Step 2: Testing torch.nn import...")
    try:
        import torch.nn as nn
        print("   ✓ torch.nn imported successfully")
    except Exception as e:
        print(f"   ✗ torch.nn import failed: {e}")
        return False
    
    print("Step 3: Testing transformers import...")
    try:
        from transformers import GPT2PreTrainedModel
        print("   ✓ transformers imported successfully")
    except Exception as e:
        print(f"   ✗ transformers import failed: {e}")
        return False
    
    print("Step 4: Testing our local imports module...")
    try:
        print("   4a: Importing ConditioningEncoder...")
        from trainer.xtts.layers.xtts.local_imports import ConditioningEncoder
        print("   ✓ ConditioningEncoder imported")
        
        print("   4b: Importing LearnedPositionEmbeddings...")
        from trainer.xtts.layers.xtts.local_imports import LearnedPositionEmbeddings
        print("   ✓ LearnedPositionEmbeddings imported")
        
        print("   4c: Importing GPT2InferenceModel...")
        from trainer.xtts.layers.xtts.local_imports import GPT2InferenceModel
        print("   ✓ GPT2InferenceModel imported")
        
        print("   4d: Testing ConditioningEncoder creation...")
        encoder = ConditioningEncoder(spec_dim=80, embedding_dim=512)
        print("   ✓ ConditioningEncoder created")
        
    except Exception as e:
        print(f"   ✗ Local imports failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    
    print("Step 5: Testing phase2_enhanced_gpt_local import...")
    try:
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt_local import create_phase2_enhanced_gpt
        print("   ✓ phase2_enhanced_gpt_local imported successfully")
    except Exception as e:
        print(f"   ✗ phase2_enhanced_gpt_local import failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    
    print("\n✓ All imports successful!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Step-by-step Import Testing")
    print("=" * 50)
    
    start_time = time.time()
    success = test_step_by_step()
    end_time = time.time()
    
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    
    if success:
        print("SUCCESS: All imports working!")
    else:
        print("FAILURE: Import issue detected.")
