#!/usr/bin/env python3
"""
Debug Phase 2 tensor size mismatch using actual training configuration.
This script will try to reproduce the tensor size mismatch error.
"""

import sys
import os
import torch
import json
import traceback

# Add the project root to Python path
project_root = "/home/ubuntu/projects/coqui-ai-Trainer"
if project_root not in sys.path:
    sys.path.append(project_root)

def debug_tensor_mismatch():
    """Debug the tensor size mismatch issue with Phase 2 model."""
    
    print("🔍 Debugging Phase 2 tensor size mismatch...")
    
    try:
        # Load the Phase 2 configuration
        config_path = "phase2_output/phase2_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        print(f"✅ Loaded config from {config_path}")
        
        # Import Phase 2 components with explicit error handling  
        print("📦 Importing Phase 2 components...")
        from train_phase2_xtts import Phase2EnhancedXTTSModel
        
        # Convert dict to object for easier access
        class ConfigObject:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, ConfigObject(value))
                    else:
                        setattr(self, key, value)
        
        config = ConfigObject(config_dict)
        print("✅ Configuration converted to object")
        
        # Create the model
        print("🔧 Creating Phase 2 Enhanced XTTS model...")
        model = Phase2EnhancedXTTSModel(config)
        print("✅ Model created successfully")
        
        # Put model in eval mode to avoid training-specific operations
        model.eval()
        
        # Create test inputs with realistic sizes that might trigger the issue
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"🔧 Using device: {device}")
        
        # Test with different input sizes to trigger the tensor mismatch
        test_cases = [
            {
                "name": "Small batch",
                "batch_size": 2,
                "text_seq_len": 50,
                "audio_seq_len": 80,
                "conditioning_frames": 24
            },
            {
                "name": "Medium batch", 
                "batch_size": 2,
                "text_seq_len": 100,
                "audio_seq_len": 150,
                "conditioning_frames": 50
            },
            {
                "name": "Large conditioning",
                "batch_size": 2, 
                "text_seq_len": 50,
                "audio_seq_len": 80,
                "conditioning_frames": 2048  # This might trigger the error
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            print(f"   Batch size: {test_case['batch_size']}")
            print(f"   Text length: {test_case['text_seq_len']}")
            print(f"   Audio length: {test_case['audio_seq_len']}")  
            print(f"   Conditioning frames: {test_case['conditioning_frames']}")
            
            try:
                batch_size = test_case['batch_size']
                text_seq_len = test_case['text_seq_len']
                audio_seq_len = test_case['audio_seq_len']
                conditioning_frames = test_case['conditioning_frames']
                
                # Create batch in the format expected by Phase 2 model
                batch = {
                    'text_inputs': torch.randint(0, 512, (batch_size, text_seq_len), device=device),
                    'text_lengths': torch.tensor([text_seq_len] * batch_size, device=device),
                    'audio_codes': torch.randint(0, 8194, (batch_size, audio_seq_len), device=device),
                    'wav_lengths': torch.tensor([22050] * batch_size, device=device),  # 1 second
                    'cond_mels': torch.randn(batch_size, 80, conditioning_frames, device=device),
                }
                
                print(f"   Created inputs:")
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"     {key}: {value.shape}")
                    else:
                        print(f"     {key}: {value}")
                
                # Test forward pass
                with torch.no_grad():
                    outputs = model(batch)
                
                print(f"   ✅ Forward pass successful!")
                if isinstance(outputs, tuple):
                    print(f"      Output shapes: {[out.shape if torch.is_tensor(out) else type(out) for out in outputs]}")
                elif isinstance(outputs, dict):
                    print(f"      Output keys: {list(outputs.keys())}")
                else:
                    print(f"      Output type: {type(outputs)}")
                    
            except RuntimeError as e:
                error_msg = str(e)
                print(f"   ❌ RuntimeError: {error_msg}")
                
                # Check if this is the tensor size mismatch we're looking for
                if "size of tensor a" in error_msg and "must match the size of tensor b" in error_msg:
                    print(f"   🎯 FOUND THE TENSOR SIZE MISMATCH ERROR!")
                    print(f"   📋 Error details: {error_msg}")
                    
                    # Try to extract more information about the error
                    print(f"   🔍 Analyzing error location...")
                    traceback.print_exc()
                    
                    return True  # Found the error we're looking for
                else:
                    print(f"   🔍 Different error, continuing...")
            
            except Exception as e:
                print(f"   ❌ Other error: {e}")
                traceback.print_exc()
        
        print("\n✅ All test cases completed without the target tensor mismatch error.")
        return False
        
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2 Enhanced XTTS Tensor Size Mismatch Debugger")
    print("=" * 70)
    
    found_error = debug_tensor_mismatch()
    
    print("\n" + "=" * 70)
    if found_error:
        print("🎯 SUCCESS: Found and analyzed the tensor size mismatch error!")
        print("    Now we can work on fixing the root cause.")
    else:
        print("🤔 The tensor size mismatch error was not reproduced.")
        print("    It might occur only under specific training conditions.")
    print("=" * 70)
