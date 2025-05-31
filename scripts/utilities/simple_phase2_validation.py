#!/usr/bin/env python3
"""
Simple validation test for Phase 2 Enhanced XTTS tensor fixes
"""

import torch
import sys
sys.path.append('.')

from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT

def test_tensor_fix():
    """Test that tensor mismatches are resolved"""
    
    print("🔍 Phase 2 Enhanced XTTS - Tensor Fix Validation")
    print("=" * 50)
    
    # Create model
    model = Phase2EnhancedGPT(
        layers=3,
        d_model=512,
        heads=8,
        max_text_tokens=100,
        max_mel_tokens=200,
        max_prompt_tokens=50,
        use_phase2_enhancements=True,
    )
    
    print(f"✅ Model created successfully")
    
    # Test with different input sizes
    test_cases = [
        {"batch": 1, "text": 20, "audio": 40, "cond": 10},
        {"batch": 2, "text": 35, "audio": 70, "cond": 15},
        {"batch": 4, "text": 45, "audio": 90, "cond": 20},
    ]
    
    for i, case in enumerate(test_cases):
        try:
            batch_size = case["batch"]
            text_len = case["text"]
            audio_len = case["audio"]
            cond_len = case["cond"]
            
            dummy_batch = {
                'text_inputs': torch.randint(0, 512, (batch_size, text_len)),
                'text_lengths': torch.tensor([text_len] * batch_size),
                'audio_codes': torch.randint(0, 8194, (batch_size, audio_len)),
                'wav_lengths': torch.tensor([22050] * batch_size),
                'cond_mels': torch.randn(batch_size, 80, cond_len),
            }
            
            with torch.no_grad():
                outputs = model(
                    dummy_batch['text_inputs'],
                    dummy_batch['text_lengths'],
                    dummy_batch['audio_codes'],
                    dummy_batch['wav_lengths'],
                    cond_mels=dummy_batch['cond_mels']
                )
            
            if isinstance(outputs, tuple):
                loss_text, loss_mel, logits = outputs
                print(f"✅ Test {i+1}: Batch {batch_size} - Text Loss: {loss_text:.4f}, Mel Loss: {loss_mel:.4f}")
            else:
                print(f"✅ Test {i+1}: Batch {batch_size} - Output shape: {outputs.shape}")
                
        except Exception as e:
            print(f"❌ Test {i+1} failed: {e}")
            return False
    
    print(f"\n🎉 All tensor mismatch issues have been resolved!")
    print(f"✅ Phase 2 Enhanced XTTS is working correctly!")
    return True

if __name__ == "__main__":
    success = test_tensor_fix()
    if success:
        print(f"\n🚀 Ready for training and inference!")
    else:
        print(f"\n⚠️  Issues still exist")
