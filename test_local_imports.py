#!/usr/bin/env python3
"""
Test the local imports version of Phase 2 Enhanced GPT
to verify it initializes without hanging on external TTS dependencies.
"""

import sys
import os
import torch
import traceback

# Add the project root to Python path
project_root = "/home/ubuntu/projects/coqui-ai-Trainer"
if project_root not in sys.path:
    sys.path.append(project_root)

def test_local_imports():
    """Test that we can import and initialize the local imports version."""
    print("Testing local imports version of Phase 2 Enhanced GPT...")
    
    try:
        print("1. Importing local imports...")
        from trainer.xtts.layers.xtts.local_imports import (
            ConditioningEncoder,
            LearnedPositionEmbeddings, 
            GPT2InferenceModel,
            PerceiverResampler,
            _prepare_attention_mask_for_generation
        )
        print("   ✓ Local imports successful")
        
        print("2. Importing Phase 2 Enhanced GPT local version...")
        from trainer.xtts.layers.xtts.phase2_enhanced_gpt_local import create_phase2_enhanced_gpt
        print("   ✓ Phase 2 Enhanced GPT local import successful")
        
        print("3. Creating Phase 2 Enhanced GPT model...")
        config = {
            'max_conditioning_length': 132300,
            'max_audio_length': 132300,
            'max_text_length': 402,
            'max_prompt_length': 132300,
            'gpt_code_length': 1024,
            'layers': 30,
            'model_dim': 1024,
            'heads': 16,
            'number_text_tokens': 256,
            'start_text_token': 255,
            'stop_text_token': 0,
            'number_mel_codes': 8194,
            'start_mel_token': 8192,
            'stop_mel_token': 8193,
            'train_solo_embeddings': False,
            'use_mel_codes_as_input': True,
            'checkpointing': False,
            
            # Phase 2 Enhanced settings
            'use_phase2_enhanced': True,
            'use_mamba': True,
            'use_moe': True,
            'use_flash_attention': True,
            'use_rope': True,
            'use_neural_codec': True,
            'use_quality_monitor': True,
            'moe_num_experts': 8,
            'moe_top_k': 2,
            'mamba_d_state': 16,
            'mamba_d_conv': 4,
            'mamba_expand': 2,
        }
        
        model = create_phase2_enhanced_gpt(config)
        print("   ✓ Phase 2 Enhanced GPT model created successfully")
        
        print("4. Testing model properties...")
        print(f"   - Model device: {next(model.parameters()).device}")
        print(f"   - Model dtype: {next(model.parameters()).dtype}")
        print(f"   - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print("5. Testing simple forward pass with small inputs...")
        device = next(model.parameters()).device
        
        # Create minimal test inputs
        batch_size = 1
        text_seq_len = 10
        audio_seq_len = 20
        
        text_inputs = torch.randint(0, 256, (batch_size, text_seq_len), device=device)
        text_lengths = torch.tensor([text_seq_len], device=device)
        audio_codes = torch.randint(0, 8194, (batch_size, audio_seq_len), device=device)
        audio_lengths = torch.tensor([audio_seq_len], device=device)
        conditioning_input = torch.randn(batch_size, 80, 10, device=device)  # mel features
        
        print(f"   - Text inputs shape: {text_inputs.shape}")
        print(f"   - Audio codes shape: {audio_codes.shape}")  
        print(f"   - Conditioning input shape: {conditioning_input.shape}")
        
        try:
            # Test forward pass
            with torch.no_grad():
                outputs = model(
                    text_inputs=text_inputs,
                    text_lengths=text_lengths,
                    audio_codes=audio_codes,
                    audio_lengths=audio_lengths,
                    conditioning_input=conditioning_input,
                    return_attentions=False
                )
            
            print(f"   ✓ Forward pass successful!")
            print(f"   - Output shape: {outputs.shape}")
            
        except Exception as e:
            print(f"   ✗ Forward pass failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False
            
        print("\n✓ All tests passed! Local imports version is working.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Phase 2 Enhanced GPT Local Imports Version")
    print("=" * 60)
    
    success = test_local_imports()
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Local imports version is working!")
        print("We can now proceed with tensor size debugging.")
    else:
        print("FAILURE: Local imports version has issues.")
        print("Need to fix local implementation before proceeding.")
    print("=" * 60)
