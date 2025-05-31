#!/usr/bin/env python3
"""
Debug script to understand tensor size mismatches in Phase 2 Enhanced XTTS model
"""

import torch
import torch.nn.functional as F
import sys
sys.path.append('.')

from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT

def debug_tensor_sizes():
    """Debug tensor sizes in Phase 2 model forward pass"""
    
    print("🔍 Debug: Phase 2 Enhanced XTTS Tensor Sizes")
    print("=" * 60)
    
    # Create model with minimal configuration
    model = Phase2EnhancedGPT(
        layers=2,  # Minimal layers for testing
        d_model=256,  # Smaller model
        heads=4,
        max_text_tokens=60,
        max_mel_tokens=120,
        max_prompt_tokens=35,
        use_phase2_enhancements=True,
    )
    
    # Create properly sized dummy inputs
    batch_size = 2
    text_seq_len = 30
    audio_seq_len = 60
    cond_mel_frames = 12  # Very small conditioning
    
    print(f"Input dimensions:")
    print(f"  batch_size: {batch_size}")
    print(f"  text_seq_len: {text_seq_len}")
    print(f"  audio_seq_len: {audio_seq_len}")
    print(f"  cond_mel_frames: {cond_mel_frames}")
    
    dummy_batch = {
        'text_inputs': torch.randint(0, 512, (batch_size, text_seq_len)),
        'text_lengths': torch.tensor([text_seq_len, text_seq_len-5]),
        'audio_codes': torch.randint(0, 8194, (batch_size, audio_seq_len)),
        'wav_lengths': torch.tensor([22050, 20000]),
        'cond_mels': torch.randn(batch_size, 80, cond_mel_frames),
    }
    
    print(f"\nDummy batch shapes:")
    for key, value in dummy_batch.items():
        print(f"  {key}: {value.shape}")
    
    # Test individual components first
    print(f"\n🔍 Testing individual components...")
    
    try:
        with torch.no_grad():
            # Test conditioning encoder
            cond_latents = model.get_style_emb(dummy_batch['cond_mels'])
            print(f"✅ Conditioning latents shape: {cond_latents.shape}")
            
            # Test text embeddings
            text_emb = model.text_embedding(dummy_batch['text_inputs'])
            print(f"✅ Text embeddings shape: {text_emb.shape}")
            
            # Test mel embeddings
            mel_emb = model.mel_embedding(dummy_batch['audio_codes'])
            print(f"✅ Mel embeddings shape: {mel_emb.shape}")
            
            # Test position embeddings
            text_pos = model.text_pos_embedding(dummy_batch['text_inputs'])
            print(f"✅ Text position embeddings shape: {text_pos.shape}")
            
            mel_pos = model.mel_pos_embedding(dummy_batch['audio_codes'])
            print(f"✅ Mel position embeddings shape: {mel_pos.shape}")
            
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return
    
    # Test forward pass step by step
    print(f"\n🔍 Testing forward pass step by step...")
    
    try:
        with torch.no_grad():
            # Step 1: Process inputs like the forward method does
            text_inputs = dummy_batch['text_inputs']
            text_lengths = dummy_batch['text_lengths'] 
            audio_codes = dummy_batch['audio_codes']
            wav_lengths = dummy_batch['wav_lengths']
            cond_mels = dummy_batch['cond_mels']
            
            max_text_len = text_lengths.max()
            code_lengths = torch.ceil(wav_lengths / model.code_stride_len).long() + 3
            
            print(f"  max_text_len: {max_text_len}")
            print(f"  code_lengths: {code_lengths}")
            
            # Step 2: Prepare inputs (like in forward method)
            max_mel_len = code_lengths.max()
            if max_mel_len > audio_codes.shape[-1]:
                audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))
            
            # Add stop tokens
            text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=model.stop_text_token)
            audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=model.stop_audio_token)
            
            print(f"  text_inputs after padding: {text_inputs.shape}")
            print(f"  audio_codes after padding: {audio_codes.shape}")
            
            # Step 3: Set inputs and targets
            text_inputs, text_targets = model.set_inputs_and_targets(
                text_inputs, model.start_text_token, model.stop_text_token
            )
            audio_codes, mel_targets = model.set_inputs_and_targets(
                audio_codes, model.start_audio_token, model.stop_audio_token
            )
            
            print(f"  text_inputs final: {text_inputs.shape}")
            print(f"  audio_codes final: {audio_codes.shape}")
            
            # Step 4: Create attention masks
            attn_mask_cond = torch.ones(cond_mels.shape[0], cond_mels.shape[-1], dtype=torch.bool, device=text_inputs.device)
            attn_mask_text = torch.ones(text_inputs.shape[0], text_inputs.shape[1], dtype=torch.bool, device=text_inputs.device)
            attn_mask_mel = torch.ones(audio_codes.shape[0], audio_codes.shape[1], dtype=torch.bool, device=audio_codes.device)
            
            print(f"  attn_mask_cond: {attn_mask_cond.shape}")
            print(f"  attn_mask_text: {attn_mask_text.shape}")
            print(f"  attn_mask_mel: {attn_mask_mel.shape}")
            
            # Step 5: Compute embeddings
            text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
            mel_emb = model.mel_embedding(audio_codes) + model.mel_pos_embedding(audio_codes)
            
            print(f"  text_emb final: {text_emb.shape}")
            print(f"  mel_emb final: {mel_emb.shape}")
            
            # Step 6: Get conditioning latents
            cond_latents = model.get_style_emb(cond_mels).transpose(1, 2)
            print(f"  cond_latents final: {cond_latents.shape}")
            
            # Step 7: Test the problematic concatenation in get_logits
            print(f"\n🔍 Testing get_logits concatenation...")
            
            # This is where the error happens - let's trace it
            first_inputs = text_emb
            second_inputs = mel_emb
            prompt = cond_latents
            
            print(f"  first_inputs (text_emb): {first_inputs.shape}")
            print(f"  second_inputs (mel_emb): {second_inputs.shape}")
            print(f"  prompt (cond_latents): {prompt.shape}")
            
            # Try the concatenation
            offset = prompt.shape[1]
            emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
            print(f"  ✅ Embedding concatenation successful: {emb.shape}")
            
            # Try attention mask concatenation
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            print(f"  attn_mask after text+mel concat: {attn_mask.shape}")
            
            attn_mask_cond_adjusted = torch.ones(prompt.shape[0], offset, dtype=torch.bool, device=emb.device)
            print(f"  attn_mask_cond_adjusted: {attn_mask_cond_adjusted.shape}")
            
            attn_mask_final = torch.cat([attn_mask_cond_adjusted, attn_mask], dim=1)
            print(f"  ✅ Final attention mask shape: {attn_mask_final.shape}")
            
            print(f"\n✅ All tensor concatenations successful!")
            
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now test the full forward pass
    print(f"\n🔍 Testing full forward pass...")
    
    try:
        with torch.no_grad():
            outputs = model(
                dummy_batch['text_inputs'],
                dummy_batch['text_lengths'],
                dummy_batch['audio_codes'],
                dummy_batch['wav_lengths'],
                cond_mels=dummy_batch['cond_mels']
            )
            print(f"✅ Full forward pass successful!")
            if isinstance(outputs, tuple):
                print(f"  Loss text: {outputs[0].item():.4f}")
                print(f"  Loss mel: {outputs[1].item():.4f}")
                print(f"  Output shape: {outputs[2].shape}")
            
    except Exception as e:
        print(f"❌ Full forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_sizes()
