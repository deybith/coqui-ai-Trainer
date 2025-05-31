#!/usr/bin/env python3
"""
Simple debug to test Phase 2 model initialization
"""

import torch
import sys
sys.path.append('.')

print("🔍 Simple Phase 2 Debug Test")
print("=" * 40)

try:
    print("📦 Importing Phase2EnhancedGPT...")
    from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedGPT
    print("✅ Import successful")
    
    print("🏗️  Creating minimal model...")
    model = Phase2EnhancedGPT(
        layers=1,  # Minimal
        d_model=128,  # Very small
        heads=2,
        max_text_tokens=30,
        max_mel_tokens=60,
        max_prompt_tokens=15,
        use_phase2_enhancements=True,
    )
    print("✅ Model created successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("🧪 Creating tiny test batch...")
    batch_size = 1
    text_seq_len = 10
    audio_seq_len = 20
    cond_mel_frames = 5
    
    dummy_batch = {
        'text_inputs': torch.randint(0, 100, (batch_size, text_seq_len)),
        'text_lengths': torch.tensor([text_seq_len]),
        'audio_codes': torch.randint(0, 1000, (batch_size, audio_seq_len)),
        'wav_lengths': torch.tensor([11025]),  # Half of 22050
        'cond_mels': torch.randn(batch_size, 80, cond_mel_frames),
    }
    
    print("✅ Batch created")
    for key, value in dummy_batch.items():
        print(f"   {key}: {value.shape}")
    
    print("🚀 Testing forward pass...")
    with torch.no_grad():
        outputs = model(
            dummy_batch['text_inputs'],
            dummy_batch['text_lengths'],
            dummy_batch['audio_codes'],
            dummy_batch['wav_lengths'],
            cond_mels=dummy_batch['cond_mels']
        )
        print("✅ Forward pass successful!")
        if isinstance(outputs, tuple):
            print(f"   Loss text: {outputs[0].item():.4f}")
            print(f"   Loss mel: {outputs[1].item():.4f}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
