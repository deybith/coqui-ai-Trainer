#!/usr/bin/env python3
"""
Debug script to check special token IDs in XTTS tokenizer
"""

import sys
import os

# Add the trainer directory to the path
sys.path.insert(0, '/home/ubuntu/projects/coqui-ai-Trainer')

print("=== XTTS Tokenizer Special Token Analysis ===")

try:
    from trainer.xtts.layers.xtts.tokenizer import VoiceBpeTokenizer
    print("✅ Successfully imported VoiceBpeTokenizer")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = VoiceBpeTokenizer()
    print("✅ Tokenizer initialized")
    
    # Get basic info
    vocab_size = tokenizer.get_number_tokens()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Valid token IDs: 0 to {vocab_size - 1}")
    
    # Check special tokens
    if tokenizer.tokenizer is not None:
        print("Checking special tokens...")
        start_token_id = tokenizer.tokenizer.token_to_id("[START]")
        stop_token_id = tokenizer.tokenizer.token_to_id("[STOP]")
        
        print(f"\nSpecial Tokens:")
        print(f"[START] token ID: {start_token_id}")
        print(f"[STOP] token ID: {stop_token_id}")
        
        # Check if special tokens are within vocabulary bounds
        print(f"\nBounds Check:")
        if start_token_id is not None:
            if start_token_id < vocab_size:
                print(f"✅ [START] token ({start_token_id}) is within vocab bounds")
            else:
                print(f"❌ [START] token ({start_token_id}) exceeds vocab bounds ({vocab_size})")
        else:
            print("❌ [START] token not found in tokenizer")
            
        if stop_token_id is not None:
            if stop_token_id < vocab_size:
                print(f"✅ [STOP] token ({stop_token_id}) is within vocab bounds")
            else:
                print(f"❌ [STOP] token ({stop_token_id}) exceeds vocab bounds ({vocab_size})")
        else:
            print("❌ [STOP] token not found in tokenizer")
        
    else:
        print("❌ Tokenizer not properly initialized")
        
except Exception as e:
    print(f"❌ Error during tokenizer analysis: {e}")
    import traceback
    traceback.print_exc()
