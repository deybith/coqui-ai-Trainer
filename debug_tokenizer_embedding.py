#!/usr/bin/env python3
"""
Debug script to investigate tokenizer vs embedding vocabulary size mismatch.
This script will help us understand why token IDs exceed the embedding table bounds.
"""

import os
import sys
import torch
import json

# Add the trainer path to sys.path
sys.path.insert(0, '/home/ubuntu/projects/coqui-ai-Trainer')

from trainer.xtts.layers.xtts.tokenizer import VoiceBpeTokenizer

def main():
    print("=== Debug Tokenizer vs Embedding Vocabulary Size ===\n")
    
    # Initialize tokenizer with the default vocabulary file  
    # First try to find the correct tokenizer file
    possible_vocab_files = [
        "/home/ubuntu/projects/coqui-ai-Trainer/trainer/xtts/layers/xtts/../data/tokenizer.json",
        "/home/ubuntu/projects/coqui-ai-Trainer/trainer/xtts/utils/assets/tortoise/tokenizer.json",
        "/home/ubuntu/projects/coqui-ai-Trainer/trainer/xtts/layers/data/tokenizer.json"
    ]
    
    default_vocab_file = None
    for vocab_file in possible_vocab_files:
        if os.path.exists(vocab_file):
            default_vocab_file = vocab_file
            break
    
    if default_vocab_file is None:
        print("ERROR: Could not find tokenizer.json file!")
        return 1
    
    print(f"Using tokenizer vocabulary file: {default_vocab_file}")
    print(f"Vocab file exists: {os.path.exists(default_vocab_file)}\n")
    
    try:
        # Initialize the tokenizer
        tokenizer = VoiceBpeTokenizer(vocab_file=default_vocab_file)
        
        # Get vocabulary information
        vocab_size = len(tokenizer)
        vocab_dict = tokenizer.tokenizer.get_vocab()
        number_tokens = tokenizer.get_number_tokens()
        
        print(f"Tokenizer vocab_size (len): {vocab_size}")
        print(f"Tokenizer number_tokens (max + 1): {number_tokens}")
        print(f"Vocab dict size: {len(vocab_dict)}")
        print(f"Max token ID in vocab: {max(vocab_dict.values())}")
        print(f"Min token ID in vocab: {min(vocab_dict.values())}")
        
        # Check for gaps in token IDs
        token_ids = sorted(vocab_dict.values())
        expected_ids = list(range(len(token_ids)))
        
        print(f"\nToken ID Analysis:")
        print(f"Expected IDs range: 0 to {len(token_ids) - 1}")
        print(f"Actual IDs range: {token_ids[0]} to {token_ids[-1]}")
        print(f"Are IDs consecutive: {token_ids == expected_ids}")
        
        if token_ids != expected_ids:
            print("WARNING: Token IDs are not consecutive!")
            missing_ids = set(expected_ids) - set(token_ids)
            extra_ids = set(token_ids) - set(expected_ids)
            if missing_ids:
                print(f"Missing IDs: {sorted(list(missing_ids))[:10]}{'...' if len(missing_ids) > 10 else ''}")
            if extra_ids:
                print(f"Extra IDs: {sorted(list(extra_ids))[:10]}{'...' if len(extra_ids) > 10 else ''}")
        
        print(f"\n=== Test Text Encoding ===")
        
        # Test some sample texts to see what token IDs are generated
        test_texts = [
            ("Hello world", "en"),
            ("This is a test", "en"),
            ("Bonjour le monde", "fr"),
            ("Hola mundo", "es"),
            ("你好世界", "zh"),
        ]
        
        for text, lang in test_texts:
            try:
                token_ids = tokenizer.encode(text, lang)
                max_token_id = max(token_ids) if token_ids else -1
                min_token_id = min(token_ids) if token_ids else -1
                
                print(f"\nText: '{text}' (lang: {lang})")
                print(f"Encoded token IDs: {token_ids}")
                print(f"Token count: {len(token_ids)}")
                print(f"Max token ID: {max_token_id}")
                print(f"Min token ID: {min_token_id}")
                
                # Check if any token ID exceeds the embedding bounds
                if max_token_id >= number_tokens:
                    print(f"🚨 ERROR: Max token ID {max_token_id} >= embedding vocab size {number_tokens}")
                
                if max_token_id >= vocab_size:
                    print(f"🚨 ERROR: Max token ID {max_token_id} >= tokenizer vocab size {vocab_size}")
                    
                # Try to decode back
                decoded = tokenizer.decode(token_ids)
                print(f"Decoded: '{decoded}'")
                
            except Exception as e:
                print(f"Error encoding '{text}' (lang: {lang}): {e}")
        
        # Check what the Phase2EnhancedGPT would expect
        print(f"\n=== Embedding Layer Configuration ===")
        print(f"self.number_text_tokens (for nn.Embedding): {number_tokens}")
        print(f"nn.Embedding will accept token IDs: 0 to {number_tokens - 1}")
        
        # Show some vocabulary samples
        print(f"\n=== Vocabulary Samples ===")
        vocab_items = list(vocab_dict.items())
        vocab_items.sort(key=lambda x: x[1])  # Sort by token ID
        
        print("First 20 tokens:")
        for token, token_id in vocab_items[:20]:
            print(f"  '{token}' -> {token_id}")
        
        print("\nLast 20 tokens:")
        for token, token_id in vocab_items[-20:]:
            print(f"  '{token}' -> {token_id}")
            
        # Look for any special tokens that might be causing issues
        special_tokens = []
        for token, token_id in vocab_dict.items():
            if any(marker in token for marker in ['[', ']', '<', '>', 'SPACE', 'STOP', 'UNK']):
                special_tokens.append((token, token_id))
        
        print(f"\nSpecial tokens found: {len(special_tokens)}")
        for token, token_id in sorted(special_tokens, key=lambda x: x[1])[:10]:
            print(f"  '{token}' -> {token_id}")
            
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
