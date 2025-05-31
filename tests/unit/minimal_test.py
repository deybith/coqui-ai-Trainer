#!/usr/bin/env python3
"""
Minimal test to load XTTS config
"""
import os
import sys

try:
    print("Step 1: Importing TTS...")
    from TTS.tts.configs.xtts_config import XttsConfig
    print("✓ TTS config imported")
    
    print("Step 2: Loading config...")
    config_path = "output_fixed/run/training/GPT_XTTS_FT-May-30-2025_11+34PM-5d91e8d/config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    config = XttsConfig()
    config.load_json(config_path)
    print("✓ Config loaded successfully")
    
    print("Step 3: Checking config attributes...")
    print(f"Model name: {getattr(config, 'model_name', 'unknown')}")
    print(f"Sample rate: {getattr(config.audio, 'sample_rate', 'unknown')}")
    
    print("✓ All steps completed successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
