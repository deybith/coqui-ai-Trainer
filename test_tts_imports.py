#!/usr/bin/env python3
"""
Test TTS imports to identify missing dependencies
"""

import sys
sys.path.append('.')

print("🔍 Testing TTS imports...")

try:
    print("1. Testing TTS import...")
    import TTS
    print("   ✅ TTS module available")
except ImportError as e:
    print(f"   ❌ TTS module not available: {e}")

try:
    print("2. Testing tortoise autoregressive import...")
    from TTS.tts.layers.tortoise.autoregressive import ConditioningEncoder
    print("   ✅ ConditioningEncoder imported")
except ImportError as e:
    print(f"   ❌ ConditioningEncoder import failed: {e}")

try:
    print("3. Testing XTTS gpt_inference import...")
    from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
    print("   ✅ GPT2InferenceModel imported")
except ImportError as e:
    print(f"   ❌ GPT2InferenceModel import failed: {e}")

try:
    print("4. Testing perceiver encoder import...")
    from TTS.tts.layers.xtts.perceiver_encoder import PerceiverResampler
    print("   ✅ PerceiverResampler imported")
except ImportError as e:
    print(f"   ❌ PerceiverResampler import failed: {e}")

print("\n🔍 Checking local alternatives...")

try:
    print("5. Testing local tortoise import...")
    from trainer.xtts.layers.tortoise.autoregressive import ConditioningEncoder
    print("   ✅ Local ConditioningEncoder found")
except ImportError as e:
    print(f"   ❌ Local ConditioningEncoder not found: {e}")

try:
    print("6. Testing local gpt_inference import...")
    from trainer.xtts.layers.xtts.gpt_inference import GPT2InferenceModel
    print("   ✅ Local GPT2InferenceModel found")
except ImportError as e:
    print(f"   ❌ Local GPT2InferenceModel not found: {e}")

try:
    print("7. Testing local perceiver import...")
    from trainer.xtts.layers.xtts.perceiver_encoder import PerceiverResampler
    print("   ✅ Local PerceiverResampler found")
except ImportError as e:
    print(f"   ❌ Local PerceiverResampler not found: {e}")
