#!/usr/bin/env python3
"""
Quick Language Configuration Demo for Enhanced XTTS
"""

# Example 1: Single Language Training (English)
english_config = {
    "run_name": "enhanced_xtts_english",
    "languages": ["en"],
    "default_language": "en",
    "target_language": "en",
    "temperature": 0.75,
    "repetition_penalty": 2.5,
    "language_conditioning_strength": 1.0,
    "batch_size": 4,
    "epochs": 1000,
    "use_phase1": True,
    "use_phase2": True
}

# Example 2: Multilingual Training (All 17 languages)
multilingual_config = {
    "run_name": "enhanced_xtts_multilingual",
    "languages": [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr",
        "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
    ],
    "default_language": "en",
    "use_language_weighted_sampler": True,
    "language_weighted_sampler_alpha": 1.0,
    "temperature": 0.70,
    "repetition_penalty": 2.8,
    "language_conditioning_strength": 1.5,
    "batch_size": 6,
    "epochs": 1500,
    "use_phase1": True,
    "use_phase2": True,
    "use_mamba": True,
    "use_moe": True
}

# Example 3: Asian Languages Specialized
asian_config = {
    "run_name": "enhanced_xtts_asian",
    "languages": ["zh-cn", "ja", "ko", "hi"],
    "default_language": "zh-cn",
    "temperature": 0.65,
    "repetition_penalty": 3.2,
    "top_k": 25,
    "language_conditioning_strength": 1.8,
    "max_text_length": 400,
    "batch_size": 4,
    "epochs": 1800,
    "use_phase1": True,
    "use_phase2": True,
    "use_mamba": True
}

# Example 4: Fine-tuning Configuration  
fine_tune_config = {
    "run_name": "enhanced_xtts_finetune_en_to_es",
    "languages": ["en", "es"],
    "default_language": "en", 
    "target_language": "es",
    "lr": 1e-5,
    "batch_size": 2,
    "epochs": 500,
    "temperature": 0.75,
    "repetition_penalty": 2.5,
    "cross_lingual_conditioning": True,
    "use_phase1": True,
    "use_phase2": True
}

print("🌍 Enhanced XTTS Language Configuration Examples")
print("=" * 60)

print("\n📍 Example 1: Single Language (English)")
print("   Languages:", english_config["languages"])
print("   Temperature:", english_config["temperature"])
print("   Repetition Penalty:", english_config["repetition_penalty"])

print("\n📍 Example 2: Full Multilingual (17 languages)")
print("   Languages:", len(multilingual_config["languages"]), "total")
print("   Sample:", multilingual_config["languages"][:5], "...")
print("   Language Conditioning:", multilingual_config["language_conditioning_strength"])

print("\n📍 Example 3: Asian Languages Specialized")
print("   Languages:", asian_config["languages"])
print("   Optimized for tonal/complex scripts")
print("   Temperature:", asian_config["temperature"], "(lower for precision)")

print("\n📍 Example 4: Fine-tuning (English → Spanish)")
print("   Base Language:", fine_tune_config["default_language"])
print("   Target Language:", fine_tune_config["target_language"])
print("   Learning Rate:", fine_tune_config["lr"], "(lower for fine-tuning)")

print("\n🎯 Key Language Parameters:")
print("   • languages: List of language codes to support")
print("   • default_language: Primary language for training")
print("   • target_language: Specific language to optimize for")
print("   • language_conditioning_strength: How strongly to enforce language")
print("   • temperature: Generation creativity (lower = more consistent)")
print("   • repetition_penalty: Prevents language mixing (higher = less mixing)")

print("\n✨ Enhanced Phase 2 Features for Languages:")
print("   • use_mamba: Excellent for sequence modeling (Asian languages)")
print("   • use_moe: Multi-expert routing for language specialization")
print("   • use_neural_codec: Enhanced audio quality for all languages")
print("   • cross_lingual_conditioning: Transfer learning between languages")

print("\n🚀 To use these configurations:")
print("   1. Save as JSON file in ./config/ directory")
print("   2. Use with train_full_enhanced_xtts.py")
print("   3. Or integrate into Python training script")

print("\n📂 See train_multilingual_enhanced_xtts.py for complete examples!")
