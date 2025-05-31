#!/usr/bin/env python3
"""
Enhanced XTTS Language Configuration Examples
Demonstrates various language configuration scenarios for Enhanced XTTS training.
"""

import os
import json
from dataclasses import asdict
from trainer.xtts.enhanced_configs import EnhancedXTTSConfig, EnhancedGPTTrainerConfig

def create_single_language_config(language="en"):
    """Create configuration for single-language training."""
    
    config = EnhancedXTTSConfig(
        run_name=f"enhanced_xtts_{language}",
        languages=[language],  # Train on single language
        default_language=language,
        target_language=language,
        
        # Language-optimized parameters
        temperature=0.75,
        repetition_penalty=2.5,
        use_language_conditioning=True,
        language_conditioning_strength=1.0,
        
        # Enhanced training
        use_phase1=True,
        use_phase2=True,
        batch_size=4,
        epochs=1000
    )
    
    return config

def create_multilingual_config():
    """Create configuration for full multilingual training."""
    
    config = EnhancedXTTSConfig(
        run_name="enhanced_xtts_multilingual_full",
        
        # All 17 supported languages
        languages=[
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
        ],
        default_language="en",
        
        # Multilingual-specific settings
        use_language_weighted_sampler=True,
        language_weighted_sampler_alpha=1.0,
        language_conditioning_strength=1.5,
        enable_code_switching=True,
        cross_lingual_conditioning=True,
        
        # Enhanced generation for language consistency  
        temperature=0.70,          # Lower for better consistency
        repetition_penalty=2.8,    # Higher to prevent language mixing
        top_k=35,                 # More focused sampling
        top_p=0.8,
        
        # Training parameters
        batch_size=6,             # Larger batch for multilingual
        epochs=1500,              # More epochs for multilingual
        lr=3e-5,                  # Lower LR for stability
        
        # Enhanced features
        use_phase1=True,
        use_phase2=True,
        use_mamba=True,
        use_moe=True,
        use_flash_attention=True,
    )
    
    return config

def create_european_languages_config():
    """Create configuration for European languages subset."""
    
    config = EnhancedXTTSConfig(
        run_name="enhanced_xtts_european",
        
        # European languages subset
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "nl", "cs", "ru"],
        default_language="en",
        
        # Optimized for European language similarity
        language_conditioning_strength=1.2,
        temperature=0.72,
        repetition_penalty=2.6,
        
        batch_size=5,
        epochs=1200,
        use_phase1=True,
        use_phase2=True
    )
    
    return config

def create_asian_languages_config():
    """Create configuration for Asian languages subset."""
    
    config = EnhancedXTTSConfig(
        run_name="enhanced_xtts_asian",
        
        # Asian languages subset  
        languages=["zh-cn", "ja", "ko", "hi"],
        default_language="zh-cn",
        
        # Optimized for tonal/complex languages
        language_conditioning_strength=1.8,
        temperature=0.65,          # Lower for tonal precision
        repetition_penalty=3.2,    # Higher for complex scripts
        top_k=25,                 # More focused for precision
        
        # Enhanced tokenization for Asian scripts
        max_text_length=400,       # Longer for complex characters
        
        batch_size=4,
        epochs=1800,               # More training for complex languages
        use_phase1=True,
        use_phase2=True,
        use_mamba=True            # Mamba excels at sequence modeling
    )
    
    return config

def create_fine_tuning_config(base_language="en", target_language="es"):
    """Create configuration for fine-tuning from one language to another."""
    
    config = EnhancedXTTSConfig(
        run_name=f"enhanced_xtts_finetune_{base_language}_to_{target_language}",
        
        # Start with base, target specific language
        languages=[base_language, target_language],
        default_language=base_language,
        target_language=target_language,
        
        # Fine-tuning specific parameters
        lr=1e-5,                  # Lower learning rate for fine-tuning
        batch_size=2,             # Smaller batch for fine-tuning
        epochs=500,               # Fewer epochs
        
        # Enhanced transfer learning
        cross_lingual_conditioning=True,
        multilingual_speaker_adaptation=True,
        
        # Moderate parameters for transfer
        temperature=0.75,
        repetition_penalty=2.5,
        language_conditioning_strength=1.3,
        
        use_phase1=True,
        use_phase2=True
    )
    
    return config

def create_voice_cloning_config(target_language="en"):
    """Create configuration optimized for voice cloning in specific language."""
    
    config = EnhancedXTTSConfig(
        run_name=f"enhanced_xtts_voice_clone_{target_language}",
        
        languages=[target_language],
        default_language=target_language,
        
        # Voice cloning optimized parameters
        min_reference_length=3.0,
        max_reference_length=30.0,
        speaker_embedding_dim=512,
        
        # Precise generation for voice matching
        temperature=0.70,          # Balanced for voice consistency
        repetition_penalty=2.0,    # Lower to allow natural repetition
        top_k=50,                 # Broader sampling for naturalness
        top_p=0.85,
        
        # Enhanced speaker adaptation
        multilingual_speaker_adaptation=True,
        use_language_conditioning=True,
        language_conditioning_strength=1.0,
        
        batch_size=4,
        epochs=800,
        use_phase1=True,
        use_phase2=True,
        use_neural_codec=True      # Enhanced audio quality
    )
    
    return config

def save_config_examples():
    """Save example configurations to JSON files."""
    
    configs = {
        "single_language_english": create_single_language_config("en"),
        "single_language_spanish": create_single_language_config("es"), 
        "single_language_chinese": create_single_language_config("zh-cn"),
        "multilingual_full": create_multilingual_config(),
        "european_languages": create_european_languages_config(),
        "asian_languages": create_asian_languages_config(),
        "fine_tune_en_to_es": create_fine_tuning_config("en", "es"),
        "fine_tune_en_to_zh": create_fine_tuning_config("en", "zh-cn"),
        "voice_clone_english": create_voice_cloning_config("en"),
        "voice_clone_mandarin": create_voice_cloning_config("zh-cn")
    }
    
    os.makedirs("./config/examples", exist_ok=True)
    
    for name, config in configs.items():
        config_dict = asdict(config)
        with open(f"./config/examples/{name}.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: ./config/examples/{name}.json")

def print_language_optimization_guide():
    """Print optimization guidelines for different languages."""
    
    guide = """
🌍 Enhanced XTTS Language Optimization Guide

┌─ LANGUAGE-SPECIFIC PARAMETER RECOMMENDATIONS ─┐

📍 ENGLISH (en)
   • Temperature: 0.75        • Rep. Penalty: 2.5
   • Top-K: 40               • Top-P: 0.8
   • Lang. Conditioning: 1.0  • Notes: Baseline settings

📍 SPANISH (es) / FRENCH (fr) / ITALIAN (it)
   • Temperature: 0.70        • Rep. Penalty: 2.8  
   • Top-K: 35               • Top-P: 0.85
   • Lang. Conditioning: 1.2  • Notes: Romance languages

📍 GERMAN (de)
   • Temperature: 0.68        • Rep. Penalty: 3.0
   • Top-K: 30               • Top-P: 0.8
   • Lang. Conditioning: 1.3  • Notes: Complex compound words

📍 CHINESE (zh-cn)
   • Temperature: 0.65        • Rep. Penalty: 3.2
   • Top-K: 25               • Top-P: 0.75
   • Lang. Conditioning: 1.5  • Notes: Tonal language, logographic

📍 JAPANESE (ja)
   • Temperature: 0.65        • Rep. Penalty: 3.0
   • Top-K: 30               • Top-P: 0.75  
   • Lang. Conditioning: 1.4  • Notes: Mixed scripts, pitch accent

📍 ARABIC (ar)
   • Temperature: 0.70        • Rep. Penalty: 2.8
   • Top-K: 35               • Top-P: 0.8
   • Lang. Conditioning: 1.3  • Notes: RTL script, root system

📍 HINDI (hi)
   • Temperature: 0.68        • Rep. Penalty: 2.9
   • Top-K: 30               • Top-P: 0.8
   • Lang. Conditioning: 1.3  • Notes: Devanagari script

└────────────────────────────────────────────────┘

🎯 TRAINING RECOMMENDATIONS:

🔹 SINGLE LANGUAGE: 800-1000 epochs, batch_size=4
🔹 MULTILINGUAL: 1200-1500 epochs, batch_size=6
🔹 FINE-TUNING: 300-500 epochs, lr=1e-5
🔹 VOICE CLONING: 600-800 epochs, enhanced speaker adaptation

🚀 PHASE 2 FEATURES FOR LANGUAGES:

✅ English: All features recommended
✅ European: Phase 1+2, MoE, FlashAttention  
✅ Asian: Phase 1+2, Mamba, Neural Codec
✅ Arabic/RTL: Phase 1+2, enhanced text processing
✅ Voice Cloning: Neural Codec, Quality Monitoring
"""
    print(guide)

if __name__ == "__main__":
    print("🎌 Enhanced XTTS Language Configuration Generator")
    print("=" * 60)
    
    # Save configuration examples
    save_config_examples()
    
    print()
    print_language_optimization_guide()
    
    print("\n📂 Example configurations saved to ./config/examples/")
    print("🔧 Use these as templates for your language-specific training!")
