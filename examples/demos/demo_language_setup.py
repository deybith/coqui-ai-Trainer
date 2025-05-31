#!/usr/bin/env python3
"""
Enhanced XTTS Language Configuration Demo
Shows how to configure and train models for different language scenarios.
"""

import json
import os
from pathlib import Path

def create_sample_data_csv(language, output_path, num_samples=100):
    """Create sample training data CSV for demonstration."""
    
    # Sample texts by language
    sample_texts = {
        "en": [
            "Hello, welcome to the enhanced text-to-speech system.",
            "This model supports multiple languages and high-quality voice synthesis.",
            "The weather is beautiful today, perfect for a walk in the park.",
            "Machine learning has revolutionized artificial intelligence applications."
        ],
        "es": [
            "Hola, bienvenido al sistema mejorado de texto a voz.",
            "Este modelo admite múltiples idiomas y síntesis de voz de alta calidad.",
            "El clima está hermoso hoy, perfecto para caminar en el parque.",
            "El aprendizaje automático ha revolucionado las aplicaciones de inteligencia artificial."
        ],
        "fr": [
            "Bonjour, bienvenue dans le système de synthèse vocale amélioré.",
            "Ce modèle prend en charge plusieurs langues et une synthèse vocale de haute qualité.",
            "Le temps est magnifique aujourd'hui, parfait pour une promenade dans le parc.",
            "L'apprentissage automatique a révolutionné les applications d'intelligence artificielle."
        ],
        "de": [
            "Hallo, willkommen zum verbesserten Text-zu-Sprache-System.",
            "Dieses Modell unterstützt mehrere Sprachen und hochwertige Sprachsynthese.",
            "Das Wetter ist heute wunderschön, perfekt für einen Spaziergang im Park.",
            "Maschinelles Lernen hat Anwendungen der künstlichen Intelligenz revolutioniert."
        ],
        "zh-cn": [
            "您好，欢迎使用增强的文本转语音系统。",
            "该模型支持多种语言和高质量的语音合成。",
            "今天天气很好，非常适合在公园里散步。",
            "机器学习已经彻底改变了人工智能应用。"
        ]
    }
    
    texts = sample_texts.get(language, sample_texts["en"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("audio_file,text,speaker_name,language,emotion\n")
        
        for i in range(num_samples):
            text = texts[i % len(texts)]
            f.write(f"audio/sample_{language}_{i:03d}.wav,\"{text}\",speaker_demo,{language},neutral\n")
    
    print(f"✅ Created sample data: {output_path}")
    return output_path

def demo_single_language_config():
    """Demonstrate single language configuration."""
    
    print("\n🎯 DEMO 1: Single Language Configuration (English)")
    print("=" * 60)
    
    config = {
        "run_name": "enhanced_xtts_english_demo",
        "model": "enhanced_xtts",
        "run_description": "Single language English training demo",
        
        # Language configuration
        "languages": ["en"],
        "default_language": "en",
        "target_language": "en",
        
        # Optimized for English
        "temperature": 0.75,
        "repetition_penalty": 2.5,
        "top_k": 40,
        "top_p": 0.8,
        "language_conditioning_strength": 1.0,
        
        # Training parameters
        "batch_size": 4,
        "epochs": 1000,
        "lr": 5e-5,
        
        # Enhanced features
        "use_phase1": True,
        "use_phase2": True,
        "use_flash_attention": True,
        
        # Paths
        "train_csv": "./data/demo/english_train.csv",
        "eval_csv": "./data/demo/english_eval.csv",
        "output_path": "./models/demo/enhanced_english"
    }
    
    # Save configuration
    config_path = "./config/demo_english_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create sample data
    create_sample_data_csv("en", "./data/demo/english_train.csv", 200)
    create_sample_data_csv("en", "./data/demo/english_eval.csv", 50)
    
    print(f"📁 Configuration saved: {config_path}")
    print("🚀 To train: python train_full_enhanced_xtts.py --config config/demo_english_config.json")
    
    return config

def demo_multilingual_config():
    """Demonstrate multilingual configuration."""
    
    print("\n🌍 DEMO 2: Multilingual Configuration (5 Languages)")
    print("=" * 60)
    
    config = {
        "run_name": "enhanced_xtts_multilingual_demo",
        "model": "enhanced_xtts",
        "run_description": "Multilingual training demo with 5 languages",
        
        # Multilingual configuration
        "languages": ["en", "es", "fr", "de", "zh-cn"],
        "default_language": "en",
        
        # Multilingual optimization
        "use_language_weighted_sampler": True,
        "language_weighted_sampler_alpha": 1.0,
        "language_conditioning_strength": 1.5,
        "cross_lingual_conditioning": True,
        "enable_code_switching": True,
        
        # Optimized for multilingual consistency
        "temperature": 0.70,
        "repetition_penalty": 2.8,
        "top_k": 35,
        "top_p": 0.8,
        
        # Training parameters
        "batch_size": 6,
        "epochs": 1500,
        "lr": 3e-5,
        
        # Enhanced features for multilingual
        "use_phase1": True,
        "use_phase2": True,
        "use_mamba": True,
        "use_moe": True,
        "use_flash_attention": True,
        "moe_num_experts": 8,
        
        # Quality monitoring
        "enable_language_quality_monitoring": True,
        "per_language_validation": True,
        
        # Paths
        "train_csv": "./data/demo/multilingual_train.csv",
        "eval_csv": "./data/demo/multilingual_eval.csv",
        "output_path": "./models/demo/enhanced_multilingual"
    }
    
    # Save configuration
    config_path = "./config/demo_multilingual_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create multilingual sample data
    multilingual_train_path = "./data/demo/multilingual_train.csv"
    multilingual_eval_path = "./data/demo/multilingual_eval.csv"
    
    # Create combined multilingual CSV
    os.makedirs(os.path.dirname(multilingual_train_path), exist_ok=True)
    
    with open(multilingual_train_path, 'w', encoding='utf-8') as f:
        f.write("audio_file,text,speaker_name,language,emotion\n")
        
        for lang in ["en", "es", "fr", "de", "zh-cn"]:
            temp_file = f"./temp_{lang}_train.csv"
            create_sample_data_csv(lang, temp_file, 100)
            
            with open(temp_file, 'r', encoding='utf-8') as temp_f:
                lines = temp_f.readlines()[1:]  # Skip header
                f.writelines(lines)
            
            os.remove(temp_file)
    
    # Create eval data
    with open(multilingual_eval_path, 'w', encoding='utf-8') as f:
        f.write("audio_file,text,speaker_name,language,emotion\n")
        
        for lang in ["en", "es", "fr", "de", "zh-cn"]:
            temp_file = f"./temp_{lang}_eval.csv"
            create_sample_data_csv(lang, temp_file, 20)
            
            with open(temp_file, 'r', encoding='utf-8') as temp_f:
                lines = temp_f.readlines()[1:]  # Skip header
                f.writelines(lines)
            
            os.remove(temp_file)
    
    print(f"📁 Configuration saved: {config_path}")
    print("🚀 To train: python train_multilingual_enhanced_xtts.py --train_csv data/demo/multilingual_train.csv --eval_csv data/demo/multilingual_eval.csv")
    
    return config

def demo_asian_languages_config():
    """Demonstrate Asian languages specialized configuration."""
    
    print("\n🏮 DEMO 3: Asian Languages Specialized Configuration")
    print("=" * 60)
    
    config = {
        "run_name": "enhanced_xtts_asian_demo",
        "model": "enhanced_xtts",
        "run_description": "Asian languages specialized training demo",
        
        # Asian languages subset
        "languages": ["zh-cn", "ja", "ko"],
        "default_language": "zh-cn",
        
        # Optimized for complex scripts and tonal languages
        "temperature": 0.65,           # Lower for tonal precision
        "repetition_penalty": 3.2,     # Higher for script complexity
        "top_k": 25,                  # More focused sampling
        "top_p": 0.75,
        "language_conditioning_strength": 1.8,  # Strong conditioning
        
        # Enhanced text processing for Asian scripts
        "max_text_length": 400,       # Longer for complex characters
        "text_cleaners": ["multilingual_cleaners"],
        
        # Training parameters optimized for complexity
        "batch_size": 4,
        "epochs": 1800,               # More epochs for complex languages
        "lr": 2e-5,                   # Lower learning rate
        
        # Enhanced features particularly good for Asian languages
        "use_phase1": True,
        "use_phase2": True,
        "use_mamba": True,            # Excellent for sequence modeling
        "use_neural_codec": True,     # Enhanced audio quality for tonal languages
        "use_rope": True,             # Better positional encoding
        
        # Paths
        "train_csv": "./data/demo/asian_train.csv",
        "eval_csv": "./data/demo/asian_eval.csv",
        "output_path": "./models/demo/enhanced_asian"
    }
    
    # Save configuration
    config_path = "./config/demo_asian_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create sample data for Chinese
    create_sample_data_csv("zh-cn", "./data/demo/asian_train.csv", 150)
    create_sample_data_csv("zh-cn", "./data/demo/asian_eval.csv", 30)
    
    print(f"📁 Configuration saved: {config_path}")
    print("🚀 To train: python train_multilingual_enhanced_xtts.py --train_csv data/demo/asian_train.csv --eval_csv data/demo/asian_eval.csv --language zh-cn")
    
    return config

def demo_fine_tuning_config():
    """Demonstrate fine-tuning configuration."""
    
    print("\n🎯 DEMO 4: Fine-tuning Configuration (English → Spanish)")
    print("=" * 60)
    
    config = {
        "run_name": "enhanced_xtts_finetune_demo",
        "model": "enhanced_xtts",
        "run_description": "Fine-tuning from English to Spanish demo",
        
        # Fine-tuning language setup
        "languages": ["en", "es"],
        "default_language": "en",      # Base language
        "target_language": "es",       # Target language
        
        # Fine-tuning optimized parameters
        "temperature": 0.75,
        "repetition_penalty": 2.5,
        "language_conditioning_strength": 1.3,
        "cross_lingual_conditioning": True,
        "multilingual_speaker_adaptation": True,
        
        # Fine-tuning training parameters
        "batch_size": 2,              # Smaller batch for fine-tuning
        "epochs": 500,                # Fewer epochs
        "lr": 1e-5,                   # Lower learning rate
        "grad_acumm": 2,              # Gradient accumulation
        
        # Enhanced features
        "use_phase1": True,
        "use_phase2": True,
        "use_flash_attention": True,
        
        # Fine-tuning paths
        "train_csv": "./data/demo/spanish_finetune_train.csv",
        "eval_csv": "./data/demo/spanish_finetune_eval.csv",
        "output_path": "./models/demo/enhanced_spanish_finetune",
        "continue_path": "./models/demo/enhanced_english/best_model.pth"  # Base model
    }
    
    # Save configuration
    config_path = "./config/demo_finetune_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create sample Spanish data for fine-tuning
    create_sample_data_csv("es", "./data/demo/spanish_finetune_train.csv", 100)
    create_sample_data_csv("es", "./data/demo/spanish_finetune_eval.csv", 25)
    
    print(f"📁 Configuration saved: {config_path}")
    print("🚀 To train: python train_multilingual_enhanced_xtts.py --train_csv data/demo/spanish_finetune_train.csv --eval_csv data/demo/spanish_finetune_eval.csv --language es --checkpoint models/demo/enhanced_english/best_model.pth")
    
    return config

def show_training_commands():
    """Show practical training commands for each demo."""
    
    print("\n🚀 TRAINING COMMANDS FOR EACH DEMO")
    print("=" * 60)
    
    commands = [
        {
            "name": "Single Language (English)",
            "command": "python train_full_enhanced_xtts.py --config config/demo_english_config.json"
        },
        {
            "name": "Multilingual (5 Languages)",
            "command": "python train_multilingual_enhanced_xtts.py --train_csv data/demo/multilingual_train.csv --eval_csv data/demo/multilingual_eval.csv --language en"
        },
        {
            "name": "Asian Languages",
            "command": "python train_multilingual_enhanced_xtts.py --train_csv data/demo/asian_train.csv --eval_csv data/demo/asian_eval.csv --language zh-cn"
        },
        {
            "name": "Fine-tuning (EN→ES)",
            "command": "python train_multilingual_enhanced_xtts.py --train_csv data/demo/spanish_finetune_train.csv --eval_csv data/demo/spanish_finetune_eval.csv --language es --checkpoint models/demo/enhanced_english/best_model.pth"
        }
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. {cmd['name']}:")
        print(f"   {cmd['command']}")

def main():
    """Run all configuration demos."""
    
    print("🌍 Enhanced XTTS Language Configuration Demonstrations")
    print("=" * 80)
    
    # Create demo configurations
    config1 = demo_single_language_config()
    config2 = demo_multilingual_config()
    config3 = demo_asian_languages_config()
    config4 = demo_fine_tuning_config()
    
    # Show training commands
    show_training_commands()
    
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 60)
    print("✅ 4 demo configurations created")
    print("✅ Sample training data generated")
    print("✅ All configurations saved to ./config/")
    print("✅ Training commands provided above")
    
    print("\n🎯 KEY LANGUAGE PARAMETERS EXPLAINED:")
    print("• languages: List of language codes to support")
    print("• default_language: Primary training language")
    print("• target_language: Language to optimize for (fine-tuning)")
    print("• language_conditioning_strength: How strongly to enforce language consistency (1.0-2.0)")
    print("• temperature: Generation creativity (0.6-0.8, lower = more consistent)")
    print("• repetition_penalty: Prevents language mixing (2.0-3.5, higher = less mixing)")
    print("• use_language_weighted_sampler: Balance training across languages")
    print("• cross_lingual_conditioning: Enable transfer learning between languages")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Choose a demo configuration that matches your use case")
    print("2. Modify the CSV data paths to point to your actual data")
    print("3. Adjust language-specific parameters if needed")
    print("4. Run the training command")
    print("5. Monitor language-specific metrics during training")
    
    print("\n📚 For complete documentation, see:")
    print("• LANGUAGE_CONFIGURATION_GUIDE.md")
    print("• train_multilingual_enhanced_xtts.py")
    print("• XTTS_ENHANCEMENT_GUIDE.md")

if __name__ == "__main__":
    main()
