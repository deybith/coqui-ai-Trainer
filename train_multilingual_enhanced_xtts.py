#!/usr/bin/env python3
"""
Enhanced XTTS Multilingual Training Script
Demonstrates how to configure language settings for Enhanced XTTS training.
"""

import os
import sys
import torch
from pathlib import Path

# Add the trainer path
sys.path.append(str(Path(__file__).parent))

from trainer.xtts.enhanced_configs import EnhancedXTTSConfig, EnhancedGPTTrainerConfig, EnhancedAudioConfig
from trainer.xtts.train_model import train_xtts

def create_multilingual_config():
    """Create a comprehensive multilingual training configuration."""
    
    # Audio configuration with enhanced settings
    audio_config = EnhancedAudioConfig()
    
    # Enhanced GPT trainer configuration
    trainer_config = EnhancedGPTTrainerConfig(
        # Core training parameters
        batch_size=4,
        eval_batch_size=2,
        learning_rate=5e-5,
        num_epochs=1000,
        
        # Language-specific enhancements
        use_language_conditioning=True,
        language_conditioning_strength=1.5,
        
        # Phase 2 enhancements
        use_phase1=True,
        use_phase2=True,
        use_mamba=True,
        use_moe=True,
        use_flash_attention=True,
        use_rope=True,
        use_neural_codec=True,
        use_streaming=True,
        use_quality_monitoring=True
    )
    
    # Main Enhanced XTTS configuration
    config = EnhancedXTTSConfig(
        # Model identification
        run_name="enhanced_xtts_multilingual",
        model="enhanced_xtts",
        run_description="Enhanced XTTS with multilingual language conditioning",
        
        # Language configuration - ALL 17 supported languages
        languages=[
            "en",     # English
            "es",     # Spanish
            "fr",     # French
            "de",     # German
            "it",     # Italian
            "pt",     # Portuguese
            "pl",     # Polish
            "tr",     # Turkish
            "ru",     # Russian
            "nl",     # Dutch
            "cs",     # Czech
            "ar",     # Arabic
            "zh-cn",  # Chinese (Simplified)
            "ja",     # Japanese
            "hu",     # Hungarian
            "ko",     # Korean
            "hi"      # Hindi
        ],
        default_language="en",
        
        # Language-aware training
        use_language_weighted_sampler=True,
        language_weighted_sampler_alpha=1.0,
        
        # Enhanced text processing
        max_text_length=350,
        text_cleaners=["multilingual_cleaners"],
        
        # Enhanced audio processing
        max_wav_length=660000,  # 30 seconds
        
        # Enhanced generation parameters for better language consistency
        temperature=0.75,           # Balanced creativity vs consistency
        repetition_penalty=2.5,     # Strong language mixing prevention
        length_penalty=1.0,
        top_k=40,                   # Focused sampling
        top_p=0.8,                  # Quality-focused
        
        # Training configuration
        epochs=1000,
        batch_size=4,
        eval_batch_size=2,
        lr=5e-5,
        grad_acumm=1,
        
        # Enhanced configs
        audio=audio_config,
        trainer_config=trainer_config,
        
        # File paths
        output_path="./enhanced_multilingual_output",
    )
    
    return config

def train_multilingual_model(
    train_csv_path: str,
    eval_csv_path: str,
    target_language: str = "en",
    speaker_reference_path: str = None,
    fine_tune_checkpoint: str = None
):
    """
    Train Enhanced XTTS model with multilingual support.
    
    Args:
        train_csv_path: Path to training data CSV
        eval_csv_path: Path to evaluation data CSV  
        target_language: Primary target language for training
        speaker_reference_path: Path to speaker reference audio
        fine_tune_checkpoint: Path to checkpoint for fine-tuning
    """
    
    # Create multilingual configuration
    config = create_multilingual_config()
    
    # Set target language
    config.default_language = target_language
    
    # Set data paths
    config.train_csv = train_csv_path
    config.eval_csv = eval_csv_path
    
    # Configure speaker reference if provided
    if speaker_reference_path:
        config.speaker_reference = speaker_reference_path
    
    # Configure fine-tuning if checkpoint provided
    if fine_tune_checkpoint:
        config.continue_path = fine_tune_checkpoint
    
    print(f"🚀 Starting Enhanced XTTS Multilingual Training")
    print(f"📊 Target Language: {target_language}")
    print(f"🌍 Supported Languages: {len(config.languages)} languages")
    print(f"🎯 Languages: {', '.join(config.languages)}")
    print(f"📈 Enhanced Features: Phase 1 + Phase 2")
    print(f"🔧 Training Configuration:")
    print(f"   - Batch Size: {config.batch_size}")
    print(f"   - Learning Rate: {config.lr}")
    print(f"   - Epochs: {config.epochs}")
    print(f"   - Max Audio Length: {config.max_wav_length / 22050:.1f} seconds")
    print(f"   - Repetition Penalty: {config.repetition_penalty}")
    print(f"   - Temperature: {config.temperature}")
    
    # Start training
    try:
        result = train_xtts(config)
        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved to: {config.output_path}")
        return result
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

def configure_language_specific_training(target_language: str):
    """
    Configure language-specific parameters for optimal performance.
    
    Args:
        target_language: The target language code (e.g., 'en', 'es', 'zh-cn')
    """
    
    # Language-specific optimization parameters
    language_configs = {
        "en": {
            "temperature": 0.75,
            "repetition_penalty": 2.5,
            "top_k": 40,
            "top_p": 0.8,
            "language_conditioning_strength": 1.0
        },
        "es": {
            "temperature": 0.70,
            "repetition_penalty": 2.8,
            "top_k": 35,
            "top_p": 0.85,
            "language_conditioning_strength": 1.2
        },
        "fr": {
            "temperature": 0.70,
            "repetition_penalty": 2.8,
            "top_k": 35,
            "top_p": 0.85,
            "language_conditioning_strength": 1.2
        },
        "de": {
            "temperature": 0.68,
            "repetition_penalty": 3.0,
            "top_k": 30,
            "top_p": 0.8,
            "language_conditioning_strength": 1.3
        },
        "zh-cn": {
            "temperature": 0.65,
            "repetition_penalty": 3.2,
            "top_k": 25,
            "top_p": 0.75,
            "language_conditioning_strength": 1.5
        },
        "ja": {
            "temperature": 0.65,
            "repetition_penalty": 3.0,
            "top_k": 30,
            "top_p": 0.75,
            "language_conditioning_strength": 1.4
        },
        "ar": {
            "temperature": 0.70,
            "repetition_penalty": 2.8,
            "top_k": 35,
            "top_p": 0.8,
            "language_conditioning_strength": 1.3
        },
        "hi": {
            "temperature": 0.68,
            "repetition_penalty": 2.9,
            "top_k": 30,
            "top_p": 0.8,
            "language_conditioning_strength": 1.3
        }
    }
    
    # Default configuration for unlisted languages
    default_config = {
        "temperature": 0.75,
        "repetition_penalty": 2.5,
        "top_k": 40,
        "top_p": 0.8,
        "language_conditioning_strength": 1.0
    }
    
    return language_configs.get(target_language, default_config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced XTTS Multilingual Training")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--eval_csv", type=str, required=True, help="Path to evaluation CSV file")
    parser.add_argument("--language", type=str, default="en", help="Target language code")
    parser.add_argument("--speaker_reference", type=str, help="Path to speaker reference audio")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for fine-tuning")
    parser.add_argument("--output_path", type=str, default="./enhanced_multilingual_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Validate language
    supported_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
    if args.language not in supported_languages:
        print(f"❌ Unsupported language: {args.language}")
        print(f"✅ Supported languages: {', '.join(supported_languages)}")
        sys.exit(1)
    
    # Start training
    train_multilingual_model(
        train_csv_path=args.train_csv,
        eval_csv_path=args.eval_csv,
        target_language=args.language,
        speaker_reference_path=args.speaker_reference,
        fine_tune_checkpoint=args.checkpoint
    )
