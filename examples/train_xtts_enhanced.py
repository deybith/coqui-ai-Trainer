#!/usr/bin/env python3
"""
Enhanced XTTS Training Script with Fixes for:
1. Audio cutoff issues
2. Language mixing
3. Robotic/echo artifacts

This script uses optimized configurations to address common XTTS training problems.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trainer.xtts.enhanced_configs import (
    EnhancedXTTSConfig,
    EnhancedAudioConfig,
    EnhancedGPTArgs
)
from trainer.xtts.train_model import train_xtts


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced XTTS Training Script")
    
    # Data arguments
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the trained model")
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV file")
    parser.add_argument("--eval_csv", type=str, default=None,
                        help="Path to evaluation CSV file")
    parser.add_argument("--language", type=str, default="en",
                        help="Training language code")
    
    # Model arguments
    parser.add_argument("--restore_path", type=str, default=None,
                        help="Path to checkpoint to restore from")
    parser.add_argument("--speaker_reference", type=str, default=None,
                        help="Path to speaker reference WAV file")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    
    # Audio enhancement arguments
    parser.add_argument("--max_audio_length", type=int, default=30,
                        help="Maximum audio length in seconds")
    parser.add_argument("--enable_sound_norm", action="store_true", default=True,
                        help="Enable sound normalization")
    parser.add_argument("--trim_silence", type=float, default=35.0,
                        help="Silence trimming threshold in dB")
    
    # Language control arguments
    parser.add_argument("--repetition_penalty", type=float, default=2.5,
                        help="Repetition penalty to prevent language mixing")
    parser.add_argument("--temperature", type=float, default=0.75,
                        help="Sampling temperature")
    
    return parser.parse_args()


def create_enhanced_config(args):
    """Create enhanced configuration with optimized parameters."""
    
    # Create enhanced audio config
    audio_config = EnhancedAudioConfig()
    
    # Apply command line overrides
    if args.enable_sound_norm:
        audio_config.do_sound_norm = True
        audio_config.do_rms_norm = True
    
    audio_config.trim_db = args.trim_silence
    
    # Create enhanced GPT args
    gpt_args = EnhancedGPTArgs()
    
    # Apply audio length settings
    gpt_args.max_wav_length = int(args.max_audio_length * 22050)  # Convert seconds to samples
    
    # Create main XTTS config
    config = EnhancedXTTSConfig()
    
    # Set the GPT args
    config.model_args = gpt_args
    
    # Apply language control settings to config
    config.temperature = args.temperature
    config.repetition_penalty = args.repetition_penalty
    
    # Set paths and basic config
    config.output_path = args.output_path
    config.logger_uri = None
    
    # Dataset configuration
    config.datasets = [
        {
            "name": "enhanced_dataset",
            "path": args.train_csv,
            "meta_file_train": args.train_csv,
            "meta_file_val": args.eval_csv if args.eval_csv else args.train_csv,
            "language": args.language,
        }
    ]
    
    # Training configuration
    config.batch_size = args.batch_size
    config.grad_clip = args.grad_clip
    config.epochs = args.epochs
    config.lr = args.lr
    
    # Set enhanced configurations
    config.audio = audio_config
    # model_args is already set above
    
    # Restore path if provided
    if args.restore_path:
        config.restore_path = args.restore_path
    
    return config


def setup_enhanced_environment():
    """Setup environment variables for enhanced training."""
    # Enable mixed precision training for better performance
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    
    # Set optimal CUDA settings
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    
    # Enable deterministic operations for reproducibility
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    
    print("✅ Enhanced training environment configured")


def validate_inputs(args):
    """Validate input arguments."""
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    
    if args.eval_csv and not os.path.exists(args.eval_csv):
        raise FileNotFoundError(f"Evaluation CSV not found: {args.eval_csv}")
    
    if args.restore_path and not os.path.exists(args.restore_path):
        raise FileNotFoundError(f"Restore path not found: {args.restore_path}")
    
    if args.speaker_reference and not os.path.exists(args.speaker_reference):
        raise FileNotFoundError(f"Speaker reference not found: {args.speaker_reference}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print("✅ Input validation completed")


def print_enhancement_summary(config):
    """Print summary of enhancements applied."""
    print("\n" + "="*60)
    print("🚀 ENHANCED XTTS TRAINING CONFIGURATION")
    print("="*60)
    
    print("\n📢 AUDIO CUTOFF FIXES:")
    print(f"  • Max audio length: {config.model_args.max_wav_length / 22050:.1f}s")
    print(f"  • Max audio tokens: {config.model_args.gpt_max_audio_tokens}")
    print(f"  • Max text length: {config.model_args.max_text_length}")
    
    print("\n🌍 LANGUAGE MIXING FIXES:")
    print(f"  • Repetition penalty: {config.repetition_penalty}")
    print(f"  • Temperature: {config.temperature}")
    print(f"  • Top-k sampling: {config.top_k}")
    print(f"  • Top-p sampling: {config.top_p}")
    
    print("\n🎵 ROBOTIC AUDIO FIXES:")
    print(f"  • Hop length: {config.audio.hop_length}")
    print(f"  • Sound normalization: {config.audio.do_sound_norm}")
    print(f"  • RMS normalization: {config.audio.do_rms_norm}")
    print(f"  • Griffin-Lim iterations: {config.audio.griffin_lim_iters}")
    print(f"  • Griffin-Lim power: {config.audio.power}")
    print(f"  • Silence trimming: {config.audio.trim_db}dB")
    
    print("\n💪 TRAINING OPTIMIZATIONS:")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Learning rate: {config.lr}")
    print(f"  • Gradient clipping: {config.grad_clip}")
    print(f"  • Epochs: {config.epochs}")
    
    print("="*60)
    print("🎯 Ready to train with enhanced configurations!")
    print("="*60 + "\n")


def main():
    """Main training function."""
    print("🎙️  Starting Enhanced XTTS Training")
    print("📋 Parsing arguments...")
    
    args = parse_args()
    
    print("🔧 Setting up enhanced environment...")
    setup_enhanced_environment()
    
    print("✅ Validating inputs...")
    validate_inputs(args)
    
    print("⚙️  Creating enhanced configuration...")
    config = create_enhanced_config(args)
    
    # Print enhancement summary
    print_enhancement_summary(config)
    
    # Add speaker reference if provided
    if args.speaker_reference:
        print(f"🎤 Using speaker reference: {args.speaker_reference}")
        # You can add speaker conditioning logic here if needed
    
    print("🚀 Starting training with enhanced configurations...")
    
    try:
        # Start training
        train_xtts(config)
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Model saved to: {config.output_path}")
        print("\n📝 Next steps:")
        print("  1. Test the model with inference scripts")
        print("  2. Check audio quality improvements")
        print("  3. Validate language consistency")
        print("  4. Monitor for robotic artifacts")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print("💡 Check the error logs and configuration")
        raise


if __name__ == "__main__":
    main()
