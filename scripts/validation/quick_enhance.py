#!/usr/bin/env python3
"""
Quick XTTS Enhancement Setup

This script provides a simple interface to apply XTTS enhancements.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_enhanced_training():
    """Setup enhanced training environment."""
    print("🔧 Setting up Enhanced XTTS Training Environment")
    print("="*50)
    
    # Get current working directory as project root
    current_dir = Path.cwd()
    
    # Check if enhanced configs exist
    enhanced_config_path = current_dir / "trainer" / "xtts" / "enhanced_configs.py"
    if not enhanced_config_path.exists():
        print("❌ Enhanced configuration file not found!")
        print(f"Expected: {enhanced_config_path}")
        print(f"Current directory: {current_dir}")
        return False
    
    # Check if enhanced training script exists  
    enhanced_script_path = current_dir / "examples" / "train_xtts_enhanced.py"
    if not enhanced_script_path.exists():
        print("❌ Enhanced training script not found!")
        print(f"Expected: {enhanced_script_path}")
        return False
    
    # Check validation script
    validation_script_path = current_dir / "scripts" / "validate_xtts_model.py"
    if not validation_script_path.exists():
        print("❌ Validation script not found!")
        print(f"Expected: {validation_script_path}")
        return False
    
    # Check test script
    test_script_path = current_dir / "examples" / "test_xtts_enhanced.py"
    if not test_script_path.exists():
        print("❌ Test script not found!")
        print(f"Expected: {test_script_path}")
        return False
    
    print("✅ Enhanced configuration files found")
    print("✅ Enhanced training script found")
    print("✅ Enhanced test script found")
    print("✅ Validation script found")
    print("✅ Setup complete!")
    
    # Test import of enhanced configs
    try:
        import sys
        sys.path.insert(0, str(current_dir))
        from trainer.xtts.enhanced_configs import EnhancedXTTSConfig
        print("✅ Enhanced configuration classes can be imported")
        return True
    except ImportError as e:
        print(f"❌ Failed to import enhanced configs: {e}")
        return False

def check_data_format(csv_path):
    """Check if CSV data is in correct format."""
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    try:
        # Try to import pandas
        try:
            import pandas as pd
        except ImportError:
            print("⚠️  pandas not installed. Install with: pip install pandas")
            print("✅ CSV file exists, skipping format validation")
            return True
            
        df = pd.read_csv(csv_path)
        
        required_columns = ['audio_file', 'text', 'speaker_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns in {csv_path}: {missing_columns}")
            print(f"Required columns: {required_columns}")
            return False
        
        print(f"✅ CSV format is correct: {len(df)} samples")
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV: {str(e)}")
        return False

def generate_training_command(args):
    """Generate the enhanced training command."""
    cmd_parts = [
        "python",
        "examples/train_xtts_enhanced.py",
        f"--output_path {args.output_path}",
        f"--train_csv {args.train_csv}",
        f"--language {args.language}",
        f"--batch_size {args.batch_size}",
        f"--epochs {args.epochs}",
        f"--max_audio_length {args.max_audio_length}",
        f"--repetition_penalty {args.repetition_penalty}",
        f"--temperature {args.temperature}",
    ]
    
    if args.eval_csv:
        cmd_parts.append(f"--eval_csv {args.eval_csv}")
    
    if args.restore_path:
        cmd_parts.append(f"--restore_path {args.restore_path}")
    
    if args.speaker_reference:
        cmd_parts.append(f"--speaker_reference {args.speaker_reference}")
    
    cmd_parts.extend([
        "--enable_sound_norm",
        f"--trim_silence {args.trim_silence}",
    ])
    
    return " \\\n    ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(description="XTTS Enhancement Setup")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup enhanced training environment')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check data format')
    check_parser.add_argument('--csv', type=str, required=True, help='CSV file to check')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Generate enhanced training command')
    train_parser.add_argument('--output_path', type=str, required=True, help='Output path for trained model')
    train_parser.add_argument('--train_csv', type=str, required=True, help='Training CSV file')
    train_parser.add_argument('--eval_csv', type=str, help='Evaluation CSV file')
    train_parser.add_argument('--language', type=str, default='en', help='Language code')
    train_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    train_parser.add_argument('--max_audio_length', type=int, default=30, help='Max audio length in seconds')
    train_parser.add_argument('--repetition_penalty', type=float, default=2.5, help='Repetition penalty')
    train_parser.add_argument('--temperature', type=float, default=0.75, help='Sampling temperature')
    train_parser.add_argument('--trim_silence', type=float, default=35.0, help='Silence trimming threshold')
    train_parser.add_argument('--restore_path', type=str, help='Path to checkpoint to restore from')
    train_parser.add_argument('--speaker_reference', type=str, help='Speaker reference WAV file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Generate testing command')
    test_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    test_parser.add_argument('--speaker_wav', type=str, required=True, help='Speaker reference WAV')
    test_parser.add_argument('--text', type=str, default='TEST_SUITE', help='Text to synthesize or TEST_SUITE')
    test_parser.add_argument('--output_path', type=str, default='./test_output.wav', help='Output WAV file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Generate validation command')
    validate_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    validate_parser.add_argument('--speaker_wav', type=str, required=True, help='Speaker reference WAV')
    validate_parser.add_argument('--output_dir', type=str, default='./validation_results', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'setup':
        if setup_enhanced_training():
            print("\n🎉 Enhanced XTTS is ready to use!")
            print("\nNext steps:")
            print("  1. Prepare your training data in CSV format")
            print("  2. Run: python quick_enhance.py check --csv your_data.csv")
            print("  3. Run: python quick_enhance.py train --output_path ./model --train_csv ./data.csv")
        else:
            print("\n❌ Setup failed. Please check the error messages above.")
            
    elif args.command == 'check':
        if check_data_format(args.csv):
            print(f"\n✅ {args.csv} is ready for enhanced training!")
        else:
            print(f"\n❌ {args.csv} needs to be fixed before training.")
            
    elif args.command == 'train':
        print("🚀 Enhanced XTTS Training Command:")
        print("="*50)
        print(generate_training_command(args))
        print("\n💡 Copy and run the command above to start enhanced training!")
        
    elif args.command == 'test':
        print("🧪 Enhanced XTTS Testing Command:")
        print("="*50)
        cmd = f"""python examples/test_xtts_enhanced.py \\
    --model_path {args.model_path} \\
    --speaker_wav {args.speaker_wav} \\
    --text "{args.text}" \\
    --output_path {args.output_path}"""
        print(cmd)
        print("\n💡 Copy and run the command above to test your enhanced model!")
        
    elif args.command == 'validate':
        print("🔍 Enhanced XTTS Validation Command:")
        print("="*50)
        cmd = f"""python scripts/validate_xtts_model.py \\
    --model_path {args.model_path} \\
    --speaker_wav {args.speaker_wav} \\
    --output_dir {args.output_dir}"""
        print(cmd)
        print("\n💡 Copy and run the command above to validate your enhanced model!")

if __name__ == "__main__":
    main()
