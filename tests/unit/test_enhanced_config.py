#!/usr/bin/env python3
"""
Test script to validate the enhanced XTTS configuration
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_configuration():
    """Test the enhanced configuration setup"""
    print("🧪 Testing Enhanced XTTS Configuration...")
    
    try:
        # Test imports
        from trainer.xtts.enhanced_configs import (
            EnhancedXTTSConfig,
            EnhancedAudioConfig,
            EnhancedGPTArgs
        )
        print("✅ Successfully imported configuration classes")
        
        # Test configuration creation
        config = EnhancedXTTSConfig()
        audio_config = EnhancedAudioConfig()
        gpt_args = EnhancedGPTArgs()
        
        print("✅ Successfully created configuration objects")
        
        # Test configuration assignment
        config.audio = audio_config
        config.model_args = gpt_args
        
        print("✅ Successfully assigned configurations")
        
        # Test configuration properties
        print(f"🎵 Audio sample rate: {config.audio.sample_rate}")
        print(f"🎵 Audio hop length: {config.audio.hop_length}")
        print(f"🤖 Model args type: {type(config.model_args)}")
        
        if hasattr(config.model_args, 'max_wav_length'):
            print(f"📏 Max audio length: {config.model_args.max_wav_length}")
        
        if hasattr(config, 'temperature'):
            print(f"🌡️ Temperature: {config.temperature}")
        
        # Test argument parsing simulation
        from examples.train_xtts_enhanced import parse_args
        
        # Create fake arguments for testing
        test_args = [
            '--output_path', '/tmp/test_output',
            '--train_csv', '/tmp/test_train.csv',
            '--language', 'en',
            '--batch_size', '2',
            '--epochs', '1'
        ]
        
        # Override sys.argv temporarily
        original_argv = sys.argv
        sys.argv = ['test'] + test_args
        
        try:
            args = parse_args()
            print("✅ Successfully parsed arguments")
            print(f"📂 Output path: {args.output_path}")
            print(f"📊 Train CSV: {args.train_csv}")
            print(f"🌍 Language: {args.language}")
            print(f"📦 Batch size: {args.batch_size}")
        finally:
            sys.argv = original_argv
        
        print("\n🎉 All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_configuration()
    sys.exit(0 if success else 1)
