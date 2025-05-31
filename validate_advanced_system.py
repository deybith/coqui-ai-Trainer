#!/usr/bin/env python3
"""
Quick validation test for advanced voice cloning features.
"""

import os
import sys

def test_imports():
    """Test if all required imports work."""
    
    print("🔄 Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import torchaudio
        print("✓ TorchAudio available")
    except ImportError:
        print("❌ TorchAudio not available")
        return False
    
    try:
        from TTS.api import TTS
        print("✓ TTS API available")
    except ImportError:
        print("❌ TTS API not available")
        return False
    
    return True

def test_reference_audio():
    """Test if reference audio exists."""
    
    print("\n🔄 Testing reference audio...")
    
    ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
    if os.path.exists(ref_audio):
        size = os.path.getsize(ref_audio)
        print(f"✓ Reference audio found: {ref_audio} ({size:,} bytes)")
        return True
    else:
        print(f"❌ Reference audio not found: {ref_audio}")
        return False

def test_existing_demos():
    """Test existing demo files."""
    
    print("\n🔄 Testing existing demo files...")
    
    demo_files = ['demo_output_1.wav', 'demo_output_2.wav', 'demo_output_3.wav', 'demo_output_4.wav']
    found_files = []
    
    for file in demo_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size:,} bytes)")
            found_files.append(file)
        else:
            print(f"❌ {file} not found")
    
    return len(found_files) > 0

def quick_tts_test():
    """Quick TTS functionality test."""
    
    print("\n🔄 Quick TTS test...")
    
    try:
        from TTS.api import TTS
        
        # Load model
        print("  Loading TTS model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        print("  ✓ Model loaded successfully")
        
        # Test generation
        ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
        if not os.path.exists(ref_audio):
            print(f"  ❌ Reference audio not found: {ref_audio}")
            return False
        
        output_file = "validation_test.wav"
        test_text = "This is a quick validation test of the voice cloning system."
        
        print("  Generating test sample...")
        tts.tts_to_file(
            text=test_text,
            speaker_wav=ref_audio,
            language="en",
            file_path=output_file
        )
        
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"  ✓ Test sample generated: {output_file} ({size:,} bytes)")
            return True
        else:
            print("  ❌ Failed to generate test sample")
            return False
            
    except Exception as e:
        print(f"  ❌ TTS test failed: {e}")
        return False

def main():
    """Run validation tests."""
    
    print("🧪 Advanced Voice Cloning Validation Test")
    print("=" * 45)
    
    # Run tests
    tests = [
        ("Import validation", test_imports),
        ("Reference audio check", test_reference_audio), 
        ("Existing demos check", test_existing_demos),
        ("TTS functionality test", quick_tts_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    print(f"\n📊 Validation Summary")
    print("=" * 25)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All validation tests passed! Your advanced system is ready.")
        
        print(f"\n🚀 Next Steps:")
        print("1. Run enhanced_voice_cloning_demo.py for advanced features")
        print("2. Use voice_quality_monitor.py for quality analysis")
        print("3. Try batch_voice_processor.py for bulk processing")
        print("4. Use model_evaluator.py for comprehensive evaluation")
        
    elif passed >= total * 0.75:
        print("🟡 Most tests passed. System is mostly functional.")
    else:
        print("⚠️ Multiple test failures. Please check your setup.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
