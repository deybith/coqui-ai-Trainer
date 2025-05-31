#!/usr/bin/env python3
"""
Advanced Features Demo - Testing New Components
This script demonstrates the new advanced voice cloning features.
"""

import os
import sys
import time
from pathlib import Path

def test_voice_quality_monitor():
    """Test the voice quality monitor system."""
    print("🎙️  Testing Voice Quality Monitor")
    print("-" * 40)
    
    try:
        # Import our quality monitor
        sys.path.append(os.getcwd())
        from voice_quality_monitor import VoiceQualityMonitor
        
        # Initialize monitor
        monitor = VoiceQualityMonitor(output_dir="test_quality_reports")
        
        # Test with sample data (create dummy files)
        test_dir = Path("test_audio")
        test_dir.mkdir(exist_ok=True)
        
        # Create dummy audio files for testing
        import torch
        import torchaudio
        
        # Create reference audio
        ref_audio = test_dir / "reference.wav"
        sample_rate = 22050
        duration = 2.0
        waveform = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, int(sample_rate * duration)))
        waveform = waveform.unsqueeze(0)
        torchaudio.save(str(ref_audio), waveform, sample_rate)
        
        # Create generated audio
        gen_audio = test_dir / "generated.wav"
        waveform2 = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration*1.1, int(sample_rate * duration * 1.1)))
        waveform2 = waveform2.unsqueeze(0) * 0.8  # Slightly different amplitude
        torchaudio.save(str(gen_audio), waveform2, sample_rate)
        
        # Test monitoring
        result = monitor.monitor_generation(
            reference_audio=str(ref_audio),
            output_audio=str(gen_audio),
            text="This is a test of the quality monitoring system.",
            generation_time=2.5
        )
        
        print(f"✅ Quality Monitor Test Passed!")
        print(f"   Quality Score: {result['generated_quality']:.1f}/100")
        print(f"   Consistency: {result['consistency_score']:.1f}/100")
        
        # Test trends
        trends = monitor.get_quality_trends()
        print(f"   Trends: {trends['total_generations']} generations monitored")
        
        # Generate report
        report_file = monitor.generate_quality_report()
        print(f"   Report saved: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice Quality Monitor test failed: {e}")
        return False

def test_batch_processor():
    """Test the batch voice processor system."""
    print("\n🚀 Testing Batch Voice Processor")
    print("-" * 40)
    
    try:
        from batch_voice_processor import BatchVoiceProcessor
        
        # Initialize processor
        processor = BatchVoiceProcessor(output_dir="test_batch_output")
        
        # Create sample configuration
        config_file = processor.create_sample_config("test_batch_config.json")
        print(f"✅ Sample config created: {config_file}")
        
        # Add test jobs
        test_texts = [
            "This is the first test job for batch processing.",
            "This is the second test job with different content.",
            "This is the third and final test job for validation."
        ]
        
        job_ids = processor.add_jobs_from_list(test_texts)
        print(f"✅ Added {len(job_ids)} test jobs")
        
        # Show queue status
        processor.show_queue_status()
        
        # Process batch (simulation mode)
        print("🔄 Processing batch (simulation mode)...")
        stats = processor.process_batch(max_workers=1)
        
        print(f"✅ Batch Processor Test Passed!")
        print(f"   Total jobs: {stats['total_jobs']}")
        print(f"   Completed: {stats['completed']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch Processor test failed: {e}")
        return False

def test_model_evaluator():
    """Test the model evaluation system."""
    print("\n🎯 Testing Model Evaluator")
    print("-" * 40)
    
    try:
        from model_evaluator import ModelEvaluator
        
        # Initialize evaluator
        evaluator = ModelEvaluator(output_dir="test_evaluation_results")
        
        # Create dummy reference audio
        test_dir = Path("test_audio")
        test_dir.mkdir(exist_ok=True)
        
        import torch
        import torchaudio
        
        ref_audio = test_dir / "eval_reference.wav"
        sample_rate = 22050
        duration = 3.0
        waveform = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, int(sample_rate * duration)))
        waveform = waveform.unsqueeze(0)
        torchaudio.save(str(ref_audio), waveform, sample_rate)
        
        # Add test scenarios
        evaluator.add_test_scenario(
            name="test_simple",
            text="This is a simple test sentence.",
            reference_audio=str(ref_audio),
            category="test"
        )
        
        evaluator.add_test_scenario(
            name="test_complex",
            text="This is a more complex test sentence with multiple clauses and technical terminology.",
            reference_audio=str(ref_audio),
            category="test"
        )
        
        # Run speed benchmark
        speed_results = evaluator.run_speed_benchmark(
            text_samples=["Short test.", "Medium length test sentence.", "Longer test sentence with more content."],
            iterations=3
        )
        
        print(f"✅ Speed benchmark completed")
        print(f"   Success rate: {speed_results['summary']['success_rate']:.1f}%")
        print(f"   Avg time: {speed_results['summary']['avg_generation_time']:.2f}s")
        
        # Run consistency test
        consistency_results = evaluator.run_consistency_test(
            text="Consistency test sentence.",
            reference_audio=str(ref_audio),
            iterations=5
        )
        
        print(f"✅ Consistency test completed")
        if 'avg_quality' in consistency_results['summary']:
            print(f"   Avg quality: {consistency_results['summary']['avg_quality']:.1f}")
            print(f"   Consistency score: {consistency_results['summary']['consistency_score']:.1f}/100")
        
        print(f"✅ Model Evaluator Test Passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Model Evaluator test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    import shutil
    
    test_dirs = [
        "test_audio",
        "test_quality_reports", 
        "test_batch_output",
        "test_evaluation_results"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    
    test_files = [
        "test_batch_config.json"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    """Run comprehensive test of all advanced features."""
    print("🚀 ADVANCED FEATURES TESTING SUITE")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test results tracking
    tests = [
        ("Voice Quality Monitor", test_voice_quality_monitor),
        ("Batch Voice Processor", test_batch_processor),
        ("Model Evaluator", test_model_evaluator)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n🎉 TESTING SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Total Time: {total_time:.1f} seconds")
    
    print(f"\n📋 Detailed Results:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        print("Advanced voice cloning features are working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
    
    # Clean up test files
    print(f"\n🧹 Cleaning up test files...")
    cleanup_test_files()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    main()
