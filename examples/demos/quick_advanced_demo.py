#!/usr/bin/env python3
"""
Quick Demo of Advanced Voice Features
"""

import os
import sys
import time
import torch
import torchaudio
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

def create_test_audio():
    """Create test audio files for demonstration."""
    test_dir = Path("demo_test_audio")
    test_dir.mkdir(exist_ok=True)
    
    sample_rate = 22050
    duration = 2.0
    
    # Create reference audio (sine wave at 440 Hz)
    ref_audio = test_dir / "reference.wav"
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform_ref = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    torchaudio.save(str(ref_audio), waveform_ref, sample_rate)
    
    # Create generated audio (slightly different frequency and amplitude)
    gen_audio = test_dir / "generated.wav"
    waveform_gen = torch.sin(2 * torch.pi * 442 * t).unsqueeze(0) * 0.9
    torchaudio.save(str(gen_audio), waveform_gen, sample_rate)
    
    return str(ref_audio), str(gen_audio)

def demo_quality_monitor():
    """Demonstrate voice quality monitoring."""
    print("🎙️  Voice Quality Monitor Demo")
    print("=" * 40)
    
    # Execute the VoiceQualityMonitor class definition
    exec(open('voice_quality_monitor.py').read(), globals())
    
    # Create test audio
    ref_audio, gen_audio = create_test_audio()
    
    # Initialize monitor
    monitor = VoiceQualityMonitor(output_dir="demo_quality_reports")
    
    # Monitor a generation
    result = monitor.monitor_generation(
        reference_audio=ref_audio,
        output_audio=gen_audio,
        text="This is a demonstration of the voice quality monitoring system.",
        generation_time=2.5
    )
    
    print(f"\n📊 Monitoring Results:")
    print(f"   Quality Score: {result['generated_quality']:.1f}/100")
    print(f"   Consistency Score: {result['consistency_score']:.1f}/100")
    print(f"   Overall Score: {result['overall_score']:.1f}/100")
    print(f"   Real-time Factor: {result['real_time_factor']:.2f}x")
    
    # Generate trends
    trends = monitor.get_quality_trends()
    print(f"\n📈 Quality Trends:")
    print(f"   Total Generations: {trends['total_generations']}")
    print(f"   Average Quality: {trends['average_quality']:.1f}/100")
    
    # Generate report
    report_file = monitor.generate_quality_report()
    print(f"\n📋 Report Generated: {report_file}")
    
    return True

def demo_batch_processor():
    """Demonstrate batch voice processing."""
    print("\n🚀 Batch Voice Processor Demo")
    print("=" * 40)
    
    # Execute the BatchVoiceProcessor class definition
    exec(open('batch_voice_processor.py').read(), globals())
    
    # Initialize processor
    processor = BatchVoiceProcessor(output_dir="demo_batch_output")
    
    # Add some test jobs
    test_texts = [
        "This is the first batch processing test.",
        "Here's another sentence for batch processing.",
        "And this is the final test sentence."
    ]
    
    for i, text in enumerate(test_texts):
        processor.add_job(
            text=text,
            output_name=f"demo_batch_{i+1}.wav"
        )
    
    print(f"✅ Added {len(test_texts)} jobs to queue")
    
    # Show queue status
    processor.show_queue_status()
    
    # Process batch
    print("\n🔄 Processing batch...")
    stats = processor.process_batch(max_workers=1)
    
    print(f"\n📊 Batch Processing Results:")
    print(f"   Total Jobs: {stats['total_jobs']}")
    print(f"   Completed: {stats['completed']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Processing Time: {stats['processing_time']:.1f}s")
    
    return True

def demo_model_evaluator():
    """Demonstrate model evaluation."""
    print("\n🎯 Model Evaluator Demo")
    print("=" * 40)
    
    # Execute the ModelEvaluator class definition
    exec(open('model_evaluator.py').read(), globals())
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir="demo_evaluation_results")
    
    # Create reference audio
    ref_audio, _ = create_test_audio()
    
    # Add test scenarios
    evaluator.add_test_scenario(
        name="demo_simple",
        text="This is a simple evaluation test.",
        reference_audio=ref_audio,
        category="demo"
    )
    
    evaluator.add_test_scenario(
        name="demo_complex",
        text="This is a more complex evaluation test with multiple technical terms and longer sentences.",
        reference_audio=ref_audio,
        category="demo"
    )
    
    print(f"✅ Added {len(evaluator.test_scenarios)} test scenarios")
    
    # Run speed benchmark
    print("\n🚀 Running speed benchmark...")
    speed_results = evaluator.run_speed_benchmark(
        text_samples=["Short test.", "Medium length test."],
        iterations=3
    )
    
    print(f"📊 Speed Benchmark Results:")
    if speed_results.get('summary'):
        summary = speed_results['summary']
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Avg Generation Time: {summary['avg_generation_time']:.2f}s")
        print(f"   Avg RT Factor: {summary['avg_rt_factor']:.2f}x")
    
    # Run consistency test
    print("\n🎯 Running consistency test...")
    consistency_results = evaluator.run_consistency_test(
        text="Consistency test sentence for evaluation.",
        reference_audio=ref_audio,
        iterations=5
    )
    
    print(f"📊 Consistency Test Results:")
    if consistency_results.get('summary'):
        summary = consistency_results['summary']
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        if 'avg_quality' in summary:
            print(f"   Avg Quality: {summary['avg_quality']:.1f} ± {summary['quality_std']:.1f}")
            print(f"   Consistency Score: {summary['consistency_score']:.1f}/100")
    
    return True

def cleanup_demo_files():
    """Clean up demo files."""
    import shutil
    
    demo_dirs = [
        "demo_test_audio",
        "demo_quality_reports",
        "demo_batch_output", 
        "demo_evaluation_results"
    ]
    
    for demo_dir in demo_dirs:
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)
            print(f"🗑️  Cleaned up {demo_dir}")

def main():
    """Run the advanced features demo."""
    print("🚀 ADVANCED VOICE CLONING FEATURES DEMO")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run demos
        demo_quality_monitor()
        demo_batch_processor()
        demo_model_evaluator()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY! 🎉")
        print(f"Total Demo Time: {total_time:.1f} seconds")
        
        print(f"\n📁 Generated Output Directories:")
        print("   - demo_quality_reports/ (Quality monitoring reports)")
        print("   - demo_batch_output/ (Batch processing results)")
        print("   - demo_evaluation_results/ (Model evaluation results)")
        
        # Ask if user wants to clean up
        print(f"\n🧹 Demo files generated. Clean up? (y/n): ", end="")
        # For automated demo, clean up after a pause
        time.sleep(2)
        print("y")
        cleanup_demo_files()
        print("✅ Demo cleanup completed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
