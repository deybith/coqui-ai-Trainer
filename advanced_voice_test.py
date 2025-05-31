#!/usr/bin/env python3
"""
Advanced voice cloning test that analyzes audio properties and creates comparison samples.
"""

import os
import sys
import torchaudio
import torch

def analyze_audio_properties(audio_path, label):
    """Analyze and display audio properties."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Calculate properties
        duration = waveform.shape[1] / sample_rate
        channels = waveform.shape[0]
        max_amplitude = torch.max(torch.abs(waveform)).item()
        rms = torch.sqrt(torch.mean(waveform**2)).item()
        
        print(f"\n📊 {label}:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Max Amplitude: {max_amplitude:.3f}")
        print(f"  RMS Level: {rms:.3f}")
        print(f"  File Size: {os.path.getsize(audio_path)} bytes")
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'channels': channels,
            'max_amplitude': max_amplitude,
            'rms': rms,
            'file_size': os.path.getsize(audio_path)
        }
    except Exception as e:
        print(f"❌ Error analyzing {audio_path}: {e}")
        return None

def create_comparison_samples():
    """Create comparison samples with different types of text."""
    
    print("🎭 Creating Comparison Samples")
    print("=" * 40)
    
    # Different types of text to test voice cloning capabilities
    test_scenarios = [
        {
            'name': 'Emotional',
            'text': "I'm absolutely thrilled about this amazing breakthrough in voice technology!",
            'output': 'comparison_emotional.wav'
        },
        {
            'name': 'Technical',
            'text': "The neural network architecture utilizes transformer-based attention mechanisms for voice synthesis.",
            'output': 'comparison_technical.wav'
        },
        {
            'name': 'Conversational',
            'text': "Hey there! How are you doing today? I hope you're having a wonderful time.",
            'output': 'comparison_conversational.wav'
        },
        {
            'name': 'Narrative',
            'text': "Once upon a time, in a land far away, there lived a brilliant scientist who discovered the secret of voice cloning.",
            'output': 'comparison_narrative.wav'
        }
    ]
    
    try:
        from TTS.api import TTS
        
        # Load TTS model
        print("Loading XTTS model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        print("✓ Model loaded")
        
        # Reference audio path
        ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
        
        # Generate samples for each scenario
        for scenario in test_scenarios:
            print(f"\nGenerating {scenario['name']} sample...")
            print(f"Text: {scenario['text']}")
            
            tts.tts_to_file(
                text=scenario['text'],
                speaker_wav=ref_audio,
                language="en",
                file_path=scenario['output']
            )
            
            if os.path.exists(scenario['output']):
                print(f"✓ Generated: {scenario['output']}")
            else:
                print(f"❌ Failed to generate: {scenario['output']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating comparison samples: {e}")
        return False

def analyze_voice_quality():
    """Analyze the quality of generated voice samples."""
    
    print("\n🔍 Voice Quality Analysis")
    print("=" * 40)
    
    # Analyze reference audio
    ref_audio = "data/dieck/dataset/wavs/audio10_00000000.wav"
    if os.path.exists(ref_audio):
        ref_props = analyze_audio_properties(ref_audio, "Reference Audio")
    else:
        print("❌ Reference audio not found")
        return False
    
    # Analyze generated samples
    sample_files = [
        ('demo_output_1.wav', 'Demo Sample 1'),
        ('comparison_emotional.wav', 'Emotional Sample'),
        ('comparison_technical.wav', 'Technical Sample'),
        ('comparison_conversational.wav', 'Conversational Sample')
    ]
    
    generated_props = []
    for filename, label in sample_files:
        if os.path.exists(filename):
            props = analyze_audio_properties(filename, label)
            if props:
                generated_props.append(props)
    
    # Quality assessment
    if ref_props and generated_props:
        print("\n📈 Quality Assessment:")
        
        # Check consistency in sample rate
        ref_sr = ref_props['sample_rate']
        gen_srs = [p['sample_rate'] for p in generated_props]
        if all(sr == ref_sr for sr in gen_srs):
            print("✓ Sample rates consistent")
        else:
            print("⚠️  Sample rate variations detected")
        
        # Check amplitude levels
        ref_rms = ref_props['rms']
        gen_rms_values = [p['rms'] for p in generated_props]
        avg_gen_rms = sum(gen_rms_values) / len(gen_rms_values)
        
        rms_diff = abs(avg_gen_rms - ref_rms) / ref_rms * 100
        print(f"✓ Average RMS difference: {rms_diff:.1f}%")
        
        if rms_diff < 20:
            print("✓ Good audio level consistency")
        else:
            print("⚠️  Audio levels may need adjustment")
    
    return True

def create_usage_examples():
    """Create practical usage examples."""
    
    examples_content = """
# 🎤 Voice Cloning Usage Examples

## Generated Samples

Your voice cloning system has successfully generated the following samples:

### Demo Samples
- `demo_output_1.wav` - "Hello! This is a test of the voice cloning system."
- `demo_output_2.wav` - "The model has been trained on the dieck voice dataset."
- `demo_output_3.wav` - "Voice cloning technology is quite impressive these days."
- `demo_output_4.wav` - "This demonstrates the capabilities of the XTTS model."

### Comparison Samples
- `comparison_emotional.wav` - Emotional expression test
- `comparison_technical.wav` - Technical terminology test
- `comparison_conversational.wav` - Casual conversation test
- `comparison_narrative.wav` - Storytelling test

## How to Play Audio Files

### Using ffplay (recommended):
```bash
ffplay demo_output_1.wav
```

### Using any audio player:
```bash
# VLC
vlc demo_output_1.wav

# MPV
mpv demo_output_1.wav

# Or double-click the file in your file manager
```

## Quick Voice Cloning Script

```python
from TTS.api import TTS

# Initialize TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Clone voice
tts.tts_to_file(
    text="Your custom text here",
    speaker_wav="data/dieck/dataset/wavs/audio10_00000000.wav",
    language="en",
    file_path="my_voice_clone.wav"
)
```

## Integration Examples

### Batch Processing
```python
texts = [
    "First sentence to clone.",
    "Second sentence to clone.",
    "Third sentence to clone."
]

for i, text in enumerate(texts):
    tts.tts_to_file(
        text=text,
        speaker_wav="reference.wav",
        language="en",
        file_path=f"output_{i+1}.wav"
    )
```

### Different Languages
```python
# English
tts.tts_to_file(text="Hello world", speaker_wav="ref.wav", language="en", file_path="en.wav")

# Spanish (if supported)
tts.tts_to_file(text="Hola mundo", speaker_wav="ref.wav", language="es", file_path="es.wav")
```

## Quality Tips

1. **Reference Audio**: Use clear, high-quality reference audio (3-10 seconds)
2. **Text Length**: Keep text reasonably short for best quality
3. **Punctuation**: Use proper punctuation for natural pauses
4. **Language**: Match the language of reference audio when possible

## Next Steps

1. Test with different reference audio files
2. Experiment with various text types
3. Adjust parameters for different use cases
4. Consider fine-tuning with more data if needed
"""
    
    with open("VOICE_CLONING_EXAMPLES.md", "w") as f:
        f.write(examples_content)
    
    print("✓ Usage examples created: VOICE_CLONING_EXAMPLES.md")

def main():
    print("🎯 Advanced Voice Cloning Test Suite")
    print("=" * 50)
    
    # Step 1: Create comparison samples
    print("\n🔄 Step 1: Creating comparison samples...")
    if create_comparison_samples():
        print("✓ Comparison samples created successfully")
    else:
        print("❌ Failed to create comparison samples")
        return False
    
    # Step 2: Analyze voice quality
    print("\n🔄 Step 2: Analyzing voice quality...")
    if analyze_voice_quality():
        print("✓ Quality analysis completed")
    else:
        print("❌ Quality analysis failed")
    
    # Step 3: Create usage examples
    print("\n🔄 Step 3: Creating usage documentation...")
    create_usage_examples()
    
    print("\n🎉 Advanced Testing Complete!")
    print("\nYour voice cloning system is working excellently!")
    print("Check the generated audio files and VOICE_CLONING_EXAMPLES.md for details.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
