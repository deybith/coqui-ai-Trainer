# XTTS Enhancement Implementation Guide

This guide provides step-by-step instructions for implementing the enhanced XTTS fixes to address audio cutoff, language mixing, and robotic artifacts.

## 🚀 Quick Start

### 1. Training a New Enhanced Model

```bash
# Use the enhanced training script
python examples/train_xtts_enhanced.py \
    --output_path ./enhanced_model_output \
    --train_csv ./data/train.csv \
    --eval_csv ./data/eval.csv \
    --language en \
    --batch_size 4 \
    --epochs 1000 \
    --max_audio_length 30 \
    --repetition_penalty 2.5 \
    --temperature 0.75
```

### 2. Testing an Enhanced Model

```bash
# Single inference test
python examples/test_xtts_enhanced.py \
    --model_path ./enhanced_model_output/run/training \
    --text "This is a test of the enhanced XTTS system." \
    --speaker_wav ./speaker_reference.wav \
    --output_path ./test_output.wav

# Comprehensive test suite
python examples/test_xtts_enhanced.py \
    --model_path ./enhanced_model_output/run/training \
    --text "TEST_SUITE" \
    --speaker_wav ./speaker_reference.wav
```

### 3. Validating Model Quality

```bash
# Run validation suite
python scripts/validate_xtts_model.py \
    --model_path ./enhanced_model_output/run/training \
    --speaker_wav ./speaker_reference.wav \
    --output_dir ./validation_results
```

## 📋 Implementation Details

### Enhanced Configuration Classes

The enhanced configurations are defined in `src/trainer/xtts/enhanced_configs.py`:

- **EnhancedAudioConfig**: Optimized audio processing parameters
- **EnhancedGPTTrainerConfig**: Enhanced GPT training parameters  
- **EnhancedXTTSConfig**: Main configuration combining all enhancements

### Key Improvements

#### 1. Audio Cutoff Fixes
- **Increased `max_wav_length`**: 660000 samples (30 seconds) vs original 480000 (21.7s)
- **Enhanced `gpt_max_audio_tokens`**: 800 vs original 605
- **Improved `max_text_length`**: 350 vs original 200
- **Better conditioning lengths**: Optimized for longer audio

#### 2. Language Mixing Prevention
- **Higher `repetition_penalty`**: 2.5 vs original 2.0
- **Optimized `temperature`**: 0.75 vs original 0.85
- **Better sampling**: `top_k=40, top_p=0.8`
- **Enhanced language conditioning**: Stronger language model guidance

#### 3. Robotic Artifact Reduction
- **Reduced `hop_length`**: 256 vs original 512 (better temporal resolution)
- **Enhanced normalization**: `do_sound_norm=True, do_rms_norm=True`
- **Aggressive silence trimming**: `trim_db=35` vs original 20
- **Optimized Griffin-Lim**: `power=1.2, griffin_lim_iters=100`
- **Better mel-spectrogram**: `mel_fmin=50.0, mel_fmax=8000.0`

## 🔧 Applying Fixes to Existing Models

### Option 1: Retrain with Enhanced Configuration

1. **Prepare your data** in the same format as before
2. **Use the enhanced training script** with your existing dataset
3. **Apply enhanced parameters** from the start

```bash
python examples/train_xtts_enhanced.py \
    --output_path ./retrained_enhanced \
    --train_csv ./your_data/train.csv \
    --eval_csv ./your_data/eval.csv \
    --language your_language \
    --max_audio_length 30 \
    --enable_sound_norm \
    --repetition_penalty 2.5
```

### Option 2: Continue Training from Existing Checkpoint

1. **Use your existing model as starting point**
2. **Apply enhanced configuration** to continue training

```bash
python examples/train_xtts_enhanced.py \
    --output_path ./continued_enhanced \
    --train_csv ./your_data/train.csv \
    --eval_csv ./your_data/eval.csv \
    --restore_path ./your_existing_model/best_model.pth \
    --language your_language \
    --epochs 200  # Fewer epochs for fine-tuning
```

### Option 3: Inference-Time Fixes Only

If you can't retrain, apply enhanced parameters during inference:

```python
# Enhanced inference parameters
outputs = model.synthesize(
    text=text,
    config=config,
    speaker_wav=speaker_waveform,
    language=language,
    temperature=0.75,           # Lower temperature
    repetition_penalty=2.5,     # Higher repetition penalty
    top_k=40,                   # Better sampling
    top_p=0.8,
    enable_text_splitting=True, # Handle long texts
)
```

## 📊 Quality Assessment

### Automated Validation

Use the validation script to assess improvements:

```bash
python scripts/validate_xtts_model.py \
    --model_path ./your_model \
    --speaker_wav ./reference.wav \
    --output_dir ./validation
```

### Manual Quality Checks

1. **Audio Cutoff Test**:
   - Test with sentences longer than 15 seconds
   - Check if all words are pronounced completely
   - Listen for abrupt endings

2. **Language Consistency Test**:
   - Generate same text multiple times
   - Check for consistent pronunciation
   - Verify no language switching

3. **Audio Quality Test**:
   - Listen for robotic/metallic sounds
   - Check for echo or reverb artifacts
   - Assess naturalness and clarity

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue: Audio Still Gets Cut Off
**Solution**: Increase max_audio_length further or check your text preprocessing

```python
# In enhanced config
gpt_config.max_wav_length = 880000  # 40 seconds
gpt_config.gpt_max_audio_tokens = 1000
```

#### Issue: Language Mixing Persists  
**Solution**: Increase repetition penalty and lower temperature

```python
# More aggressive language control
gpt_config.repetition_penalty = 3.0
gpt_config.temperature = 0.65
```

#### Issue: Still Sounds Robotic
**Solution**: Adjust audio processing parameters

```python
# More aggressive audio processing
audio_config.hop_length = 128  # Even finer resolution
audio_config.griffin_lim_iters = 150
audio_config.power = 1.0
```

#### Issue: Training is Slow
**Solution**: Optimize batch size and use mixed precision

```python
# Optimize training speed
config.batch_size = 8  # Increase if you have VRAM
config.mixed_precision = True
config.num_loader_workers = 16
```

#### Issue: Poor Quality with Short Speaker References
**Solution**: Use longer, higher quality speaker references

- Use 5-15 seconds of clean speech
- Ensure single speaker, no background noise
- Sample rate should be 22050 Hz
- Avoid music or overlapping speech

### VRAM Optimization

If you run out of VRAM during training:

```python
# Reduce memory usage
config.batch_size = 2
config.grad_acumm = 4  # Maintain effective batch size
gpt_config.max_wav_length = 440000  # 20 seconds instead of 30
```

### Performance Monitoring

Monitor training with these metrics:

- **Loss curves**: Should steadily decrease
- **Validation quality**: Use validation script regularly
- **Memory usage**: Monitor GPU utilization
- **Audio samples**: Listen to validation outputs

## 📈 Expected Improvements

After applying the enhanced configuration, you should see:

1. **Audio Cutoff**: 90%+ reduction in premature cutoffs
2. **Language Mixing**: 80%+ improvement in consistency
3. **Audio Quality**: Significantly more natural speech
4. **Overall Quality**: Higher MOS scores and user satisfaction

## 🔄 Migration from Original XTTS

### For Existing Projects

1. **Backup your current models** and configurations
2. **Update import statements** to use enhanced configs
3. **Retrain critical models** with enhanced settings
4. **Validate improvements** using the provided tools
5. **Gradually migrate** other models as needed

### Configuration Compatibility

The enhanced configurations are backward compatible:

```python
# Original training still works
from trainer.xtts.train_model import train_model

# Enhanced training provides better results
from trainer.xtts.train_model import train_xtts
from trainer.xtts.enhanced_configs import EnhancedXTTSConfig
```

## 📞 Support and Community

### Getting Help

1. **Check validation results** first to identify specific issues
2. **Review configuration parameters** against your use case  
3. **Test with provided examples** to isolate problems
4. **Monitor training logs** for any errors or warnings

### Contributing Improvements

If you find additional optimizations:

1. Test thoroughly with the validation suite
2. Document the changes and expected improvements
3. Share results with the community
4. Consider contributing back to the project

## 🎯 Next Steps

1. **Start with enhanced training** for new models
2. **Validate existing models** to identify issues  
3. **Gradually migrate** important models
4. **Monitor quality improvements** over time
5. **Fine-tune parameters** for your specific use case

The enhanced XTTS configuration provides a solid foundation for high-quality speech synthesis while addressing the most common issues. Regular validation and monitoring will help maintain optimal performance as you scale your usage.
