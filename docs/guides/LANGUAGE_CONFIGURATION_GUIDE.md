# 🌍 Enhanced XTTS Language Configuration Guide

## Overview

Your Enhanced XTTS system supports **17 languages** with advanced multilingual capabilities. This guide shows you exactly how to configure language settings for optimal training results.

## 🎯 Supported Languages

```
✅ TIER 1 (Extensively Tested):
   • en  - English
   • es  - Spanish  
   • fr  - French
   • de  - German

✅ TIER 2 (Well Supported):
   • it  - Italian
   • pt  - Portuguese
   • ru  - Russian
   • nl  - Dutch
   • pl  - Polish

✅ TIER 3 (Specialized Support):
   • zh-cn - Chinese (Simplified)
   • ja     - Japanese
   • ko     - Korean
   • ar     - Arabic
   • hi     - Hindi
   • cs     - Czech
   • tr     - Turkish
   • hu     - Hungarian
```

## 🚀 Quick Start Examples

### 1. Single Language Training (English)

```bash
# Command line approach
python train_full_enhanced_xtts.py \
    --config config/enhanced_xtts_config.json \
    --language en \
    --train_csv ./data/train_en.csv \
    --eval_csv ./data/eval_en.csv \
    --output_path ./models/enhanced_english
```

### 2. Multilingual Training (All Languages)

```bash
# Full multilingual training
python train_multilingual_enhanced_xtts.py \
    --train_csv ./data/multilingual_train.csv \
    --eval_csv ./data/multilingual_eval.csv \
    --language en \
    --output_path ./models/enhanced_multilingual
```

### 3. Language-Specific Fine-tuning

```bash
# Fine-tune English model for Spanish
python train_multilingual_enhanced_xtts.py \
    --train_csv ./data/spanish_train.csv \
    --eval_csv ./data/spanish_eval.csv \
    --language es \
    --checkpoint ./models/enhanced_english/best_model.pth \
    --output_path ./models/enhanced_spanish
```

## 📋 Configuration Methods

### Method 1: JSON Configuration File

Create `config/my_language_config.json`:

```json
{
  "run_name": "enhanced_xtts_multilingual",
  "languages": ["en", "es", "fr", "de", "it"],
  "default_language": "en",
  "target_language": "en",
  
  "use_language_weighted_sampler": true,
  "language_weighted_sampler_alpha": 1.0,
  "language_conditioning_strength": 1.5,
  
  "temperature": 0.75,
  "repetition_penalty": 2.5,
  "top_k": 40,
  "top_p": 0.8,
  
  "batch_size": 4,
  "epochs": 1000,
  "lr": 5e-5,
  
  "use_phase1": true,
  "use_phase2": true,
  "use_mamba": true,
  "use_moe": true
}
```

### Method 2: Python Configuration

```python
from trainer.xtts.enhanced_configs import EnhancedXTTSConfig

config = EnhancedXTTSConfig(
    # Language settings
    languages=["en", "es", "fr"],
    default_language="en",
    target_language="es",
    
    # Language optimization
    language_conditioning_strength=1.2,
    use_language_weighted_sampler=True,
    
    # Generation parameters
    temperature=0.75,
    repetition_penalty=2.5,
    
    # Enhanced features
    use_phase1=True,
    use_phase2=True,
    use_mamba=True
)
```

## 🎛️ Language-Specific Parameter Optimization

### English (en) - Baseline
```python
{
    "temperature": 0.75,
    "repetition_penalty": 2.5,
    "top_k": 40,
    "top_p": 0.8,
    "language_conditioning_strength": 1.0
}
```

### Romance Languages (es, fr, it, pt)
```python
{
    "temperature": 0.70,        # Slightly lower for consistency
    "repetition_penalty": 2.8,  # Higher to prevent mixing
    "top_k": 35,
    "top_p": 0.85,
    "language_conditioning_strength": 1.2
}
```

### Germanic Languages (de, nl)
```python
{
    "temperature": 0.68,        # Lower for compound words
    "repetition_penalty": 3.0,  # Higher for complex morphology
    "top_k": 30,
    "top_p": 0.8,
    "language_conditioning_strength": 1.3
}
```

### East Asian Languages (zh-cn, ja, ko)
```python
{
    "temperature": 0.65,        # Lower for tonal precision
    "repetition_penalty": 3.2,  # Highest for script complexity
    "top_k": 25,               # More focused sampling
    "top_p": 0.75,
    "language_conditioning_strength": 1.5,
    "max_text_length": 400     # Longer for complex characters
}
```

### Arabic/RTL Languages (ar)
```python
{
    "temperature": 0.70,
    "repetition_penalty": 2.8,
    "top_k": 35,
    "top_p": 0.8,
    "language_conditioning_strength": 1.3,
    "enable_rtl_processing": true
}
```

## 🔧 Advanced Language Features

### 1. Language-Weighted Sampling
```python
# Automatically balance training across languages
config.use_language_weighted_sampler = True
config.language_weighted_sampler_alpha = 1.0  # 1.0 = balanced, >1.0 = more uniform
```

### 2. Cross-Lingual Conditioning
```python
# Enable transfer learning between languages
config.cross_lingual_conditioning = True
config.multilingual_speaker_adaptation = True
```

### 3. Code-Switching Support
```python
# Handle mixed-language sentences
config.enable_code_switching = True
config.code_switch_probability = 0.1  # 10% of training samples
```

### 4. Language Quality Monitoring
```python
# Monitor per-language quality during training
config.enable_language_quality_monitoring = True
config.per_language_validation = True
config.language_specific_metrics = True
```

## 🎯 Training Scenarios

### Scenario 1: New Model from Scratch (Single Language)
```bash
python train_full_enhanced_xtts.py \
    --config config/single_lang_config.json \
    --language en \
    --train_csv data/english_train.csv \
    --eval_csv data/english_eval.csv \
    --epochs 1000 \
    --batch_size 4
```

### Scenario 2: Multilingual Model (All Languages)
```bash
python train_multilingual_enhanced_xtts.py \
    --train_csv data/multilingual_train.csv \
    --eval_csv data/multilingual_eval.csv \
    --language en \
    --epochs 1500 \
    --batch_size 6
```

### Scenario 3: Fine-tune Existing Model
```bash
python train_multilingual_enhanced_xtts.py \
    --train_csv data/spanish_train.csv \
    --eval_csv data/spanish_eval.csv \
    --language es \
    --checkpoint models/enhanced_english/best_model.pth \
    --epochs 500 \
    --batch_size 2
```

### Scenario 4: Voice Cloning (Language-Specific)
```bash
python train_multilingual_enhanced_xtts.py \
    --train_csv data/voice_clone_train.csv \
    --eval_csv data/voice_clone_eval.csv \
    --language en \
    --speaker_reference data/target_speaker.wav \
    --epochs 800 \
    --batch_size 4
```

## 📊 Data Preparation for Multilingual Training

### CSV Format
```csv
audio_file,text,speaker_name,language,emotion
audio/en_001.wav,"Hello world",speaker_1,en,neutral
audio/es_001.wav,"Hola mundo",speaker_1,es,neutral
audio/fr_001.wav,"Bonjour monde",speaker_1,fr,neutral
```

### Data Distribution Recommendations
- **Single Language**: 10,000+ samples
- **Multilingual**: 5,000+ samples per language
- **Fine-tuning**: 1,000+ samples minimum
- **Voice Cloning**: 500+ samples, same speaker

## 🚀 Phase 2 Enhanced Features for Languages

### Mamba Architecture
- **Best for**: Asian languages (zh-cn, ja, ko)
- **Benefits**: Superior sequence modeling for complex scripts
- **Usage**: `use_mamba: true`

### Mixture of Experts (MoE)
- **Best for**: Multilingual training
- **Benefits**: Specialized experts per language
- **Usage**: `use_moe: true, moe_num_experts: 8`

### Neural Codec
- **Best for**: All languages, especially tonal
- **Benefits**: Enhanced audio quality
- **Usage**: `use_neural_codec: true`

### Flash Attention
- **Best for**: Long sequences, all languages
- **Benefits**: Memory efficiency, faster training
- **Usage**: `use_flash_attention: true`

## 🔍 Troubleshooting Language Issues

### Issue: Language Mixing in Output
**Solution**: Increase repetition penalty and language conditioning
```python
config.repetition_penalty = 3.0          # Higher
config.language_conditioning_strength = 1.8  # Stronger
config.temperature = 0.65                # Lower
```

### Issue: Poor Quality for Non-English Languages
**Solution**: Use language-specific optimization
```python
# For Asian languages
config.use_mamba = True
config.max_text_length = 400
config.language_conditioning_strength = 1.5

# For Arabic/RTL
config.enable_rtl_processing = True
config.language_conditioning_strength = 1.3
```

### Issue: Slow Convergence in Multilingual Training
**Solution**: Use weighted sampling and transfer learning
```python
config.use_language_weighted_sampler = True
config.cross_lingual_conditioning = True
config.lr = 3e-5  # Lower learning rate
```

## 📈 Performance Optimization by Language

### Memory Requirements
- **English**: 8GB VRAM (batch_size=4)
- **Multilingual**: 12GB VRAM (batch_size=6)
- **Asian Languages**: 10GB VRAM (longer sequences)
- **Fine-tuning**: 6GB VRAM (batch_size=2)

### Training Time Estimates
- **Single Language**: 12-24 hours (1000 epochs)
- **Multilingual**: 24-48 hours (1500 epochs)  
- **Fine-tuning**: 6-12 hours (500 epochs)
- **Voice Cloning**: 8-16 hours (800 epochs)

## 🎉 Ready-to-Use Examples

All the configuration files and training scripts are ready in your workspace:

- `train_multilingual_enhanced_xtts.py` - Complete multilingual training
- `config/multilingual_training_config.json` - Example configuration
- `demo_language_configs.py` - Configuration examples

## 🚀 Next Steps

1. **Choose your scenario** (single language, multilingual, fine-tuning)
2. **Prepare your data** in the CSV format shown above
3. **Select configuration** based on your target language(s)
4. **Start training** with the provided scripts
5. **Monitor language-specific metrics** during training

Your Enhanced XTTS system is now ready for advanced multilingual training with optimal language configuration!
