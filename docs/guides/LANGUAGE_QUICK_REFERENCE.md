# 🌍 Enhanced XTTS Language Configuration Quick Reference

## ✅ Supported Languages (17 Total)
```
en es fr de it pt pl tr ru nl cs ar zh-cn ja hu ko hi
```

## 🚀 Quick Start Commands

### Single Language Training
```bash
python train_full_enhanced_xtts.py \
    --config config/enhanced_xtts_config.json \
    --language en \
    --train_csv data/train.csv \
    --eval_csv data/eval.csv
```

### Multilingual Training  
```bash
python train_multilingual_enhanced_xtts.py \
    --train_csv data/multilingual_train.csv \
    --eval_csv data/multilingual_eval.csv \
    --language en
```

### Fine-tuning
```bash
python train_multilingual_enhanced_xtts.py \
    --train_csv data/spanish_train.csv \
    --eval_csv data/spanish_eval.csv \
    --language es \
    --checkpoint models/english_model.pth
```

## 📋 Key Configuration Parameters

| Parameter | Description | Range | Example |
|-----------|-------------|-------|---------|
| `languages` | Supported language codes | List | `["en", "es", "fr"]` |
| `default_language` | Primary training language | String | `"en"` |
| `target_language` | Optimization target | String | `"es"` |
| `temperature` | Generation creativity | 0.6-0.8 | `0.75` |
| `repetition_penalty` | Language mixing prevention | 2.0-3.5 | `2.5` |
| `language_conditioning_strength` | Language enforcement | 1.0-2.0 | `1.5` |

## 🎯 Language-Specific Optimization

### English (Baseline)
```json
{
  "temperature": 0.75,
  "repetition_penalty": 2.5,
  "language_conditioning_strength": 1.0
}
```

### Romance Languages (es, fr, it, pt)
```json
{
  "temperature": 0.70,
  "repetition_penalty": 2.8,
  "language_conditioning_strength": 1.2
}
```

### Asian Languages (zh-cn, ja, ko)
```json
{
  "temperature": 0.65,
  "repetition_penalty": 3.2,
  "language_conditioning_strength": 1.5,
  "max_text_length": 400
}
```

### German/Dutch (Complex Morphology)
```json
{
  "temperature": 0.68,
  "repetition_penalty": 3.0,
  "language_conditioning_strength": 1.3
}
```

## 🔧 Enhanced Features by Use Case

### Multilingual Training
```json
{
  "use_language_weighted_sampler": true,
  "cross_lingual_conditioning": true,
  "use_moe": true,
  "moe_num_experts": 8
}
```

### Asian Languages
```json
{
  "use_mamba": true,
  "use_neural_codec": true,
  "use_rope": true
}
```

### Fine-tuning
```json
{
  "lr": 1e-5,
  "batch_size": 2,
  "epochs": 500,
  "cross_lingual_conditioning": true
}
```

## 📊 Training Recommendations

| Scenario | Epochs | Batch Size | Learning Rate | Memory |
|----------|--------|------------|---------------|---------|
| Single Language | 1000 | 4 | 5e-5 | 8GB |
| Multilingual | 1500 | 6 | 3e-5 | 12GB |
| Fine-tuning | 500 | 2 | 1e-5 | 6GB |
| Voice Cloning | 800 | 4 | 5e-5 | 8GB |

## 🎁 Ready-to-Use Files

- `config/multilingual_training_config.json` - Complete multilingual config
- `train_multilingual_enhanced_xtts.py` - Training script with language support
- `LANGUAGE_CONFIGURATION_GUIDE.md` - Detailed documentation
- `demo_language_setup.py` - Configuration examples

## 🚨 Common Issues & Solutions

### Language Mixing
**Problem**: Model switches languages mid-sentence  
**Solution**: Increase `repetition_penalty` to 3.0+ and `language_conditioning_strength` to 1.5+

### Poor Non-English Quality
**Problem**: English sounds great, other languages poor  
**Solution**: Use language-weighted sampling and language-specific parameters

### Slow Training
**Problem**: Multilingual training is very slow  
**Solution**: Use MoE (`use_moe: true`) and FlashAttention (`use_flash_attention: true`)

## 🎯 Example Training Command (Copy & Paste Ready)

```bash
# Multilingual training with all 17 languages
python train_multilingual_enhanced_xtts.py \
    --train_csv data/multilingual_train.csv \
    --eval_csv data/multilingual_eval.csv \
    --language en \
    --output_path models/enhanced_multilingual \
    --batch_size 6 \
    --epochs 1500

# Single language fine-tuning
python train_multilingual_enhanced_xtts.py \
    --train_csv data/spanish_data.csv \
    --eval_csv data/spanish_eval.csv \
    --language es \
    --checkpoint models/enhanced_english/best_model.pth \
    --output_path models/enhanced_spanish \
    --batch_size 2 \
    --epochs 500
```

## ✨ Phase 2 Enhanced Features

- ✅ **Mamba Architecture**: Best for Asian languages
- ✅ **Mixture of Experts**: Specialized language experts  
- ✅ **Neural Codec**: Enhanced audio quality
- ✅ **Flash Attention**: Memory efficient training
- ✅ **RoPE**: Better positional encoding
- ✅ **Quality Monitoring**: Per-language metrics

Your Enhanced XTTS is ready for professional multilingual TTS training! 🎉
