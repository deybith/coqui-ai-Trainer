# Full Enhanced XTTS Training Guide

This guide explains how to train the full version of enhanced XTTS that combines both Phase 1 and Phase 2 enhancements.

## Overview

The full enhanced XTTS model includes:

### Phase 1 Enhancements
- **Neural Codec Integration**: EnCodec for high-quality audio compression/decompression
- **Streaming Architecture**: Real-time inference capabilities
- **Quality Monitoring**: Advanced metrics and monitoring systems
- **Advanced Loss Functions**: Improved training stability

### Phase 2 Enhancements
- **Mamba/State Space Models**: Linear complexity attention mechanisms
- **Mixture of Experts (MoE)**: Scalable model specialization
- **Flash Attention 2.0**: Memory-efficient attention computation
- **Rotary Position Embedding (RoPE)**: Better positional encoding

## Prerequisites

### System Requirements
- CUDA-capable GPU with at least 8GB VRAM (16GB+ recommended)
- Python 3.8+
- PyTorch 2.0+
- 32GB+ RAM recommended

### Dependencies
```bash
pip install torch>=2.0 transformers>=4.30 torchaudio tensorboard tqdm numpy matplotlib
```

## Data Preparation

### 1. Organize Your Data
Create the following directory structure:
```
data/
├── audio/           # Audio files (.wav format, 22kHz recommended)
│   ├── speaker1_001.wav
│   ├── speaker1_002.wav
│   ├── speaker2_001.wav
│   └── ...
└── metadata.txt     # Metadata file
```

### 2. Create Metadata File
Format: `audio_filename|transcription_text|speaker_id`

Example `metadata.txt`:
```
speaker1_001.wav|Hello, this is a sample transcription.|speaker1
speaker1_002.wav|The weather is nice today.|speaker1
speaker2_001.wav|Good morning everyone.|speaker2
speaker2_002.wav|How are you doing?|speaker2
```

### 3. Audio Requirements
- Format: WAV files
- Sample rate: 22050 Hz (recommended)
- Duration: 1-11 seconds per file
- Quality: Clean, noise-free audio preferred

## Training Configuration

### 1. Basic Configuration
The configuration file `config/full_enhanced_config.json` contains all training parameters:

```json
{
  "data_path": "./data",
  "use_phase1": true,
  "use_phase2": true,
  "d_model": 1024,
  "n_layers": 12,
  "n_heads": 16,
  "batch_size": 4,
  "learning_rate": 1e-4,
  "num_epochs": 100
}
```

### 2. Phase Configuration Options

#### Phase 1 Options
- `use_phase1`: Enable Phase 1 enhancements
- `use_neural_codec`: Enable neural codec
- `use_streaming`: Enable streaming architecture
- `use_quality_monitoring`: Enable quality monitoring

#### Phase 2 Options
- `use_phase2`: Enable Phase 2 enhancements
- `use_mamba`: Enable Mamba/SSM layers
- `use_flash_attention`: Enable Flash Attention 2.0
- `use_rope`: Enable Rotary Position Embedding
- `use_moe`: Enable Mixture of Experts

### 3. Hardware-Specific Settings

#### For GPUs with 8-12GB VRAM:
```json
{
  "batch_size": 2,
  "gradient_checkpointing": true,
  "mixed_precision": true,
  "d_model": 768
}
```

#### For GPUs with 16GB+ VRAM:
```json
{
  "batch_size": 4,
  "gradient_checkpointing": true,
  "mixed_precision": true,
  "d_model": 1024
}
```

#### For GPUs with 24GB+ VRAM:
```json
{
  "batch_size": 8,
  "gradient_checkpointing": false,
  "mixed_precision": true,
  "d_model": 1536
}
```

## Running Training

### 1. Quick Start (Full Enhanced)
```bash
# Make sure you're in the project directory
cd /home/ubuntu/projects/coqui-ai-Trainer

# Run training with both Phase 1 and Phase 2
python train_full_enhanced_xtts.py \
    --config config/full_enhanced_config.json \
    --use_phase1 \
    --use_phase2 \
    --output_dir ./output \
    --data_path ./data
```

### 2. Using the Launcher Script
```bash
# Make script executable (if not already)
chmod +x run_full_enhanced_training.sh

# Run the training
./run_full_enhanced_training.sh
```

### 3. Training Variations

#### Phase 2 Only (Advanced Neural Architecture)
```bash
python train_full_enhanced_xtts.py \
    --config config/full_enhanced_config.json \
    --use_phase2 \
    --output_dir ./output_phase2 \
    --data_path ./data
```

#### Phase 1 Only (Neural Codec + Streaming)
```bash
python train_full_enhanced_xtts.py \
    --config config/full_enhanced_config.json \
    --use_phase1 \
    --output_dir ./output_phase1 \
    --data_path ./data
```

#### Original XTTS (Baseline)
```bash
python train_full_enhanced_xtts.py \
    --config config/full_enhanced_config.json \
    --output_dir ./output_baseline \
    --data_path ./data
```

## Monitoring Training

### 1. Terminal Output
The training script provides real-time progress updates:
```
🚀 Training Configuration:
   Phase 1 enabled: True
   Phase 2 enabled: True
   Model size: 1024d, 12L, 16H
   Batch size: 4
   Learning rate: 0.0001

Epoch 1/100: 100%|██████| 250/250 [15:23<00:00, text_loss=2.34, mel_loss=1.87, total_loss=4.21]
```

### 2. TensorBoard
```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir=./output/logs

# Open browser to: http://localhost:6006
```

### 3. Key Metrics to Monitor
- **Text Loss**: Should decrease steadily
- **Mel Loss**: Audio reconstruction quality
- **Total Loss**: Combined loss
- **Learning Rate**: Should follow the schedule
- **GPU Memory**: Monitor for out-of-memory issues

## Checkpoints and Output

### Directory Structure
```
output/
├── checkpoints/          # Model checkpoints
│   ├── best_model.pth   # Best validation model
│   ├── epoch_5.pth      # Periodic saves
│   ├── epoch_10.pth
│   └── final_model.pth  # Final trained model
├── logs/                 # TensorBoard logs
└── config.json          # Training configuration used
```

### Loading Checkpoints
```python
# Load the best model
checkpoint = torch.load('output/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### 1. Out of Memory Errors
- Reduce `batch_size` in config
- Enable `gradient_checkpointing`
- Enable `mixed_precision`
- Reduce `d_model` size

### 2. Slow Training
- Increase `batch_size` if memory allows
- Disable `gradient_checkpointing` on high-memory GPUs
- Use multiple GPUs if available

### 3. Poor Quality Results
- Increase training data quantity and quality
- Adjust learning rate (try 5e-5 or 2e-4)
- Increase model size (`d_model`, `n_layers`)
- Train for more epochs

### 4. Component Not Available Errors
If you see warnings about Phase 1 or Phase 2 components not being available:
- Ensure all enhanced components are properly installed
- Check import paths in the training script
- Verify that the enhanced model files exist

## Advanced Configuration

### 1. Mixture of Experts Settings
```json
{
  "use_moe": true,
  "moe_num_experts": 8,
  "moe_top_k": 2
}
```

### 2. Mamba/SSM Settings
```json
{
  "use_mamba": true,
  "mamba_d_state": 16
}
```

### 3. Custom Scheduler Settings
```json
{
  "scheduler": "cosine_with_warmup",
  "warmup_steps": 1000
}
```

## Performance Expectations

### Training Time
- **Small dataset (1-2 hours audio)**: 2-4 hours on RTX 3090
- **Medium dataset (10-20 hours audio)**: 1-2 days on RTX 3090
- **Large dataset (100+ hours audio)**: 1-2 weeks on RTX 3090

### Model Quality
- Phase 1 + Phase 2 provides the best quality
- Phase 2 alone offers excellent efficiency
- Phase 1 alone provides good streaming capabilities

## Next Steps

After training completes:
1. Test the model with inference scripts
2. Fine-tune on specific speakers if needed
3. Evaluate using quality metrics
4. Deploy for production use

For inference and deployment, refer to the enhanced inference documentation.
