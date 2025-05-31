# 🎙️ Coqui AI Trainer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A comprehensive, production-ready training framework for Text-to-Speech (TTS) models, with enhanced XTTS implementations and state-of-the-art features.

## ✨ Key Features

### 🚀 Enhanced XTTS Training
- **Advanced Architecture**: Phase 2 enhancements with Mamba, Flash Attention, RoPE, and Mixture of Experts
- **Production Ready**: Optimized training pipeline with quality monitoring
- **Multi-language Support**: Comprehensive language configuration system
- **Quality Improvements**: Advanced loss functions and training techniques

### 🏗️ Framework Capabilities
- **Modular Design**: Flexible architecture for various TTS models
- **Distributed Training**: Multi-GPU and distributed training support
- **Advanced Optimization**: DeepSpeed integration and automatic mixed precision
- **Comprehensive Monitoring**: Real-time training metrics and quality assessment

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/coqui-ai-Trainer.git
cd coqui-ai-Trainer

# Install dependencies
pip install -r requirements.txt
```

### Basic Training
```bash
# Train an enhanced XTTS model
python examples/xtts/train_xtts_enhanced.py \
    --config configs/xtts/enhanced_xtts_config.json \
    --data_path ./data/your_dataset \
    --output_path ./output/my_model

# Quick validation
python scripts/validation/quick_validate.py --model_path ./output/my_model
```

## 📁 Project Structure

This project follows a well-organized structure for optimal development workflow:

```
├── 📂 src/trainer/           # Core training framework
├── 📂 examples/              # Usage examples and demos
├── 📂 scripts/               # Automation and utility scripts
├── 📂 configs/               # Model configurations
├── 📂 tests/                 # Test suites
├── 📂 docs/                  # Documentation
├── 📂 data/                  # Training datasets
└── 📂 output/                # Model outputs
```

📖 **See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization**

## 🎯 Training Examples

### Enhanced XTTS Training
```bash
# Basic enhanced training
python examples/xtts/train_xtts_enhanced.py \
    --output_path ./models/my_voice \
    --train_csv ./data/train.csv \
    --eval_csv ./data/eval.csv \
    --language en

# Advanced training with Phase 2 enhancements
python scripts/training/launch_full_enhanced_training.py \
    --config configs/xtts/phase2_xtts_config.json \
    --use_phase2_enhancements \
    --use_mamba \
    --use_flash_attention
```

### Multi-language Training
```bash
# Configure for your language
python scripts/utilities/setup_custom_training.py \
    --language es \
    --config_output ./configs/my_spanish_config.json

# Train with custom configuration
python examples/xtts/train_xtts_enhanced.py \
    --config ./configs/my_spanish_config.json
```

## 🔧 Key Scripts and Tools

### Training Scripts
- `examples/xtts/train_xtts_enhanced.py` - Enhanced XTTS training
- `scripts/training/launch_full_enhanced_training.py` - Automated training launcher
- `scripts/training/run_full_enhanced_training.sh` - Shell-based training

### Validation and Testing
- `scripts/validation/validate_enhanced_xtts.py` - Comprehensive model validation
- `scripts/validation/quick_validate.py` - Quick quality check
- `scripts/validation/phase2_final_validation.py` - Phase 2 feature validation

### Utilities
- `scripts/utilities/prepare_training_data.py` - Data preparation
- `scripts/utilities/setup_custom_training.py` - Custom configuration setup
- `scripts/utilities/migrate_to_phase2.py` - Model migration tool

## 📊 Performance Features

### Advanced Training Techniques
- **Gradient Accumulation**: Efficient large batch training
- **Mixed Precision**: Faster training with reduced memory usage
- **Learning Rate Scheduling**: Optimized convergence
- **Quality Monitoring**: Real-time audio quality assessment

### Model Enhancements
- **Streaming Architecture**: Real-time TTS capability
- **Neural Codec**: High-quality audio compression
- **Prosody Control**: Fine-grained speech characteristics
- **Multi-speaker Support**: Voice adaptation and cloning

## 📚 Documentation

### User Guides
- **[Getting Started](docs/guides/FULL_ENHANCED_TRAINING_GUIDE.md)** - Complete training walkthrough
- **[Language Configuration](docs/guides/LANGUAGE_CONFIGURATION_GUIDE.md)** - Multi-language setup
- **[Advanced Usage](docs/guides/ADVANCED_USAGE_GUIDE.md)** - Advanced features and optimization

### Development
- **[Roadmap](docs/roadmaps/XTTS_PHASE2_ROADMAP.md)** - Future development plans
- **[Architecture](docs/implementation/)** - Technical implementation details
- **[Contributing](docs/CONTRIBUTING.md)** - Development guidelines

### Status Reports
- **[Phase 1 Completion](docs/status-reports/PHASE1_COMPLETION_REPORT.md)** - Phase 1 achievements
- **[Phase 2 Completion](docs/status-reports/PHASE2_COMPLETION_REPORT.md)** - Latest enhancements
- **[Project Status](docs/status-reports/PROJECT_STATUS.md)** - Current development status

## 🛠️ Configuration Management

### Pre-configured Setups
```bash
# Enhanced XTTS (recommended)
configs/xtts/enhanced_xtts_config.json

# Full advanced features
configs/xtts/full_enhanced_config.json

# Phase 2 with state-of-the-art features
configs/xtts/phase2_xtts_config.json

# Multi-language support
configs/xtts/multilingual_training_config.json
```

### Custom Configuration
```bash
# Generate custom config for your use case
python scripts/utilities/setup_custom_training.py \
    --language your_language \
    --features enhanced,streaming,quality_monitoring \
    --output ./configs/custom_config.json
```

## 🧪 Testing and Validation

### Automated Testing
```bash
# Run full test suite
python -m pytest tests/

# Quick integration tests
python tests/unit/test_enhanced_integration.py

# Validate Phase 2 features
python tests/unit/test_phase2_integration.py
```

### Quality Assessment
```bash
# Comprehensive model validation
python scripts/validation/validate_enhanced_xtts.py \
    --model_path ./output/my_model \
    --test_data ./data/test_set

# Voice quality monitoring
python examples/demos/voice_quality_monitor.py \
    --model_path ./output/my_model
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- **Coqui AI Team** - Original framework foundation
- **Community Contributors** - Enhancements and improvements
- **Research Community** - Advanced techniques and methodologies

## 📞 Support

- **Documentation**: Comprehensive guides in `docs/`
- **Examples**: Working examples in `examples/`
- **Issues**: GitHub issues for bug reports and feature requests
- **Community**: Join our community discussions

---

**Ready to create amazing voices? Start with our [Quick Start Guide](docs/guides/FULL_ENHANCED_TRAINING_GUIDE.md)!** 🎤✨
