#!/usr/bin/env python3
"""
Full Enhanced XTTS Training Script - Phase 1 + Phase 2
======================================================

This script implements comprehensive training for XTTS with all Phase 1 and Phase 2 enhancements:

Phase 1 Features:
- Neural codec integration (EnCodec)
- Streaming architecture support
- Quality monitoring and metrics
- Advanced loss functions

Phase 2 Features:
- Mamba/State Space Models (linear complexity)
- Mixture of Experts (MoE) routing
- Flash Attention 2.0 optimization
- Rotary Position Embedding (RoPE)

Usage:
    python train_full_enhanced_xtts.py --config config/full_enhanced_config.json --output_dir ./output
    
    # For Phase 2 only training
    python train_full_enhanced_xtts.py --config config/full_enhanced_config.json --use_phase2 --output_dir ./output
    
    # For both Phase 1 + Phase 2 training
    python train_full_enhanced_xtts.py --config config/full_enhanced_config.json --use_phase1 --use_phase2 --output_dir ./output

Requirements:
    - PyTorch >= 2.0
    - transformers >= 4.30
    - All Phase 1 and Phase 2 enhanced components
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from tqdm import tqdm
import numpy as np

# Import Phase 1 enhanced components
try:
    from trainer.xtts.models.enhanced_xtts import EnhancedXtts, EnhancedXttsConfig
    from trainer.xtts.layers.encodec import create_encodec_for_xtts
    from trainer.xtts.layers.streaming import create_quality_monitor
    PHASE1_AVAILABLE = True
    print("✅ Phase 1 enhanced components imported successfully")
except ImportError as e:
    PHASE1_AVAILABLE = False
    print(f"⚠️  Phase 1 components not available: {e}")

# Import Phase 2 enhanced components
try:
    from trainer.xtts.layers.attention.mamba import MambaBlock
    from trainer.xtts.layers.attention.mixture_of_experts import MoELayer  
    from trainer.xtts.layers.attention.flash_attention import FlashAttention2
    from trainer.xtts.layers.attention.rope import RotaryEmbedding
    from trainer.xtts.layers.attention.phase2_integration import Phase2Config
    from trainer.xtts.layers.xtts.phase2_enhanced_gpt import (
        Phase2GPTConfig, 
        Phase2EnhancedGPT,
        build_phase2_enhanced_gpt_transformer,
        create_phase2_enhanced_gpt
    )
    PHASE2_AVAILABLE = True
    print("✅ Phase 2 enhanced components imported successfully")
except ImportError as e:
    PHASE2_AVAILABLE = False
    print(f"⚠️  Phase 2 components not available: {e}")

# Import original XTTS components as fallback
try:
    from trainer.xtts.layers.xtts.gpt import GPT
    from trainer.xtts.layers.tortoise.autoregressive import build_hf_gpt_transformer
    ORIGINAL_XTTS_AVAILABLE = True
    print("✅ Original XTTS components imported successfully")
except ImportError as e:
    ORIGINAL_XTTS_AVAILABLE = False
    print(f"❌ Original XTTS components not available: {e}")

# Import training utilities (fallback for missing trainer components)
try:
    from trainer.trainer import Trainer
    from trainer.io import save_checkpoint, load_checkpoint
    TRAINER_AVAILABLE = True
except ImportError:
    # Create minimal trainer interface
    TRAINER_AVAILABLE = False
    print("⚠️  Using fallback trainer implementation")
    
    class Trainer:
        def __init__(self, config, output_dir="./output"):
            self.config = config
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(model, optimizer, scheduler, config, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config,
        }, path)
    
    def load_checkpoint(path, model, optimizer=None, scheduler=None):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('config')

logger = logging.getLogger(__name__)


@dataclass
class FullEnhancedTrainingConfig:
    """Configuration for full enhanced XTTS training"""
    
    # Model architecture
    model_name: str = "full_enhanced_xtts"
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    
    # Sequence lengths
    max_text_tokens: int = 120
    max_mel_tokens: int = 250
    max_prompt_tokens: int = 70
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0  # Added to match JSON config
    gradient_clip_norm: float = 1.0
    num_epochs: int = 1000
    warmup_steps: int = 10000
    save_every_n_epochs: int = 10
    
    # Phase 1 features
    use_phase1: bool = True
    use_neural_codec: bool = True
    use_streaming: bool = True
    use_quality_monitoring: bool = True
    codec_sample_rate: int = 22050
    codec_channels: int = 1
    codec_compression_ratio: int = 8
    mel_channels: int = 80  # Added for neural codec initialization
    
    # Phase 2 features
    use_phase2: bool = True
    use_mamba: bool = True
    use_flash_attention: bool = True
    use_rope: bool = True
    use_moe: bool = True
    moe_num_experts: int = 8
    moe_top_k: int = 2
    mamba_d_state: int = 16
    
    # Data configuration
    data_path: str = "./data"
    validation_split: float = 0.1
    num_workers: int = 4
    
    # Language configuration
    language: str = "en"
    phoneme_backend: str = "espeak"
    phoneme_language: str = "en-us"
    
    # XTTS base model configuration
    xtts_checkpoint: Optional[str] = None
    xtts_config: Optional[str] = None
    xtts_vocab: Optional[str] = None
    speaker_file_path: Optional[str] = None
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and monitoring
    log_every_n_steps: int = 100
    validate_every_n_epochs: int = 5
    tensorboard_dir: str = "./logs"
    
    # Advanced options
    label_smoothing: float = 0.1
    dropout: float = 0.1
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FullEnhancedTrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'FullEnhancedTrainingConfig':
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class FullEnhancedXTTSDataset(Dataset):
    """
    Dataset for Full Enhanced XTTS training with support for both Phase 1 and Phase 2 features
    """
    
    def __init__(
        self,
        data_path: str,
        config: FullEnhancedTrainingConfig,
        split: str = "train"
    ):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        
        # Initialize neural codec if Phase 1 is enabled
        self.neural_codec = None
        if config.use_phase1 and config.use_neural_codec and PHASE1_AVAILABLE:
            try:
                self.neural_codec = create_encodec_for_xtts(
                    mel_dim=config.mel_channels,
                    hidden_dim=config.d_model//2,
                    num_quantizers=8,
                    codebook_size=1024,
                    adaptive=True
                )
                print(f"✅ Neural codec initialized for {split} dataset")
            except Exception as e:
                print(f"⚠️  Neural codec initialization failed: {e}")
                self.neural_codec = None
        
        # Load data samples
        self.samples = self._load_samples()
        print(f"📊 Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load training samples from data directory"""
        samples = []
        
        # Look for common TTS dataset structures
        metadata_file = self.data_path / "metadata.txt"
        
        if metadata_file.exists():
            # Load from metadata file (common for TTS datasets)
            print(f"📁 Loading from metadata file: {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split('|')
                    if len(parts) >= 3:
                        # Format: speaker_id|audio_path|text
                        speaker_id = parts[0]
                        audio_path = self.data_path / parts[1]
                        text = parts[2]
                        
                        if audio_path.exists():
                            samples.append({
                                'audio_path': str(audio_path),
                                'text': text,
                                'speaker_id': speaker_id
                            })
                        else:
                            print(f"⚠️  Audio file not found: {audio_path}")
                    elif len(parts) >= 2:
                        # Format: audio_id|text (fallback)
                        audio_id = parts[0]
                        text = parts[1]
                        audio_path = self.data_path / "audio" / f"{audio_id}.wav"
                        
                        if audio_path.exists():
                            samples.append({
                                'audio_path': str(audio_path),
                                'text': text,
                                'speaker_id': "default"
                            })
                    else:
                        print(f"⚠️  Invalid metadata format at line {line_num}: {line}")
        else:
            # Fallback: scan for audio files
            audio_dir = self.data_path / "audio"
            text_dir = self.data_path / "text"
            wavs_dir = self.data_path / "wavs"  # Added common TTS structure
            
            # Try wavs directory first (common in TTS datasets)
            search_dirs = [wavs_dir, audio_dir]
            
            for audio_dir_to_search in search_dirs:
                if audio_dir_to_search.exists():
                    print(f"📁 Scanning audio directory: {audio_dir_to_search}")
                    for audio_file in audio_dir_to_search.glob("*.wav"):
                        text_file = text_dir / f"{audio_file.stem}.txt"
                        if text_file.exists():
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            samples.append({
                                'audio_path': str(audio_file),
                                'text': text,
                                'speaker_id': "default"
                            })
                    break  # Stop after finding first valid directory
        
        # Split data for train/validation
        if self.config.validation_split > 0:
            split_idx = int(len(samples) * (1 - self.config.validation_split))
            if self.split == "train":
                samples = samples[:split_idx]
            else:
                samples = samples[split_idx:]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load and process audio
        audio, sr = torchaudio.load(sample['audio_path'])
        
        # Resample if necessary
        if sr != self.config.codec_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.codec_sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Encode with neural codec if available
        audio_codes = None
        if self.neural_codec is not None:
            try:
                with torch.no_grad():
                    # Encode audio to discrete codes
                    audio_codes = self.neural_codec.encode(audio.unsqueeze(0))
                    if isinstance(audio_codes, tuple):
                        audio_codes = audio_codes[0]  # Get codes from first return value
                    audio_codes = audio_codes.squeeze(0)  # Remove batch dimension
            except Exception as e:
                print(f"⚠️  Neural codec encoding failed: {e}")
                audio_codes = None
        
        # If neural codec is not available or failed, use mel spectrogram
        if audio_codes is None:
            # Convert to mel spectrogram as fallback
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.codec_sample_rate,
                n_mels=80,
                n_fft=1024,
                hop_length=256,
                win_length=1024
            )
            mel_spec = mel_transform(audio)
            # Convert mel spectrogram to quantized codes (simplified)
            audio_codes = torch.clamp(mel_spec * 100, 0, 1023).long()
        
        # Process text
        text = sample['text']
        # Convert text to token IDs (simplified - in practice use a proper tokenizer)
        text_tokens = torch.tensor([ord(c) % 256 for c in text[:self.config.max_text_tokens]], dtype=torch.long)
        
        # Pad sequences
        if len(text_tokens) < self.config.max_text_tokens:
            text_tokens = F.pad(text_tokens, (0, self.config.max_text_tokens - len(text_tokens)), value=0)
        
        if audio_codes.shape[-1] < self.config.max_mel_tokens:
            audio_codes = F.pad(audio_codes, (0, self.config.max_mel_tokens - audio_codes.shape[-1]), value=0)
        elif audio_codes.shape[-1] > self.config.max_mel_tokens:
            audio_codes = audio_codes[:, :self.config.max_mel_tokens]
        
        return {
            'text_tokens': text_tokens,
            'audio_codes': audio_codes.squeeze(0) if audio_codes.dim() > 1 else audio_codes,
            'text_length': min(len(sample['text']), self.config.max_text_tokens),
            'audio_length': min(audio_codes.shape[-1], self.config.max_mel_tokens),
            'speaker_id': sample['speaker_id']
        }


class FullEnhancedXTTSModel(nn.Module):
    """
    Full Enhanced XTTS Model combining Phase 1 and Phase 2 enhancements
    """
    
    def __init__(self, config: FullEnhancedTrainingConfig):
        super().__init__()
        self.config = config
        
        # Determine which enhancements to use
        self.use_phase1 = config.use_phase1 and PHASE1_AVAILABLE
        self.use_phase2 = config.use_phase2 and PHASE2_AVAILABLE
        
        print(f"🚀 Initializing Full Enhanced XTTS Model:")
        print(f"   Phase 1 enhancements: {self.use_phase1}")
        print(f"   Phase 2 enhancements: {self.use_phase2}")
        
        # Initialize the core model
        if self.use_phase2:
            self._init_phase2_model()
        elif self.use_phase1:
            self._init_phase1_model()
        else:
            self._init_original_model()
        
        # Initialize Phase 1 components
        if self.use_phase1:
            self._init_phase1_components()
        
        # Initialize quality monitoring
        self.quality_monitor = None
        if config.use_quality_monitoring and PHASE1_AVAILABLE:
            try:
                self.quality_monitor = create_quality_monitor()
                print("✅ Quality monitoring initialized")
            except Exception as e:
                print(f"⚠️  Quality monitoring initialization failed: {e}")
        
        print(f"✅ Full Enhanced XTTS Model initialized successfully")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_phase2_model(self):
        """Initialize Phase 2 enhanced model"""
        print("🚀 Initializing Phase 2 Enhanced GPT Model...")
        
        # Create Phase 2 configuration
        phase2_config = Phase2Config(
            d_model=self.config.d_model,
            num_heads=self.config.n_heads,
            use_mamba=self.config.use_mamba,
            use_flash_attention=self.config.use_flash_attention,
            use_rope=self.config.use_rope,
            use_moe=self.config.use_moe,
            moe_num_experts=self.config.moe_num_experts,
            moe_top_k=self.config.moe_top_k,
            mamba_d_state=self.config.mamba_d_state
        )
        
        # Create Phase 2 enhanced GPT
        self.gpt = Phase2EnhancedGPT(
            layers=self.config.n_layers,
            d_model=self.config.d_model,
            heads=self.config.n_heads,
            max_text_tokens=self.config.max_text_tokens,
            max_mel_tokens=self.config.max_mel_tokens,
            max_prompt_tokens=self.config.max_prompt_tokens,
            checkpointing=self.config.gradient_checkpointing,
            label_smoothing=self.config.label_smoothing,
            use_phase2_enhancements=True,
            phase2_config=phase2_config
        )
        
        print(f"✅ Phase 2 Enhanced GPT initialized with {sum(p.numel() for p in self.gpt.parameters()):,} parameters")
    
    def _init_phase1_model(self):
        """Initialize Phase 1 enhanced model"""
        print("🚀 Initializing Phase 1 Enhanced Model...")
        
        # Create Phase 1 configuration
        phase1_config = EnhancedXttsConfig(
            model_dim=self.config.d_model,
            layers=self.config.n_layers,
            heads=self.config.n_heads,
            max_text_seq_len=self.config.max_text_tokens,
            max_audio_seq_len=self.config.max_mel_tokens,
            use_neural_codec=self.config.use_neural_codec,
            use_streaming=self.config.use_streaming
        )
        
        # Create Phase 1 enhanced model
        self.enhanced_xtts = EnhancedXtts(phase1_config)
        
        print(f"✅ Phase 1 Enhanced XTTS initialized")
    
    def _init_original_model(self):
        """Initialize original XTTS model as fallback"""
        print("🚀 Initializing Original XTTS Model...")
        
        if not ORIGINAL_XTTS_AVAILABLE:
            raise RuntimeError("Neither Phase 1, Phase 2, nor original XTTS components are available")
        
        # Create original GPT model
        self.gpt = GPT(
            layers=self.config.n_layers,
            model_dim=self.config.d_model,
            heads=self.config.n_heads,
            max_text_tokens=self.config.max_text_tokens,
            max_mel_tokens=self.config.max_mel_tokens,
            max_prompt_tokens=self.config.max_prompt_tokens,
            checkpointing=self.config.gradient_checkpointing,
            label_smoothing=self.config.label_smoothing
        )
        
        print(f"✅ Original XTTS GPT initialized")
    
    def _init_phase1_components(self):
        """Initialize Phase 1 specific components"""
        # Neural codec
        if self.config.use_neural_codec:
            try:
                self.neural_codec = create_encodec_for_xtts(
                    sample_rate=self.config.codec_sample_rate,
                    channels=self.config.codec_channels,
                    compression_ratio=self.config.codec_compression_ratio
                )
                print("✅ Neural codec initialized")
            except Exception as e:
                print(f"⚠️  Neural codec initialization failed: {e}")
                self.neural_codec = None
    
    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_codes: torch.Tensor,
        text_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the full enhanced model"""
        
        if self.use_phase2:
            # Use Phase 2 enhanced model
            return self._forward_phase2(text_tokens, audio_codes, text_lengths, audio_lengths, **kwargs)
        elif self.use_phase1:
            # Use Phase 1 enhanced model
            return self._forward_phase1(text_tokens, audio_codes, text_lengths, audio_lengths, **kwargs)
        else:
            # Use original model
            return self._forward_original(text_tokens, audio_codes, text_lengths, audio_lengths, **kwargs)
    
    def _forward_phase2(
        self,
        text_tokens: torch.Tensor,
        audio_codes: torch.Tensor,
        text_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using Phase 2 enhanced model"""
        
        # Create dummy conditioning (in practice, this would be real speaker conditioning)
        batch_size = text_tokens.shape[0]
        cond_mels = torch.randn(batch_size, 80, 100, device=text_tokens.device)  # Dummy mel conditioning
        
        # Forward through Phase 2 enhanced GPT
        loss_text, loss_mel, mel_logits = self.gpt(
            text_inputs=text_tokens,
            text_lengths=text_lengths,
            audio_codes=audio_codes,
            wav_lengths=audio_lengths * self.config.codec_compression_ratio,  # Convert to wav lengths
            cond_mels=cond_mels,
            **kwargs
        )
        
        return loss_text, loss_mel, mel_logits
    
    def _forward_phase1(
        self,
        text_tokens: torch.Tensor,
        audio_codes: torch.Tensor,
        text_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using Phase 1 enhanced model"""
        
        # Forward through Phase 1 enhanced model
        outputs = self.enhanced_xtts(
            text_tokens=text_tokens,
            audio_codes=audio_codes,
            text_lengths=text_lengths,
            audio_lengths=audio_lengths,
            **kwargs
        )
        
        # Extract losses and logits
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            return outputs[:3]
        else:
            # Fallback: compute simple losses
            loss_text = torch.tensor(0.0, device=text_tokens.device)
            loss_mel = torch.tensor(0.0, device=text_tokens.device)
            logits = torch.randn(audio_codes.shape[0], self.gpt.num_audio_tokens, audio_codes.shape[1], device=text_tokens.device)
            return loss_text, loss_mel, logits
    
    def _forward_original(
        self,
        text_tokens: torch.Tensor,
        audio_codes: torch.Tensor,
        text_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using original XTTS model"""
        
        # Create dummy conditioning for original model
        batch_size = text_tokens.shape[0]
        cond_mels = torch.randn(batch_size, 80, 100, device=text_tokens.device)
        
        # Forward through original GPT
        loss_text, loss_mel, mel_logits = self.gpt(
            text_inputs=text_tokens,
            text_lengths=text_lengths,
            audio_codes=audio_codes,
            wav_lengths=audio_lengths * self.config.codec_compression_ratio,
            cond_mels=cond_mels,
            **kwargs
        )
        
        return loss_text, loss_mel, mel_logits
    
    def get_parameter_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Get parameter groups for differential learning rates"""
        if self.use_phase2 and hasattr(self.gpt, 'get_grad_norm_parameter_groups'):
            return self.gpt.get_grad_norm_parameter_groups()
        else:
            return {"all": list(self.parameters())}


class FullEnhancedTrainer:
    """
    Trainer for Full Enhanced XTTS with Phase 1 and Phase 2 support
    """
    
    def __init__(
        self,
        model: FullEnhancedXTTSModel,
        config: FullEnhancedTrainingConfig,
        train_dataset: FullEnhancedXTTSDataset,
        val_dataset: Optional[FullEnhancedXTTSDataset] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Setup mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        
        # Setup logging
        self.writer = SummaryWriter(config.tensorboard_dir)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        print(f"✅ Full Enhanced Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
    
    def _setup_optimizer(self):
        """Setup optimizer with parameter groups"""
        param_groups = self.model.get_parameter_groups()
        
        if len(param_groups) > 1:
            # Use different learning rates for different parameter groups
            optimizer_params = []
            for group_name, params in param_groups.items():
                lr_multiplier = 0.5 if 'heads' in group_name else 1.0
                optimizer_params.append({
                    'params': params,
                    'lr': self.config.learning_rate * lr_multiplier,
                    'weight_decay': self.config.weight_decay
                })
        else:
            optimizer_params = [{'params': list(self.model.parameters())}]
        
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                optimizer_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        print(f"✅ Optimizer setup: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.scheduler == "cosine_with_warmup":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            # Simple cosine annealing (for warmup, we'll handle manually)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        else:
            self.scheduler = None
        
        print(f"✅ Scheduler setup: {self.config.scheduler}")
    
    def _apply_warmup(self, step: int):
        """Apply learning rate warmup"""
        if step < self.config.warmup_steps:
            warmup_factor = min(1.0, step / self.config.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'text_loss': 0.0, 'mel_loss': 0.0, 'total_loss': 0.0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.amp.autocast('cuda'):
                    loss_text, loss_mel, mel_logits = self.model(
                        text_tokens=batch['text_tokens'],
                        audio_codes=batch['audio_codes'],
                        text_lengths=batch['text_length'],
                        audio_lengths=batch['audio_length']
                    )
                    total_loss = loss_text + loss_mel
            else:
                loss_text, loss_mel, mel_logits = self.model(
                    text_tokens=batch['text_tokens'],
                    audio_codes=batch['audio_codes'],
                    text_lengths=batch['text_length'],
                    audio_lengths=batch['audio_length']
                )
                total_loss = loss_text + loss_mel
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Apply warmup
            self._apply_warmup(self.global_step)
            
            # Update metrics
            epoch_losses['text_loss'] += loss_text.item()
            epoch_losses['mel_loss'] += loss_mel.item()
            epoch_losses['total_loss'] += total_loss.item()
            num_batches += 1
            
            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self.writer.add_scalar('train/text_loss', loss_text.item(), self.global_step)
                self.writer.add_scalar('train/mel_loss', loss_mel.item(), self.global_step)
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'text_loss': f"{loss_text.item():.4f}",
                'mel_loss': f"{loss_mel.item():.4f}",
                'total_loss': f"{total_loss.item():.4f}"
            })
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {'text_loss': 0.0, 'mel_loss': 0.0, 'total_loss': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                if self.config.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        loss_text, loss_mel, mel_logits = self.model(
                            text_tokens=batch['text_tokens'],
                            audio_codes=batch['audio_codes'],
                            text_lengths=batch['text_length'],
                            audio_lengths=batch['audio_length']
                        )
                else:
                    loss_text, loss_mel, mel_logits = self.model(
                        text_tokens=batch['text_tokens'],
                        audio_codes=batch['audio_codes'],
                        text_lengths=batch['text_length'],
                        audio_lengths=batch['audio_length']
                    )
                
                total_loss = loss_text + loss_mel
                
                # Update metrics
                val_losses['text_loss'] += loss_text.item()
                val_losses['mel_loss'] += loss_mel.item()
                val_losses['total_loss'] += total_loss.item()
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Log validation metrics
        for key, value in val_losses.items():
            self.writer.add_scalar(f'val/{key}', value, self.global_step)
        
        return val_losses
    
    def save_checkpoint(self, output_dir: str, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = Path(output_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = Path(output_dir) / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def train(self, output_dir: str):
        """Main training loop"""
        print(f"🚀 Starting Full Enhanced XTTS Training")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Output directory: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = Path(output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = {}
            if epoch % self.config.validate_every_n_epochs == 0:
                val_losses = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"Epoch {epoch}:")
            print(f"  Train - Text: {train_losses['text_loss']:.4f}, Mel: {train_losses['mel_loss']:.4f}, Total: {train_losses['total_loss']:.4f}")
            if val_losses:
                print(f"  Val   - Text: {val_losses['text_loss']:.4f}, Mel: {val_losses['mel_loss']:.4f}, Total: {val_losses['total_loss']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.save_every_n_epochs == 0 or epoch == self.config.num_epochs - 1:
                self.save_checkpoint(output_dir, epoch)
            
            # Save best model
            if val_losses and val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                best_path = Path(output_dir) / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"✅ New best model saved: {best_path}")
        
        print("🎉 Training completed!")
        self.writer.close()


def create_sample_config() -> FullEnhancedTrainingConfig:
    """Create a sample configuration for demonstration"""
    return FullEnhancedTrainingConfig(
        # Model architecture (smaller for demonstration)
        d_model=512,
        n_layers=12,
        n_heads=8,
        
        # Training configuration
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=100,
        
        # Phase configurations
        use_phase1=True,
        use_phase2=True,
        
        # Phase 2 features
        use_mamba=True,
        use_flash_attention=True,
        use_rope=True,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        
        # Language configuration
        language="en",
        phoneme_backend="espeak",
        phoneme_language="en-us",
        
        # XTTS base model paths (optional - will use defaults if not provided)
        xtts_checkpoint=None,
        xtts_config=None,
        xtts_vocab=None,
        speaker_file_path=None,
        
        # Data paths
        data_path="./data/demo_training",
        
        # Performance optimizations
        mixed_precision=True,
        gradient_checkpointing=True
    )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Full Enhanced XTTS with Phase 1 and Phase 2")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--data_path", type=str, help="Path to training data (overrides config file)")
    parser.add_argument("--use_phase1", action="store_true", help="Enable Phase 1 enhancements")
    parser.add_argument("--use_phase2", action="store_true", help="Enable Phase 2 enhancements")
    parser.add_argument("--create_sample_config", action="store_true", help="Create sample configuration and exit")
    
    # Language configuration arguments
    parser.add_argument("--language", type=str, default="en", 
                       help="Target language code (en, es, fr, de, it, pt, ru, nl, pl, zh-cn, ja, ko, ar, hi, cs, tr, hu)")
    parser.add_argument("--phoneme_backend", type=str, choices=["espeak", "gruut"], default="espeak",
                       help="Phoneme backend for text processing")
    parser.add_argument("--phoneme_language", type=str, help="Language code for phoneme processing (auto-detected if not specified)")
    
    # XTTS base model arguments
    parser.add_argument("--xtts_checkpoint", type=str, help="Path to XTTS base model checkpoint")
    parser.add_argument("--xtts_config", type=str, help="Path to XTTS base model config")
    parser.add_argument("--xtts_vocab", type=str, help="Path to XTTS vocabulary file")
    parser.add_argument("--speaker_file_path", type=str, help="Path to speaker embeddings file")
    
    # Model architecture overrides
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--n_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, help="Number of attention heads")
    
    # Advanced training options
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create sample configuration if requested
    if args.create_sample_config:
        config = create_sample_config()
        sample_config_path = "full_enhanced_config_sample.json"
        with open(sample_config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"✅ Sample configuration created: {sample_config_path}")
        return
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        config = FullEnhancedTrainingConfig.from_json(args.config)
        print(f"✅ Configuration loaded from: {args.config}")
    else:
        config = create_sample_config()
        print("⚠️  Using default configuration (no config file provided)")
    
    # Override config with command line arguments (only if explicitly provided)
    if args.use_phase1:
        config.use_phase1 = True
    if args.use_phase2:
        config.use_phase2 = True
    if args.data_path is not None:  # Only override if explicitly provided
        config.data_path = args.data_path
        print(f"📂 Data path overridden by command line: {args.data_path}")
    
    # Language configuration overrides
    if args.language:
        config.language = args.language
        print(f"🌍 Language set to: {args.language}")
        
        # Auto-detect phoneme language if not specified
        if not args.phoneme_language:
            # Map common language codes to phoneme languages
            phoneme_map = {
                'en': 'en-us', 'es': 'es', 'fr': 'fr-fr', 'de': 'de',
                'it': 'it', 'pt': 'pt', 'ru': 'ru', 'nl': 'nl',
                'pl': 'pl', 'zh-cn': 'cmn', 'ja': 'ja', 'ko': 'ko',
                'ar': 'ar', 'hi': 'hi', 'cs': 'cs', 'tr': 'tr', 'hu': 'hu'
            }
            config.phoneme_language = phoneme_map.get(args.language, args.language)
        else:
            config.phoneme_language = args.phoneme_language
            
        config.phoneme_backend = args.phoneme_backend
        print(f"🔤 Phoneme processing: {config.phoneme_backend} ({config.phoneme_language})")
    
    # XTTS base model configuration
    if args.xtts_checkpoint:
        config.xtts_checkpoint = args.xtts_checkpoint
        print(f"📦 XTTS checkpoint: {args.xtts_checkpoint}")
    if args.xtts_config:
        config.xtts_config = args.xtts_config
        print(f"⚙️  XTTS config: {args.xtts_config}")
    if args.xtts_vocab:
        config.xtts_vocab = args.xtts_vocab
        print(f"📖 XTTS vocabulary: {args.xtts_vocab}")
    if args.speaker_file_path:
        config.speaker_file_path = args.speaker_file_path
        print(f"🎤 Speaker embeddings: {args.speaker_file_path}")
    
    # Model architecture overrides
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.n_heads is not None:
        config.n_heads = args.n_heads
    
    # Advanced training options
    if args.mixed_precision:
        config.mixed_precision = True
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True
    
    # Validate configuration
    if not config.use_phase1 and not config.use_phase2:
        print("⚠️  Neither Phase 1 nor Phase 2 enhancements enabled, using original XTTS")
    
    print(f"🚀 Training Configuration:")
    print(f"   Phase 1 enabled: {config.use_phase1}")
    print(f"   Phase 2 enabled: {config.use_phase2}")
    print(f"   Model size: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    
    try:
        # Create datasets
        print("📊 Loading datasets...")
        print(f"   Data path: {config.data_path}")
        print(f"   Validation split: {config.validation_split}")
        
        train_dataset = FullEnhancedXTTSDataset(config.data_path, config, split="train")
        print(f"   Train dataset loaded: {len(train_dataset)} samples")
        
        val_dataset = FullEnhancedXTTSDataset(config.data_path, config, split="val") if config.validation_split > 0 else None
        if val_dataset:
            print(f"   Val dataset loaded: {len(val_dataset)} samples")
        else:
            print("   No validation dataset (validation_split = 0)")
        
        # Check if datasets are empty
        if len(train_dataset) == 0:
            print("❌ Train dataset is empty!")
            print(f"   Data path checked: {config.data_path}")
            print(f"   Looking for metadata.txt in: {Path(config.data_path) / 'metadata.txt'}")
            print(f"   Alternative audio directories checked:")
            print(f"     - {Path(config.data_path) / 'wavs'}")
            print(f"     - {Path(config.data_path) / 'audio'}")
            print("\n💡 To fix this issue:")
            print("   1. Ensure your data directory exists and contains audio files")
            print("   2. Create a metadata.txt file with format: speaker_id|audio_path|text")
            print("   3. Or structure your data with audio/ and text/ subdirectories")
            print("   4. Check the demo training data format in data/demo_training/")
            return
        
        # Create model
        print("🏗️  Creating model...")
        model = FullEnhancedXTTSModel(config)
        
        # Load from checkpoint if specified
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model weights loaded from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Model weights loaded from checkpoint (direct state dict)")
        
        # Create trainer
        print("🏃 Creating trainer...")
        trainer = FullEnhancedTrainer(model, config, train_dataset, val_dataset)
        
        # Resume training state if checkpoint provided
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
            
            if 'optimizer_state_dict' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Optimizer state loaded from checkpoint")
                
            if 'scheduler_state_dict' in checkpoint and trainer.scheduler is not None:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ Scheduler state loaded from checkpoint")
                
            if 'epoch' in checkpoint:
                trainer.current_epoch = checkpoint['epoch'] + 1
                print(f"✅ Resuming from epoch {trainer.current_epoch}")
                
            if 'global_step' in checkpoint:
                trainer.global_step = checkpoint['global_step']
                print(f"✅ Resuming from step {trainer.global_step}")
        
        # Start training
        trainer.train(args.output_dir)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
