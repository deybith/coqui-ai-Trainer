"""
Enhanced XTTS Training Script

This script implements training for the enhanced XTTS model with:
- Neural codec integration
- Advanced loss functions
- Streaming-aware training
- Quality monitoring
- Performance optimization

Usage:
    python train_enhanced_xtts.py --config config/enhanced_xtts_config.json --output_dir ./output
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from tqdm import tqdm
import numpy as np

# Import models and components
from trainer.xtts.models.enhanced_xtts import EnhancedXtts, EnhancedXttsConfig
from trainer.xtts.layers.encodec import create_encodec_for_xtts
from trainer.xtts.layers.streaming import create_quality_monitor

# Import original training components
from trainer.trainer import Trainer
from trainer.io import save_checkpoint

logger = logging.getLogger(__name__)


class EnhancedXTTSDataset(Dataset):
    """
    Dataset for Enhanced XTTS training with codec and streaming support.
    """
    
    def __init__(
        self,
        data_path: str,
        config: EnhancedXttsConfig,
        max_audio_length: int = 220500,  # 10 seconds at 22050 Hz
        max_text_length: int = 200,
    ):
        self.data_path = Path(data_path)
        self.config = config
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        # Initialize audio transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.mel_channels,
            f_min=config.mel_fmin,
            f_max=config.mel_fmax,
        )
        
        logger.info(f"Loaded {len(self.samples)} training samples")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples from metadata."""
        samples = []
        
        # Look for dataset metadata file
        metadata_files = [
            self.data_path / "metadata.json",
            self.data_path / "dataset.json",
            self.data_path / "train.json",
        ]
        
        metadata_file = None
        for file_path in metadata_files:
            if file_path.exists():
                metadata_file = file_path
                break
        
        if metadata_file:
            with open(metadata_file) as f:
                samples = json.load(f)
        else:
            # Generate samples from directory structure
            audio_files = list(self.data_path.glob("**/*.wav"))
            for audio_file in audio_files:
                # Use filename as text (simplified)
                text = audio_file.stem.replace("_", " ")
                samples.append({
                    "audio_path": str(audio_file),
                    "text": text,
                    "speaker_id": "default",
                    "language": "en",
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load audio
        audio_path = sample["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = self.data_path / audio_path
            
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.config.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.config.sample_rate)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Trim or pad audio
        audio = audio.squeeze(0)
        if len(audio) > self.max_audio_length:
            start_idx = torch.randint(0, len(audio) - self.max_audio_length + 1, (1,)).item()
            audio = audio[start_idx:start_idx + self.max_audio_length]
        else:
            # Pad with zeros
            padding = self.max_audio_length - len(audio)
            audio = F.pad(audio, (0, padding))
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(audio)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        mel_spec = mel_spec.transpose(0, 1)  # [T, mel_channels]
        
        # Prepare text tokens (simplified)
        text = sample["text"]
        # In practice, you would use a proper tokenizer here
        text_tokens = [ord(c) % 256 for c in text[:self.max_text_length]]
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        
        # Pad text tokens
        if len(text_tokens) < self.max_text_length:
            padding = self.max_text_length - len(text_tokens)
            text_tokens = F.pad(text_tokens, (0, padding), value=0)
        
        return {
            "audio": audio,
            "mel_spec": mel_spec,
            "text_tokens": text_tokens,
            "text": text,
            "speaker_id": sample.get("speaker_id", "default"),
            "language": sample.get("language", "en"),
        }


class EnhancedXTTSTrainer:
    """
    Trainer for Enhanced XTTS model with advanced features.
    """
    
    def __init__(
        self,
        config: EnhancedXttsConfig,
        model: EnhancedXtts,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "./output",
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        else:
            self.val_loader = None
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss functions
        self.criterion = self._setup_loss_functions()
        
        # Setup logging
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Quality monitoring
        if config.enable_quality_monitoring:
            self.quality_monitor = create_quality_monitor(
                feature_dim=config.mel_channels,
                quality_threshold=config.quality_threshold,
            ).to(self.device)
        else:
            self.quality_monitor = None
        
        logger.info(f"Enhanced XTTS trainer initialized on {self.device}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with advanced features."""
        # Separate parameter groups for different components
        param_groups = []
        
        # Main model parameters
        main_params = []
        codec_params = []
        streaming_params = []
        
        for name, param in self.model.named_parameters():
            if 'neural_codec' in name:
                codec_params.append(param)
            elif 'streaming_decoder' in name:
                streaming_params.append(param)
            else:
                main_params.append(param)
        
        # Different learning rates for different components
        if main_params:
            param_groups.append({
                'params': main_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
            })
        
        if codec_params:
            param_groups.append({
                'params': codec_params,
                'lr': self.config.learning_rate * 0.5,  # Lower LR for codec
                'weight_decay': self.config.weight_decay,
            })
        
        if streaming_params:
            param_groups.append({
                'params': streaming_params,
                'lr': self.config.learning_rate * 0.8,  # Slightly lower LR for streaming
                'weight_decay': self.config.weight_decay,
            })
        
        # Use AdamW optimizer
        optimizer = optim.AdamW(param_groups)
        
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.scheduler_T_0,
            T_mult=self.config.scheduler_T_mult,
            eta_min=self.config.learning_rate * 0.01,
        )
        
        return scheduler
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup loss functions."""
        losses = {}
        
        # Main reconstruction loss
        losses['reconstruction'] = nn.CrossEntropyLoss(ignore_index=-1)
        
        # L1 loss for mel-spectrogram reconstruction
        losses['mel_l1'] = nn.L1Loss()
        
        # Multi-scale STFT loss
        losses['stft'] = MultiScaleSTFTLoss()
        
        # Adversarial loss (if enabled)
        if self.config.use_adversarial_training:
            losses['adversarial'] = nn.BCEWithLogitsLoss()
        
        return losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                text_tokens=batch["text_tokens"],
                audio_features=batch["mel_spec"],
                return_quality_metrics=True,
            )
            
            # Compute losses
            losses = self.compute_losses(outputs, batch)
            total_loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_val,
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update training state
            self.step += 1
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value.item())
            
            # Log to tensorboard
            if self.step % self.config.log_interval == 0:
                self._log_training_step(losses, batch_idx, num_batches)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            # Validation step
            if self.step % self.config.eval_interval == 0 and self.val_loader:
                val_losses = self.validate()
                self._log_validation_step(val_losses)
                self.model.train()  # Back to training mode
        
        # Average epoch losses
        avg_losses = {name: np.mean(values) for name, values in epoch_losses.items()}
        
        return avg_losses
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        losses = {}
        
        # Main reconstruction loss (if applicable)
        if 'logits' in outputs and 'text_tokens' in batch:
            target_tokens = batch['text_tokens']
            
            # Shift targets for autoregressive loss
            if outputs['logits'].shape[1] > target_tokens.shape[1]:
                # Truncate predictions
                logits = outputs['logits'][:, :target_tokens.shape[1]]
            else:
                # Pad targets
                padding = outputs['logits'].shape[1] - target_tokens.shape[1]
                target_tokens = F.pad(target_tokens, (0, padding), value=-1)
                logits = outputs['logits']
            
            recon_loss = self.criterion['reconstruction'](
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )
            losses['reconstruction'] = recon_loss
        
        # Neural codec losses
        if 'codec_info' in outputs and outputs['codec_info']:
            codec_losses = outputs['codec_info'].get('losses', {})
            for loss_name, loss_value in codec_losses.items():
                losses[f'codec_{loss_name}'] = loss_value
        
        # Enhanced losses from model
        enhanced_losses = self.model.compute_loss(outputs, batch, self.step)
        for loss_name, loss_value in enhanced_losses.items():
            if loss_name != 'total':  # Don't double-count total
                losses[loss_name] = loss_value
        
        # Quality-based losses
        if self.quality_monitor and 'quality_metrics' in outputs:
            quality_metrics = outputs['quality_metrics']
            if quality_metrics.get('needs_adjustment', False):
                # Add quality regularization loss
                quality_loss = (1.0 - quality_metrics['current_quality']) * 0.1
                losses['quality_regularization'] = torch.tensor(
                    quality_loss, device=self.device, requires_grad=True
                )
        
        # Compute total loss with weights
        total_loss = 0.0
        loss_weights = {
            'reconstruction': 1.0,
            'codec_reconstruction_loss': 0.5,
            'codec_commitment_loss': 0.1,
            'consistency': 0.2,
            'perceptual': 0.3,
            'quality_regularization': 0.1,
        }
        
        for loss_name, loss_value in losses.items():
            weight = loss_weights.get(loss_name, 0.1)  # Default small weight
            total_loss += weight * loss_value
        
        losses['total'] = total_loss
        
        return losses
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        val_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    text_tokens=batch["text_tokens"],
                    audio_features=batch["mel_spec"],
                    return_quality_metrics=True,
                )
                
                # Compute losses
                losses = self.compute_losses(outputs, batch)
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in val_losses:
                        val_losses[loss_name] = []
                    val_losses[loss_name].append(loss_value.item())
        
        # Average validation losses
        avg_val_losses = {name: np.mean(values) for name, values in val_losses.items()}
        
        return avg_val_losses
    
    def _log_training_step(
        self,
        losses: Dict[str, torch.Tensor],
        batch_idx: int,
        num_batches: int,
    ):
        """Log training step to tensorboard."""
        # Log losses
        for loss_name, loss_value in losses.items():
            self.writer.add_scalar(
                f"train/{loss_name}",
                loss_value.item(),
                self.step,
            )
        
        # Log learning rates
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f"train/lr_group_{i}",
                param_group['lr'],
                self.step,
            )
        
        # Log progress
        progress = (batch_idx + 1) / num_batches
        self.writer.add_scalar("train/epoch_progress", progress, self.step)
    
    def _log_validation_step(self, val_losses: Dict[str, float]):
        """Log validation step to tensorboard."""
        for loss_name, loss_value in val_losses.items():
            self.writer.add_scalar(f"val/{loss_name}", loss_value, self.step)
        
        # Check for best model
        if 'total' in val_losses and val_losses['total'] < self.best_loss:
            self.best_loss = val_losses['total']
            self.save_checkpoint(is_best=True)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at step {self.step}")
        
        # Save latest checkpoint
        latest_path = self.output_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed")
            logger.info(f"Training losses: {train_losses}")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        logger.info("Training completed!")


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for perceptual quality."""
    
    def __init__(self):
        super().__init__()
        self.fft_sizes = [512, 1024, 2048]
        self.hop_sizes = [128, 256, 512]
        self.win_sizes = [512, 1024, 2048]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Compute STFT
            pred_stft = torch.stft(
                pred.view(-1, pred.size(-1)).squeeze(),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                return_complex=True,
                window=torch.hann_window(win_size, device=pred.device),
            )
            target_stft = torch.stft(
                target.view(-1, target.size(-1)).squeeze(),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                return_complex=True,
                window=torch.hann_window(win_size, device=target.device),
            )
            
            # Magnitude loss
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            mag_loss = F.l1_loss(pred_mag, target_mag)
            loss += mag_loss
        
        return loss / len(self.fft_sizes)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Enhanced XTTS Training")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--val_data_path", help="Path to validation data")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = EnhancedXttsConfig()
    config.load_json(args.config)
    
    # Create datasets
    train_dataset = EnhancedXTTSDataset(args.data_path, config)
    
    val_dataset = None
    if args.val_data_path:
        val_dataset = EnhancedXTTSDataset(args.val_data_path, config)
    
    # Create model
    model = EnhancedXtts(config)
    
    # Create trainer
    trainer = EnhancedXTTSTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    main()
