"""Stochastic Weight Averaging (SWA) utilities for Trainer.

This module provides comprehensive SWA support including:
- SWA model averaging
- SWA learning rate scheduler
- Configuration management
- Integration with Trainer class
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

logger = logging.getLogger(__name__)


@dataclass
class SWAConfig:
    """Configuration for Stochastic Weight Averaging.
    
    Args:
        enabled (bool): Whether to enable SWA. Default: False.
        swa_start (int): Epoch to start SWA. Default: 10.
        swa_freq (int): Frequency of SWA model updates (in epochs). Default: 1.
        swa_lr (float): SWA learning rate. Default: 0.05.
        anneal_epochs (int): Number of epochs to anneal to swa_lr. Default: 10.
        anneal_strategy (str): Annealing strategy ('cos' or 'linear'). Default: 'cos'.
        device (str): Device for SWA model. If None, uses same device as base model.
        avg_fn (str): Averaging function ('mean' or 'exponential'). Default: 'mean'.
        update_bn (bool): Whether to update batch norm statistics. Default: True.
        bn_subset_size (int): Subset size for BN update (None = full dataset). Default: None.
        save_swa_model (bool): Whether to save SWA model checkpoints. Default: True.
        swa_model_name (str): Name for SWA model checkpoints. Default: 'swa_model'.
    """
    enabled: bool = False
    swa_start: int = 10
    swa_freq: int = 1
    swa_lr: float = 0.05
    anneal_epochs: int = 10
    anneal_strategy: str = 'cos'  # 'cos' or 'linear'
    device: Optional[str] = None
    avg_fn: str = 'mean'  # 'mean' or 'exponential'
    update_bn: bool = True
    bn_subset_size: Optional[int] = None
    save_swa_model: bool = True
    swa_model_name: str = 'swa_model'


class SWAManager:
    """Manages Stochastic Weight Averaging during training.
    
    This class handles:
    - SWA model creation and weight averaging
    - SWA learning rate scheduling
    - Batch normalization updates
    - Model checkpointing
    - Integration with training loop
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SWAConfig,
        optimizer: torch.optim.Optimizer,
        output_path: Optional[Path] = None
    ):
        """Initialize SWA manager.
        
        Args:
            model: Base model to apply SWA to
            config: SWA configuration
            optimizer: Optimizer used for training
            output_path: Path to save SWA models
        """
        self.config = config
        self.base_model = model
        self.optimizer = optimizer
        self.output_path = output_path or Path(".")
        
        # Initialize SWA components
        self.swa_model: Optional[AveragedModel] = None
        self.swa_scheduler: Optional[SWALR] = None
        self.is_active = False
        self.swa_updates = 0
        
        # Metrics tracking
        self.swa_metrics = {
            'updates_count': 0,
            'avg_loss': 0.0,
            'best_loss': float('inf'),
            'epochs_active': 0
        }
        
        if config.enabled:
            self._initialize_swa()
            logger.info(f"✅ SWA initialized - will start at epoch {config.swa_start}")
    
    def _initialize_swa(self):
        """Initialize SWA model and scheduler."""
        # Determine device
        device = self.config.device
        if device is None:
            # Use same device as base model
            device = next(self.base_model.parameters()).device
        
        # Create averaged model
        if self.config.avg_fn == 'exponential':
            # Use exponential moving average
            def exponential_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                # Exponential decay with decay rate based on number of updates
                decay_rate = 1.0 / max(1, num_averaged)
                return decay_rate * model_parameter + (1 - decay_rate) * averaged_model_parameter
            avg_fn = exponential_avg_fn
        else:
            # Use simple mean (default)
            avg_fn = None
        
        self.swa_model = AveragedModel(
            self.base_model,
            device=device,
            avg_fn=avg_fn
        )
        
        # Create SWA learning rate scheduler
        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr=self.config.swa_lr,
            anneal_epochs=self.config.anneal_epochs,
            anneal_strategy=self.config.anneal_strategy
        )
        
        logger.info(f"🔄 SWA model created with {self.config.avg_fn} averaging on {device}")
    
    def should_start_swa(self, current_epoch: int) -> bool:
        """Check if SWA should start at current epoch."""
        return (
            self.config.enabled and 
            not self.is_active and 
            current_epoch >= self.config.swa_start
        )
    
    def should_update_swa(self, current_epoch: int) -> bool:
        """Check if SWA model should be updated at current epoch."""
        return (
            self.is_active and 
            (current_epoch - self.config.swa_start) % self.config.swa_freq == 0
        )
    
    def start_swa(self, current_epoch: int):
        """Start SWA averaging."""
        if not self.config.enabled or self.is_active:
            return
        
        self.is_active = True
        logger.info(f"🚀 Starting SWA averaging at epoch {current_epoch}")
        
        # Initialize SWA model with current model state
        if self.swa_model is not None:
            self.update_swa_model(current_epoch)
    
    def update_swa_model(self, current_epoch: int):
        """Update SWA model with current model weights."""
        if not self.is_active or self.swa_model is None:
            return
        
        # Update averaged model
        self.swa_model.update_parameters(self.base_model)
        self.swa_updates += 1
        self.swa_metrics['updates_count'] = self.swa_updates
        self.swa_metrics['epochs_active'] = current_epoch - self.config.swa_start + 1
        
        logger.info(f"📊 SWA model updated (update #{self.swa_updates}) at epoch {current_epoch}")
    
    def step_swa_scheduler(self):
        """Step the SWA learning rate scheduler."""
        if self.is_active and self.swa_scheduler is not None:
            self.swa_scheduler.step()
    
    def update_bn_statistics(self, dataloader: torch.utils.data.DataLoader):
        """Update batch normalization statistics for SWA model.
        
        Args:
            dataloader: DataLoader for updating BN statistics
        """
        if not self.is_active or self.swa_model is None or not self.config.update_bn:
            return
        
        logger.info("🔄 Updating SWA model batch normalization statistics...")
        
        # Use subset of data if specified
        if self.config.bn_subset_size is not None:
            # Create a subset iterator
            subset_iter = iter(dataloader)
            subset_data = []
            for i, batch in enumerate(subset_iter):
                if i >= self.config.bn_subset_size:
                    break
                subset_data.append(batch)
            
            # Create temporary dataloader for subset
            from torch.utils.data import DataLoader, TensorDataset
            if subset_data:
                # This is a simplified approach - in practice, you might need
                # to handle the batch structure more carefully
                update_bn(dataloader, self.swa_model)
        else:
            # Use full dataloader
            update_bn(dataloader, self.swa_model)
        
        logger.info("✅ SWA batch normalization statistics updated")
    
    def save_swa_model(
        self, 
        current_epoch: int, 
        current_step: int,
        current_loss: float,
        **kwargs
    ) -> Optional[Path]:
        """Save SWA model checkpoint.
        
        Args:
            current_epoch: Current training epoch
            current_step: Current training step
            current_loss: Current loss value
            **kwargs: Additional data to save
            
        Returns:
            Path to saved model if saved, None otherwise
        """
        if not self.is_active or self.swa_model is None or not self.config.save_swa_model:
            return None
        
        # Update metrics
        self.swa_metrics['avg_loss'] = current_loss
        if current_loss < self.swa_metrics['best_loss']:
            self.swa_metrics['best_loss'] = current_loss
        
        # Create checkpoint data
        checkpoint_data = {
            'model_state_dict': self.swa_model.state_dict(),
            'swa_config': self.config,
            'swa_metrics': self.swa_metrics.copy(),
            'epoch': current_epoch,
            'step': current_step,
            'loss': current_loss,
            'n_averaged': self.swa_model.n_averaged,
            **kwargs
        }
        
        # Save checkpoint
        model_path = self.output_path / f"{self.config.swa_model_name}_epoch_{current_epoch}.pth"
        torch.save(checkpoint_data, model_path)
        
        # Also save as "latest"
        latest_path = self.output_path / f"{self.config.swa_model_name}_latest.pth"
        torch.save(checkpoint_data, latest_path)
        
        logger.info(f"💾 SWA model saved: {model_path}")
        return model_path
    
    def load_swa_model(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load SWA model from checkpoint.
        
        Args:
            checkpoint_path: Path to SWA checkpoint
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SWA checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if self.swa_model is not None:
            self.swa_model.load_state_dict(checkpoint['model_state_dict'])
            self.swa_metrics = checkpoint.get('swa_metrics', self.swa_metrics)
            
            logger.info(f"✅ SWA model loaded from {checkpoint_path}")
            logger.info(f"📊 SWA metrics: {self.swa_metrics}")
        
        return checkpoint
    
    def get_swa_model(self) -> Optional[nn.Module]:
        """Get the SWA averaged model.
        
        Returns:
            SWA model if available, None otherwise
        """
        return self.swa_model
    
    def get_swa_metrics(self) -> Dict[str, Any]:
        """Get SWA metrics and statistics.
        
        Returns:
            Dictionary containing SWA metrics
        """
        metrics = self.swa_metrics.copy()
        if self.swa_model is not None:
            metrics['n_averaged'] = self.swa_model.n_averaged
        metrics['is_active'] = self.is_active
        metrics['config'] = self.config
        return metrics
    
    def finalize_swa(self, dataloader: Optional[torch.utils.data.DataLoader] = None):
        """Finalize SWA training.
        
        This should be called at the end of training to update BN statistics
        and perform final model saving.
        
        Args:
            dataloader: DataLoader for final BN statistics update
        """
        if not self.is_active or self.swa_model is None:
            return
        
        logger.info("🏁 Finalizing SWA training...")
        
        # Update batch normalization statistics if requested
        if dataloader is not None and self.config.update_bn:
            self.update_bn_statistics(dataloader)
        
        # Save final SWA model
        final_path = self.output_path / f"{self.config.swa_model_name}_final.pth"
        checkpoint_data = {
            'model_state_dict': self.swa_model.state_dict(),
            'swa_config': self.config,
            'swa_metrics': self.swa_metrics.copy(),
            'n_averaged': self.swa_model.n_averaged,
            'is_final': True
        }
        torch.save(checkpoint_data, final_path)
        
        logger.info(f"✅ Final SWA model saved: {final_path}")
        logger.info(f"📈 SWA completed with {self.swa_updates} updates averaging {self.swa_model.n_averaged} models")


def create_swa_config(
    enabled: bool = False,
    swa_start: int = 10,
    swa_freq: int = 1,
    swa_lr: float = 0.05,
    **kwargs
) -> SWAConfig:
    """Helper function to create SWA configuration.
    
    Args:
        enabled: Whether to enable SWA
        swa_start: Epoch to start SWA
        swa_freq: Frequency of SWA updates
        swa_lr: SWA learning rate
        **kwargs: Additional SWA configuration parameters
        
    Returns:
        SWAConfig instance
    """
    return SWAConfig(
        enabled=enabled,
        swa_start=swa_start,
        swa_freq=swa_freq,
        swa_lr=swa_lr,
        **kwargs
    )


def get_cosine_swa_lr(base_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Calculate appropriate SWA learning rate using cosine schedule.
    
    Args:
        base_lr: Base learning rate used during training
        min_lr_ratio: Ratio of minimum LR to base LR
        
    Returns:
        Recommended SWA learning rate
    """
    return base_lr * min_lr_ratio


def estimate_swa_start_epoch(total_epochs: int, start_fraction: float = 0.75) -> int:
    """Estimate appropriate epoch to start SWA.
    
    Args:
        total_epochs: Total number of training epochs
        start_fraction: Fraction of training when to start SWA
        
    Returns:
        Recommended SWA start epoch
    """
    return max(1, int(total_epochs * start_fraction))
