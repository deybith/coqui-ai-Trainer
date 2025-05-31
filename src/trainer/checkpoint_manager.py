"""Checkpoint Manager for training management and persistence.

This module provides a centralized checkpoint management system that handles
saving, loading, and managing model checkpoints during training.
"""

import json
import logging
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import fsspec
import torch

from .config import TrainerConfig
from .generic_utils import to_cuda
from .io import load_fsspec, save_fsspec
from .model import TrainerModel

logger = logging.getLogger("trainer")


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint file.
    
    This class stores comprehensive information about a checkpoint,
    including training state, performance metrics, and file information.
    """
    
    # Basic checkpoint information
    step: int
    epoch: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Loss information
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    best_loss: Optional[float] = None
    
    # File information
    file_path: str = ""
    file_size: int = 0
    
    # Training configuration hash (for compatibility checking)
    config_hash: str = ""
    
    # Additional metrics
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management.
    
    This class defines how checkpoints should be saved, managed, and cleaned up.
    """
    
    # Save intervals
    save_step: int = 10000
    save_best_after: int = 0
    
    # Checkpoint retention
    keep_n_checkpoints: int = 5
    keep_all_best: bool = False
    
    # File naming
    checkpoint_prefix: str = "checkpoint"
    best_model_prefix: str = "best_model"
    
    # Validation
    validate_checkpoints: bool = True
    validate_on_save: bool = True
    
    # Compression
    compress_checkpoints: bool = False
    
    # Backup
    create_backup: bool = False
    backup_dir: str = "backups"
    
    # Metadata tracking
    save_metadata: bool = True
    metadata_file: str = "checkpoint_metadata.json"


class CheckpointManager:
    """Centralized checkpoint management system.
    
    This class provides comprehensive checkpoint management functionality including
    saving, loading, validation, cleanup, and metadata tracking. It serves as a
    single point of control for all checkpoint operations during training.
    
    Features:
        - Automatic checkpoint saving with configurable intervals
        - Best model tracking and preservation
        - Automatic cleanup of old checkpoints
        - Checkpoint validation and integrity checking
        - Metadata tracking for all checkpoints
        - Support for custom checkpoint types
        - Backup and recovery functionality
        - Thread-safe operations
    
    Args:
        output_path: Directory where checkpoints will be saved
        config: Checkpoint configuration settings
        save_func: Optional custom save function for dashboard integration
    
    Example:
        >>> manager = CheckpointManager(
        ...     output_path="./checkpoints",
        ...     config=CheckpointConfig(keep_n_checkpoints=3)
        ... )
        >>> manager.save_checkpoint(model, optimizer, step=1000, epoch=1)
        >>> metadata = manager.get_checkpoint_metadata()
        >>> manager.cleanup_old_checkpoints()
    """
    
    def __init__(
        self,
        output_path: Union[str, os.PathLike],
        config: Optional[CheckpointConfig] = None,
        save_func: Optional[Callable[[Any, Union[str, os.PathLike]], None]] = None
    ):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.config = config or CheckpointConfig()
        self.save_func = save_func
        
        # Initialize filesystem
        self.fs = fsspec.get_mapper(str(self.output_path)).fs
        
        # Metadata storage
        self.metadata_path = self.output_path / self.config.metadata_file
        self.metadata: Dict[str, CheckpointMetadata] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_checkpoint_path: Optional[str] = None
        
        logger.info(f"CheckpointManager initialized at {self.output_path}")
    
    def save_checkpoint(
        self,
        model: TrainerModel,
        optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        step: int,
        epoch: int,
        config: Optional[Union[Dict[str, Any], TrainerConfig]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        train_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        custom_metrics: Optional[Dict[str, Any]] = None,
        checkpoint_type: str = "regular",
        **kwargs
    ) -> str:
        """Save a model checkpoint with comprehensive metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer(s) to save
            step: Current training step
            epoch: Current epoch
            config: Training configuration
            scaler: AMP scaler if used
            train_loss: Current training loss
            eval_loss: Current evaluation loss
            custom_metrics: Additional metrics to save
            checkpoint_type: Type of checkpoint ('regular', 'best', 'custom')
            **kwargs: Additional data to save
            
        Returns:
            Path to the saved checkpoint file
        """
        start_time = time.time()
        
        # Determine checkpoint filename
        if checkpoint_type == "best":
            filename = f"{self.config.best_model_prefix}_{step}.pth"
        else:
            filename = f"{self.config.checkpoint_prefix}_{step}.pth"
        
        checkpoint_path = self.output_path / filename
        
        # Prepare checkpoint data
        checkpoint_data = self._prepare_checkpoint_data(
            model=model,
            optimizer=optimizer,
            step=step,
            epoch=epoch,
            config=config,
            scaler=scaler,
            train_loss=train_loss,
            eval_loss=eval_loss,
            custom_metrics=custom_metrics,
            **kwargs
        )
        
        # Save checkpoint
        try:
            if self.save_func:
                self.save_func(checkpoint_data, checkpoint_path)
            else:
                save_fsspec(checkpoint_data, checkpoint_path)
            
            # Validate if enabled
            if self.config.validate_on_save:
                if not self._validate_checkpoint(checkpoint_path):
                    raise ValueError(f"Checkpoint validation failed: {checkpoint_path}")
            
            # Create metadata
            metadata = CheckpointMetadata(
                step=step,
                epoch=epoch,
                train_loss=train_loss,
                eval_loss=eval_loss,
                file_path=str(checkpoint_path),
                file_size=os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0,
                config_hash=self._compute_config_hash(config) if config else "",
                learning_rate=self._extract_learning_rate(optimizer),
                custom_metrics=custom_metrics or {}
            )
            
            # Store metadata
            self.metadata[filename] = metadata
            self._save_metadata()
            
            # Create backup if enabled
            if self.config.create_backup:
                self._create_backup(checkpoint_path)
            
            save_time = time.time() - start_time
            logger.info(f"Checkpoint saved: {checkpoint_path} (took {save_time:.2f}s)")
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            raise
    
    def save_best_model(
        self,
        current_loss: Union[float, Dict[str, float]],
        model: TrainerModel,
        optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        step: int,
        epoch: int,
        config: Optional[Union[Dict[str, Any], TrainerConfig]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        **kwargs
    ) -> bool:
        """Save model if it achieves the best performance.
        
        Args:
            current_loss: Current loss value or loss dictionary
            model: Model to save
            optimizer: Optimizer(s) to save
            step: Current training step
            epoch: Current epoch
            config: Training configuration
            scaler: AMP scaler if used
            **kwargs: Additional data to save
            
        Returns:
            True if model was saved as new best, False otherwise
        """
        # Determine if this is the best model
        loss_value = self._extract_target_loss(current_loss)
        
        # Check if we should save (after minimum steps)
        if step < self.config.save_best_after:
            return False
        
        # Check if this is better than current best
        is_best = loss_value < self.best_loss
        
        if is_best:
            # Update best tracking
            self.best_loss = loss_value
            
            # Save as best model
            checkpoint_path = self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                config=config,
                scaler=scaler,
                train_loss=current_loss.get('train_loss') if isinstance(current_loss, dict) else current_loss,
                eval_loss=current_loss.get('eval_loss') if isinstance(current_loss, dict) else None,
                checkpoint_type="best",
                **kwargs
            )
            
            self.best_checkpoint_path = checkpoint_path
            
            # Create symlink to latest best model
            best_symlink = self.output_path / "best_model.pth"
            if best_symlink.exists() or best_symlink.is_symlink():
                best_symlink.unlink()
            
            try:
                best_symlink.symlink_to(Path(checkpoint_path).name)
            except OSError:
                # Fallback: copy file if symlink not supported
                shutil.copy2(checkpoint_path, best_symlink)
            
            # Cleanup old best models if not keeping all
            if not self.config.keep_all_best:
                self._cleanup_old_best_models(keep_current=Path(checkpoint_path).name)
            
            logger.info(f"New best model saved with loss {loss_value:.6f}: {checkpoint_path}")
            return True
        
        return False
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, os.PathLike],
        model: TrainerModel,
        optimizer: Optional[Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        load_optimizer: bool = True,
        load_scaler: bool = True,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """Load a checkpoint and restore model state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scaler: AMP scaler to load state into
            load_optimizer: Whether to load optimizer state
            load_scaler: Whether to load scaler state
            map_location: Device mapping for loading
            
        Returns:
            Dictionary containing checkpoint metadata and state
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Validate checkpoint first
        if self.config.validate_checkpoints:
            if not self._validate_checkpoint(checkpoint_path):
                raise ValueError(f"Checkpoint validation failed: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = load_fsspec(checkpoint_path, map_location=map_location)
        
        # Load model state
        if "model" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model"])
            logger.info("Model state loaded successfully")
        
        # Load optimizer state
        if load_optimizer and optimizer is not None and "optimizer" in checkpoint_data:
            if isinstance(optimizer, list):
                for i, opt in enumerate(optimizer):
                    if i < len(checkpoint_data["optimizer"]):
                        opt.load_state_dict(checkpoint_data["optimizer"][i])
            else:
                optimizer.load_state_dict(checkpoint_data["optimizer"])
            logger.info("Optimizer state loaded successfully")
        
        # Load scaler state
        if load_scaler and scaler is not None and "scaler" in checkpoint_data:
            scaler.load_state_dict(checkpoint_data["scaler"])
            logger.info("Scaler state loaded successfully")
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        checkpoints = self._list_checkpoints(self.config.checkpoint_prefix)
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=self._extract_step_number, reverse=True)
        return checkpoints[0]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best model checkpoint.
        
        Returns:
            Path to best checkpoint or None if no best model found
        """
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            return self.best_checkpoint_path
        
        # Try to find best model files
        best_models = self._list_checkpoints(self.config.best_model_prefix)
        if not best_models:
            return None
        
        # Return the most recent best model
        best_models.sort(key=self._extract_step_number, reverse=True)
        return best_models[0]
    
    def cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints according to retention policy."""
        checkpoints = self._list_checkpoints(self.config.checkpoint_prefix)
        
        if len(checkpoints) <= self.config.keep_n_checkpoints:
            return
        
        # Sort by step number and keep only the most recent ones
        checkpoints.sort(key=self._extract_step_number, reverse=True)
        to_remove = checkpoints[self.config.keep_n_checkpoints:]
        
        for checkpoint_path in to_remove:
            try:
                os.remove(checkpoint_path)
                
                # Remove from metadata
                filename = os.path.basename(checkpoint_path)
                if filename in self.metadata:
                    del self.metadata[filename]
                
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
        
        # Save updated metadata
        self._save_metadata()
    
    def get_checkpoint_metadata(self) -> Dict[str, CheckpointMetadata]:
        """Get metadata for all tracked checkpoints.
        
        Returns:
            Dictionary mapping checkpoint filenames to their metadata
        """
        return self.metadata.copy()
    
    def validate_all_checkpoints(self) -> Dict[str, bool]:
        """Validate all existing checkpoints.
        
        Returns:
            Dictionary mapping checkpoint paths to validation results
        """
        results = {}
        
        all_checkpoints = (
            self._list_checkpoints(self.config.checkpoint_prefix) +
            self._list_checkpoints(self.config.best_model_prefix)
        )
        
        for checkpoint_path in all_checkpoints:
            results[checkpoint_path] = self._validate_checkpoint(checkpoint_path)
        
        return results
    
    def create_checkpoint_summary(self) -> Dict[str, Any]:
        """Create a summary of all checkpoints and their status.
        
        Returns:
            Dictionary containing checkpoint summary information
        """
        regular_checkpoints = self._list_checkpoints(self.config.checkpoint_prefix)
        best_checkpoints = self._list_checkpoints(self.config.best_model_prefix)
        
        summary = {
            "checkpoint_manager_config": asdict(self.config),
            "output_path": str(self.output_path),
            "total_checkpoints": len(regular_checkpoints),
            "total_best_models": len(best_checkpoints),
            "latest_checkpoint": self.get_latest_checkpoint(),
            "best_checkpoint": self.get_best_checkpoint(),
            "best_loss": self.best_loss if self.best_loss != float('inf') else None,
            "total_size_mb": self._calculate_total_size() / (1024 * 1024),
            "metadata_count": len(self.metadata),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    # Private methods
    
    def _prepare_checkpoint_data(
        self,
        model: TrainerModel,
        optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        step: int,
        epoch: int,
        config: Optional[Union[Dict[str, Any], TrainerConfig]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare checkpoint data dictionary."""
        from .config import TrainerConfig
        
        # Get model state
        model_state = model.state_dict()
        
        # Get optimizer state
        if isinstance(optimizer, list):
            optimizer_state = [opt.state_dict() for opt in optimizer]
        else:
            optimizer_state = optimizer.state_dict()
        
        # Get scaler state
        scaler_state = scaler.state_dict() if scaler is not None else None
        
        # Prepare config
        if isinstance(config, TrainerConfig):
            config_dict = config.to_dict()
        elif config is not None:
            config_dict = config
        else:
            config_dict = {}
        
        # Build checkpoint data
        checkpoint_data = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scaler": scaler_state,
            "step": step,
            "epoch": epoch,
            "config": config_dict,
            "date": datetime.now().strftime("%B %d, %Y"),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add any additional data
        checkpoint_data.update(kwargs)
        
        return checkpoint_data
    
    def _validate_checkpoint(self, checkpoint_path: Union[str, os.PathLike]) -> bool:
        """Validate a checkpoint file."""
        try:
            if not os.path.exists(checkpoint_path):
                return False
            
            # Try to load the checkpoint
            checkpoint_data = load_fsspec(checkpoint_path, map_location="cpu")
            
            # Check required fields
            required_fields = ["model", "step", "epoch"]
            for field in required_fields:
                if field not in checkpoint_data:
                    logger.warning(f"Checkpoint {checkpoint_path} missing required field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Checkpoint validation failed for {checkpoint_path}: {e}")
            return False
    
    def _list_checkpoints(self, prefix: str) -> List[str]:
        """List all checkpoint files with given prefix."""
        pattern = f"{prefix}_*.pth"
        checkpoint_files = list(self.output_path.glob(pattern))
        return [str(f) for f in checkpoint_files]
    
    def _extract_step_number(self, checkpoint_path: str) -> int:
        """Extract step number from checkpoint filename."""
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'_(\d+)\.pth$', filename)
        return int(match.group(1)) if match else 0
    
    def _extract_target_loss(self, loss: Union[float, Dict[str, float]]) -> float:
        """Extract target loss value for comparison."""
        if isinstance(loss, dict):
            # Prefer eval_loss, fallback to train_loss
            return loss.get('eval_loss', loss.get('train_loss', float('inf')))
        return loss
    
    def _extract_learning_rate(self, optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]) -> Optional[float]:
        """Extract current learning rate from optimizer."""
        try:
            if isinstance(optimizer, list):
                return optimizer[0].param_groups[0]['lr']
            else:
                return optimizer.param_groups[0]['lr']
        except (IndexError, KeyError):
            return None
    
    def _compute_config_hash(self, config: Union[Dict[str, Any], TrainerConfig]) -> str:
        """Compute hash of configuration for compatibility checking."""
        import hashlib
        
        if hasattr(config, 'to_dict'):
            config_str = str(sorted(config.to_dict().items()))
        else:
            config_str = str(sorted(config.items()))
        
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _cleanup_old_best_models(self, keep_current: str) -> None:
        """Remove old best model files except the current one."""
        best_models = self._list_checkpoints(self.config.best_model_prefix)
        
        for model_path in best_models:
            filename = os.path.basename(model_path)
            if filename != keep_current:
                try:
                    os.remove(model_path)
                    if filename in self.metadata:
                        del self.metadata[filename]
                    logger.info(f"Removed old best model: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old best model {model_path}: {e}")
    
    def _create_backup(self, checkpoint_path: Union[str, os.PathLike]) -> None:
        """Create a backup copy of checkpoint."""
        backup_dir = self.output_path / self.config.backup_dir
        backup_dir.mkdir(exist_ok=True)
        
        filename = os.path.basename(checkpoint_path)
        backup_path = backup_dir / filename
        
        try:
            shutil.copy2(checkpoint_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {checkpoint_path}: {e}")
    
    def _calculate_total_size(self) -> int:
        """Calculate total size of all checkpoints in bytes."""
        total_size = 0
        
        for checkpoint_path in (
            self._list_checkpoints(self.config.checkpoint_prefix) +
            self._list_checkpoints(self.config.best_model_prefix)
        ):
            try:
                total_size += os.path.getsize(checkpoint_path)
            except OSError:
                continue
        
        return total_size
    
    def _load_metadata(self) -> None:
        """Load checkpoint metadata from file."""
        if not self.metadata_path.exists():
            return
        
        try:
            with open(self.metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            for filename, data in metadata_dict.items():
                self.metadata[filename] = CheckpointMetadata.from_dict(data)
            
            logger.info(f"Loaded metadata for {len(self.metadata)} checkpoints")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to file."""
        if not self.config.save_metadata:
            return
        
        try:
            metadata_dict = {
                filename: metadata.to_dict()
                for filename, metadata in self.metadata.items()
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint metadata: {e}")
