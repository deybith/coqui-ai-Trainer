"""Deepspeed utilities for training optimization.

This module provides comprehensive Deepspeed integration for the Trainer, including:
- Deepspeed engine initialization and configuration
- Model and optimizer wrapping
- Distributed training support
- Memory optimization features
- Checkpoint management
- Zero optimization levels support
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import TrainerModel

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None
    DeepSpeedEngine = None
    DeepSpeedZeroOptimizer = None
    DEEPSPEED_AVAILABLE = False


@dataclass
class DeepspeedConfig:
    """Configuration for Deepspeed training integration.
    
    This class provides comprehensive configuration options for Deepspeed training,
    including Zero optimization, gradient accumulation, mixed precision, and 
    memory optimization features.
    """
    
    # Core Deepspeed settings
    enable_deepspeed: bool = field(
        default=False,
        metadata={"help": "Enable/disable Deepspeed training. Defaults to False."}
    )
    config_file: str = field(
        default="",
        metadata={"help": "Path to custom Deepspeed configuration JSON file."}
    )
    
    # Zero optimization settings
    zero_stage: int = field(
        default=2,
        metadata={"help": "Zero optimization stage (0, 1, 2, or 3). Higher stages save more memory. Defaults to 2."}
    )
    
    # Batch and gradient settings
    gradient_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Gradient accumulation steps for Deepspeed."}
    )
    train_micro_batch_size_per_gpu: Optional[int] = field(
        default=None,
        metadata={"help": "Micro batch size per GPU."}
    )
    
    # Mixed precision settings
    enable_mixed_precision: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision training in Deepspeed. Defaults to True."}
    )
    fp16_enabled: bool = field(
        default=True,
        metadata={"help": "Enable FP16 mixed precision."}
    )
    bf16_enabled: bool = field(
        default=False,
        metadata={"help": "Enable BF16 mixed precision instead of FP16."}
    )
    loss_scale: float = field(
        default=0.0,
        metadata={"help": "Initial loss scale for mixed precision."}
    )
    min_loss_scale: float = field(
        default=1.0,
        metadata={"help": "Minimum loss scale for dynamic scaling."}
    )
    loss_scale_window: int = field(
        default=1000,
        metadata={"help": "Window size for dynamic loss scaling updates."}
    )
    
    # Memory optimization settings
    enable_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable CPU offloading for optimizer/parameters."}
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Pin memory for CPU offloading for faster transfers."}
    )
    overlap_comm: bool = field(
        default=True,
        metadata={"help": "Overlap communication and computation for better performance."}
    )
    reduce_bucket_size: int = field(
        default=200000000,
        metadata={"help": "Reduce bucket size for gradient communication."}
    )
    
    # Zero stage 3 specific settings
    stage3_prefetch_bucket_size: int = field(
        default=200000000,
        metadata={"help": "Prefetch bucket size for Zero stage 3."}
    )
    stage3_param_persistence_threshold: int = field(
        default=100000,
        metadata={"help": "Parameter persistence threshold for Zero stage 3."}
    )
    
    # Training settings
    enable_gradient_clipping: bool = field(
        default=True,
        metadata={"help": "Enable gradient clipping in Deepspeed."}
    )
    
    # Monitoring and debugging
    wall_clock_breakdown: bool = field(
        default=False,
        metadata={"help": "Enable detailed timing breakdown for performance analysis."}
    )
    tensorboard_enabled: bool = field(
        default=True,
        metadata={"help": "Enable Tensorboard logging through Deepspeed."}
    )
    tensorboard_output_path: str = field(
        default="",
        metadata={"help": "Output path for Tensorboard logs."}
    )
    
    # Checkpoint settings
    save_zero_checkpoint: bool = field(
        default=True,
        metadata={"help": "Save Zero-compatible checkpoints."}
    )
    load_universal_checkpoint: bool = field(
        default=False,
        metadata={"help": "Enable loading universal checkpoints across different Zero stages."}
    )
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Additional custom configuration parameters."}
    )


class DeepspeedManager:
    """Manager class for Deepspeed integration with the Trainer.
    
    This class handles all Deepspeed-related operations including:
    - Engine initialization and configuration
    - Model and optimizer wrapping
    - Training step execution
    - Checkpoint saving and loading
    - Memory optimization
    
    The manager integrates seamlessly with the existing Trainer infrastructure
    while providing advanced Deepspeed features for large-scale training.
    """
    
    def __init__(
        self,
        config: DeepspeedConfig,
        model: "TrainerModel",
        optimizer = None,
        trainer_config = None,
        output_path: Optional[str] = None,
    ):
        """Initialize the Deepspeed manager.
        
        Args:
            config: Deepspeed configuration settings
            model: The model to wrap with Deepspeed
            optimizer: Optional optimizer (can be None for auto-creation)
            trainer_config: Trainer configuration for parameter extraction
            output_path: Output path for logs and checkpoints
        """
        if not DEEPSPEED_AVAILABLE:
            raise ImportError(
                "Deepspeed is not available. Please install deepspeed: `pip install deepspeed`"
            )
        
        self.config = config
        self.model = model
        self.original_optimizer = optimizer
        self.trainer_config = trainer_config
        self.output_path = output_path or "."
        
        # Deepspeed engine and related objects
        self.engine = None
        self.is_initialized = False
        
        # Store original training state
        self._original_train_mode = None
        
    def is_available(self) -> bool:
        """Check if Deepspeed is available and enabled."""
        return DEEPSPEED_AVAILABLE and self.config.enable_deepspeed
    
    def should_use_deepspeed(self) -> bool:
        """Check if Deepspeed should be used for training."""
        return self.is_available() and self.config.enable_deepspeed
    
    def _create_deepspeed_config_dict(self) -> Dict[str, Any]:
        """Create Deepspeed configuration dictionary."""
        ds_config = {
            "train_batch_size": getattr(self.trainer_config, 'batch_size', 8),
            "train_micro_batch_size_per_gpu": (
                self.config.train_micro_batch_size_per_gpu or 
                getattr(self.trainer_config, 'batch_size', 8)
            ),
            "gradient_accumulation_steps": (
                self.config.gradient_accumulation_steps or 
                getattr(self.trainer_config, 'grad_accum_steps', 1)
            ),
        }
        
        # Add Zero optimization configuration
        if self.config.zero_stage > 0:
            ds_config["zero_optimization"] = {
                "stage": self.config.zero_stage,
                "overlap_comm": self.config.overlap_comm,
                "reduce_bucket_size": self.config.reduce_bucket_size,
            }
            
            # Add CPU offloading if enabled
            if self.config.enable_cpu_offload and self.config.zero_stage >= 2:
                ds_config["zero_optimization"]["cpu_offload"] = True
                
            # Add stage 3 specific settings
            if self.config.zero_stage == 3:
                ds_config["zero_optimization"].update({
                    "stage3_prefetch_bucket_size": self.config.stage3_prefetch_bucket_size,
                    "stage3_param_persistence_threshold": self.config.stage3_param_persistence_threshold,
                })
        
        # Add mixed precision configuration
        if self.config.enable_mixed_precision:
            if self.config.bf16_enabled:
                ds_config["bf16"] = {"enabled": True}
            else:
                ds_config["fp16"] = {
                    "enabled": self.config.fp16_enabled,
                    "loss_scale": self.config.loss_scale,
                    "min_loss_scale": self.config.min_loss_scale,
                    "loss_scale_window": self.config.loss_scale_window,
                }
        
        # Add optimizer settings if available
        if hasattr(self.trainer_config, 'lr'):
            ds_config["optimizer"] = {
                "type": getattr(self.trainer_config, 'optimizer', 'Adam'),
                "params": {
                    "lr": getattr(self.trainer_config, 'lr', 0.001),
                }
            }
        
        # Add gradient clipping
        if self.config.enable_gradient_clipping and hasattr(self.trainer_config, 'grad_clip'):
            grad_clip = getattr(self.trainer_config, 'grad_clip', 0.0)
            if grad_clip > 0:
                ds_config["gradient_clipping"] = grad_clip
        
        # Add wall clock breakdown for debugging
        if self.config.wall_clock_breakdown:
            ds_config["wall_clock_breakdown"] = True
            
        # Merge custom configuration
        if self.config.custom_config:
            ds_config.update(self.config.custom_config)
            
        return ds_config
    
    def initialize_engine(
        self, 
        model, 
        optimizer=None, 
        lr_scheduler=None, 
        training_data=None
    ) -> "DeepSpeedEngine":
        """Initialize the Deepspeed engine.
        
        Args:
            model: The model to wrap with Deepspeed
            optimizer: Optional optimizer
            lr_scheduler: Optional learning rate scheduler  
            training_data: Optional training data loader
            
        Returns:
            The initialized Deepspeed engine
        """
        if not self.should_use_deepspeed():
            raise RuntimeError("Deepspeed is not available or not enabled.")
            
        # Import torch distributed for initialization
        import torch.distributed as dist
        
        # Initialize distributed backend if not already initialized
        if not dist.is_initialized():
            try:
                # Initialize single-process distributed environment
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", "12355")
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("LOCAL_RANK", "0") 
                os.environ.setdefault("WORLD_SIZE", "1")
                
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    rank=0,
                    world_size=1
                )
            except Exception as e:
                print(f"Failed to initialize distributed backend: {e}")
                print("Attempting single-process Deepspeed initialization...")
        
        # Create Deepspeed configuration  
        ds_config = self._create_deepspeed_config_dict()
        
        try:
            # Initialize Deepspeed engine
            self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=ds_config,
            )
            
            # Ensure model is on correct device after Deepspeed initialization
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Deepspeed should handle device placement, but verify
                if hasattr(self.engine, 'module'):
                    model_device = next(self.engine.module.parameters()).device
                    print(f" > Deepspeed model device: {model_device}")
                    
        except Exception as e:
            # Fallback: try with minimal configuration for single GPU
            print(f"Initial Deepspeed initialization failed: {e}")
            print("Trying fallback configuration for single GPU...")
            
            minimal_config = {
                "train_batch_size": getattr(self.trainer_config, 'batch_size', 8),
                "train_micro_batch_size_per_gpu": getattr(self.trainer_config, 'batch_size', 8),
                "gradient_accumulation_steps": 1,
                "zero_optimization": {
                    "stage": 0,  # Disable Zero for single GPU
                },
                "fp16": {
                    "enabled": False,  # Disable mixed precision for testing
                }
            }
            
            self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=minimal_config,
            )
        
        self.is_initialized = True
        return self.engine
    
    def get_model(self):
        """Get the wrapped model from Deepspeed engine."""
        if self.engine is not None:
            return self.engine.module
        return self.model
    
    def get_optimizer(self):
        """Get the optimizer from Deepspeed engine."""
        if self.engine is not None:
            return self.engine.optimizer
        return self.original_optimizer
    
    def get_lr_scheduler(self):
        """Get the learning rate scheduler from Deepspeed engine."""
        if hasattr(self, 'lr_scheduler'):
            return self.lr_scheduler
        return None
    
    def backward(self, loss):
        """Perform backward pass through Deepspeed engine."""
        if self.engine is not None:
            self.engine.backward(loss)
        else:
            loss.backward()
    
    def step(self):
        """Perform optimizer step through Deepspeed engine."""
        if self.engine is not None:
            self.engine.step()
        elif self.original_optimizer is not None:
            self.original_optimizer.step()
    
    def is_gradient_accumulation_boundary(self) -> bool:
        """Check if current step is a gradient accumulation boundary.
        
        Returns True when it's time to perform optimizer step after gradient accumulation.
        In Deepspeed, this is handled internally by the engine.
        
        Returns:
            True if optimizer step should be performed
        """
        if self.engine is not None:
            return self.engine.is_gradient_accumulation_boundary()
        # Fallback: always step if not using Deepspeed engine
        return True
    
    def save_checkpoint(self, save_dir: str, step: int = 0, tag: str = None):
        """Save Deepspeed checkpoint.
        
        Args:
            save_dir: Directory to save checkpoint
            step: Current training step (used if tag is not provided)
            tag: Optional tag for checkpoint naming (if provided, overrides step-based naming)
        """
        if self.engine is not None and self.config.save_zero_checkpoint:
            if tag is not None:
                checkpoint_dir = os.path.join(save_dir, f"deepspeed_checkpoint_{tag}")
            else:
                checkpoint_dir = os.path.join(save_dir, f"deepspeed_checkpoint_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.engine.save_checkpoint(checkpoint_dir)
            return checkpoint_dir
        return None
    
    def load_checkpoint(self, load_dir: str):
        """Load Deepspeed checkpoint.
        
        Args:
            load_dir: Directory containing checkpoint
            
        Returns:
            Tuple of (checkpoint_path, client_state)
        """
        if self.engine is not None:
            return self.engine.load_checkpoint(load_dir)
        return None, None


def create_deepspeed_config(
    enable_deepspeed: bool = True,
    zero_stage: int = 2,
    enable_mixed_precision: bool = True,
    enable_cpu_offload: bool = False,
    gradient_accumulation_steps: Optional[int] = None,
    train_micro_batch_size_per_gpu: Optional[int] = None,
    **kwargs
) -> DeepspeedConfig:
    """Create a Deepspeed configuration with common settings.
    
    Args:
        enable_deepspeed: Enable Deepspeed training
        zero_stage: Zero optimization stage (0, 1, 2, or 3)
        enable_mixed_precision: Enable mixed precision training
        enable_cpu_offload: Enable CPU offloading for memory optimization
        gradient_accumulation_steps: Number of gradient accumulation steps
        train_micro_batch_size_per_gpu: Micro batch size per GPU
        **kwargs: Additional configuration parameters
        
    Returns:
        DeepspeedConfig object with the specified settings
    """
    return DeepspeedConfig(
        enable_deepspeed=enable_deepspeed,
        zero_stage=zero_stage,
        enable_mixed_precision=enable_mixed_precision,
        enable_cpu_offload=enable_cpu_offload,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
        **kwargs
    )


def is_deepspeed_available() -> bool:
    """Check if Deepspeed is available.
    
    Returns:
        True if Deepspeed is installed and available
    """
    return DEEPSPEED_AVAILABLE
