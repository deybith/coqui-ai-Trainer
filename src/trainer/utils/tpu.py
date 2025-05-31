"""TPU training utilities for torch_xla integration."""

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger("trainer")

# Global flag to check if torch_xla is available
_XLA_AVAILABLE = None


def is_xla_available() -> bool:
    """Check if torch_xla is available for TPU training."""
    global _XLA_AVAILABLE
    if _XLA_AVAILABLE is None:
        try:
            import torch_xla  # pylint: disable=import-outside-toplevel, unused-import
            _XLA_AVAILABLE = True
        except ImportError:
            _XLA_AVAILABLE = False
    return _XLA_AVAILABLE


def is_tpu_available() -> bool:
    """Check if TPU devices are available."""
    if not is_xla_available():
        return False
    
    try:
        import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
        return xm.xla_device_count() > 0
    except Exception:
        return False


def get_tpu_device(local_rank: Optional[int] = None) -> torch.device:
    """Get TPU device.
    
    Args:
        local_rank: Local rank for multi-TPU training. If None, uses ordinal 0.
        
    Returns:
        torch.device: TPU device
        
    Raises:
        RuntimeError: If torch_xla is not available or no TPUs found
    """
    if not is_xla_available():
        raise RuntimeError("torch_xla is required for TPU training. Install with: pip install torch_xla")
    
    if not is_tpu_available():
        raise RuntimeError("No TPU devices found. Make sure you're running on a TPU-enabled environment.")
    
    import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
    
    if local_rank is not None:
        device = xm.xla_device(local_rank)
    else:
        device = xm.xla_device()
    
    logger.info(f"Using TPU device: {device}")
    return device


def get_tpu_world_size() -> int:
    """Get the number of TPU cores available."""
    if not is_xla_available():
        return 0
    
    try:
        import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
        return xm.xla_device_count()
    except Exception:
        return 0


def get_tpu_local_rank() -> int:
    """Get local rank for TPU training."""
    if not is_xla_available():
        return 0
    
    try:
        import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
        return xm.get_local_ordinal()
    except Exception:
        return 0


def get_tpu_global_rank() -> int:
    """Get global rank for TPU training."""
    if not is_xla_available():
        return 0
    
    try:
        import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
        return xm.get_ordinal()
    except Exception:
        return 0


def setup_tpu_training_env(training_seed: int = 54321) -> tuple[torch.device, int]:
    """Setup TPU environment for training.
    
    Args:
        training_seed: Seed for random number generators
        
    Returns:
        tuple[torch.device, int]: TPU device and world size
        
    Raises:
        RuntimeError: If TPU setup fails
    """
    if not is_tpu_available():
        raise RuntimeError("TPU training requested but no TPU devices available")
    
    import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
    import torch_xla.debug.profiler as xp  # pylint: disable=import-outside-toplevel
    
    # Set up random seeds for reproducibility
    torch.manual_seed(training_seed)
    
    # Get TPU device
    device = get_tpu_device()
    world_size = get_tpu_world_size()
    local_rank = get_tpu_local_rank()
    global_rank = get_tpu_global_rank()
    
    logger.info(f"TPU training setup:")
    logger.info(f"  Device: {device}")
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Local rank: {local_rank}")
    logger.info(f"  Global rank: {global_rank}")
    
    # Enable XLA compilation debug if requested
    if os.environ.get("XLA_USE_BF16") == "1":
        logger.info("  BFloat16 precision enabled")
    
    if os.environ.get("XLA_DUMP_HLO") == "1":
        logger.info("  XLA HLO dumping enabled")
    
    return device, world_size


def mark_step():
    """Mark a step for XLA compilation."""
    if is_xla_available():
        import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
        xm.mark_step()


def wait_for_tpu():
    """Wait for all TPU operations to complete."""
    if is_xla_available():
        import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
        xm.wait_device_ops()


def rendezvous(tag: str = "init") -> None:
    """Synchronize all TPU cores."""
    if is_xla_available():
        import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
        xm.rendezvous(tag)


def all_reduce(tensor: torch.Tensor, reduce_type: str = "sum") -> torch.Tensor:
    """Perform all-reduce operation across TPU cores.
    
    Args:
        tensor: Tensor to reduce
        reduce_type: Reduction type ('sum', 'mean', 'max', 'min')
        
    Returns:
        torch.Tensor: Reduced tensor
    """
    if not is_xla_available():
        return tensor
    
    import torch_xla.core.xla_model as xm  # pylint: disable=import-outside-toplevel
    
    if reduce_type == "sum":
        return xm.all_reduce(xm.REDUCE_SUM, tensor)
    elif reduce_type == "mean":
        result = xm.all_reduce(xm.REDUCE_SUM, tensor)
        return result / get_tpu_world_size()
    elif reduce_type == "max":
        return xm.all_reduce(xm.REDUCE_MAX, tensor)
    elif reduce_type == "min":
        return xm.all_reduce(xm.REDUCE_MIN, tensor)
    else:
        raise ValueError(f"Unsupported reduce_type: {reduce_type}")


def is_main_process() -> bool:
    """Check if this is the main TPU process."""
    return get_tpu_global_rank() == 0


def print_tpu_memory_info():
    """Print TPU memory information."""
    if not is_xla_available():
        return
    
    try:
        import torch_xla.debug.metrics as met  # pylint: disable=import-outside-toplevel
        logger.info("TPU Memory Metrics:")
        logger.info(met.short_metrics_report())
    except Exception as e:
        logger.warning(f"Could not retrieve TPU memory info: {e}")


def cleanup_tpu():
    """Cleanup TPU resources."""
    if is_xla_available():
        wait_for_tpu()
        logger.info("TPU cleanup completed")


def save_model_on_tpu(model: torch.nn.Module, save_path: str, master_only: bool = True):
    """Save model on TPU with proper synchronization.
    
    Args:
        model: Model to save
        save_path: Path to save the model
        master_only: Only save on master process if True
    """
    if master_only and not is_main_process():
        return
    
    # Ensure all operations are complete before saving
    wait_for_tpu()
    
    # Move model to CPU for saving to avoid TPU-specific tensors
    cpu_model = model.cpu()
    torch.save(cpu_model.state_dict(), save_path)
    
    # Move model back to TPU
    if is_tpu_available():
        device = get_tpu_device()
        model.to(device)
    
    logger.info(f"Model saved to {save_path}")


def get_tpu_metrics() -> dict[str, Any]:
    """Get TPU performance metrics.
    
    Returns:
        dict: TPU metrics dictionary
    """
    if not is_xla_available():
        return {}
    
    try:
        import torch_xla.debug.metrics as met  # pylint: disable=import-outside-toplevel
        return met.metrics_report()
    except Exception as e:
        logger.warning(f"Could not retrieve TPU metrics: {e}")
        return {}


def optimize_for_tpu(model: torch.nn.Module) -> torch.nn.Module:
    """Apply TPU-specific optimizations to model.
    
    Args:
        model: Model to optimize
        
    Returns:
        torch.nn.Module: Optimized model
    """
    # Apply common TPU optimizations
    if hasattr(model, 'half'):
        # Use bfloat16 if available (better for TPU than float16)
        if torch.cuda.is_bf16_supported():
            model = model.bfloat16()
            logger.info("Model converted to bfloat16 for TPU optimization")
    
    return model


# Environment variable helpers
def set_tpu_env_vars():
    """Set recommended TPU environment variables."""
    env_vars = {
        "TPU_NUM_DEVICES": "8",  # Standard TPU v2/v3 configuration
        "XLA_USE_BF16": "1",     # Use bfloat16 for better TPU performance
        "XLA_TENSOR_ALLOCATOR_MAXSIZE": "100000000",  # Memory management
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")


def get_tpu_profiler(log_dir: str = "./tpu_profiler"):
    """Get TPU profiler for performance analysis.
    
    Args:
        log_dir: Directory to save profiler logs
        
    Returns:
        TPU profiler context manager or None if not available
    """
    if not is_xla_available():
        return None
    
    try:
        import torch_xla.debug.profiler as xp  # pylint: disable=import-outside-toplevel
        os.makedirs(log_dir, exist_ok=True)
        return xp.profile(log_dir)
    except Exception as e:
        logger.warning(f"Could not create TPU profiler: {e}")
        return None
