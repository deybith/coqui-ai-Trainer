import importlib.metadata

from .config import TrainerArgs, TrainerConfig, BaseTrainingConfig
from .checkpoint_manager import CheckpointManager, CheckpointConfig, CheckpointMetadata
from .swa_utils import SWAManager, SWAConfig, create_swa_config
from .model import TrainerModel
from .trainer import Trainer

# Import Deepspeed utilities with error handling
try:
    from .deepspeed_utils import DeepspeedConfig, DeepspeedManager, create_deepspeed_config, is_deepspeed_available
    DEEPSPEED_IMPORTS_AVAILABLE = True
except ImportError as e:
    # Create dummy classes if deepspeed utilities can't be imported
    class DeepspeedConfig:
        """Dummy Deepspeed config when deepspeed is not available."""
        def __init__(self, **kwargs):
            pass
    
    class DeepspeedManager:
        """Dummy Deepspeed manager when deepspeed is not available."""
        def __init__(self, **kwargs):
            pass
    
    def create_deepspeed_config(**kwargs):
        """Dummy function when deepspeed is not available."""
        return DeepspeedConfig(**kwargs)
    
    def is_deepspeed_available():
        """Always return False when deepspeed utilities can't be imported."""
        return False
    
    DEEPSPEED_IMPORTS_AVAILABLE = False

__version__ = importlib.metadata.version("coqui-tts-trainer")

__all__ = [
    "Trainer", 
    "TrainerArgs", 
    "TrainerConfig", 
    "BaseTrainingConfig", 
    "TrainerModel",
    "CheckpointManager",
    "CheckpointConfig", 
    "CheckpointMetadata",
    "SWAManager",
    "SWAConfig",
    "create_swa_config",
    "DeepspeedConfig",
    "DeepspeedManager", 
    "create_deepspeed_config",
    "is_deepspeed_available"
]
