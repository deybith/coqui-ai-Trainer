from dataclasses import dataclass, field
from typing import Any

from coqpit import Coqpit


@dataclass
class TrainerArgs(Coqpit):
    """Trainer arguments that can be accessed from the command line.

    Examples::
        >>> python train.py --restore_path /path/to/checkpoint.pth
    """

    continue_path: str = field(
        default="",
        metadata={
            "help": "Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder."
        },
    )
    restore_path: str = field(
        default="",
        metadata={
            "help": "Path to a model checkpoit. Restore the model with the given checkpoint and start a new training."
        },
    )
    best_path: str = field(
        default="",
        metadata={
            "help": "Best model file to be used for extracting the best loss. If not specified, the latest best model in continue path is used"
        },
    )
    use_ddp: bool = field(
        default=False,
        metadata={"help": "Use DDP in distributed training. It is to set in `distribute.py`. Do not set manually."},
    )
    use_accelerate: bool = field(default=False, metadata={"help": "Use HF Accelerate as the back end for training."})
    grad_accum_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps. It is used to accumulate gradients over multiple batches."
        },
    )
    overfit_batch: bool = field(default=False, metadata={"help": "Overfit a single batch for debugging. When enabled, the trainer will repeatedly train on the first batch in the dataset instead of iterating through all batches. This is useful for debugging model architecture, loss functions, and training loops."})
    skip_train_epoch: bool = field(
        default=False,
        metadata={"help": "Skip training and only run evaluation and test."},
    )
    start_with_eval: bool = field(
        default=False,
        metadata={"help": "Start with evaluation and test."},
    )
    small_run: int | None = field(
        default=None,
        metadata={
            "help": "Only use a subset of the samples for debugging. Set the number of samples to use. Defaults to None. "
        },
    )
    gpu: int | None = field(
        default=None, metadata={"help": "GPU ID to use if ```CUDA_VISIBLE_DEVICES``` is not set. Defaults to None."}
    )
    # only for DDP
    rank: int = field(default=0, metadata={"help": "Process rank in a distributed training. Don't set manually."})
    group_id: str = field(
        default="", metadata={"help": "Process group id in a distributed training. Don't set manually."}
    )


@dataclass
class TrainerConfig(Coqpit):
    """Config fields tweaking the Trainer for a model.

    A ````ModelConfig```, by inheriting ```TrainerConfig``` must be defined for using 👟.
    Inherit this by a new model config and override the fields as needed.
    All the fields can be overridden from comman-line as ```--coqpit.arg_name=value```.

    Example::

        Run the training code by overriding the ```lr``` and ```plot_step``` fields.

        >>> python train.py --coqpit.plot_step=22 --coqpit.lr=0.001

        Defining a model using ```TrainerConfig```.

        >>> from trainer import TrainerConfig
        >>> class MyModelConfig(TrainerConfig):
        ...     optimizer: str = "Adam"
        ...     lr: float = 0.001
        ...     epochs: int = 1
        ...     ...
        >>> class MyModel(nn.module):
        ...    def __init__(self, config):
        ...        ...
        >>> model = MyModel(MyModelConfig())

    """

    # Fields for the run
    output_path: str = field(default="output")
    logger_uri: str | None = field(
        default=None,
        metadata={
            "help": "URI to save training artifacts by the logger. If not set, logs will be saved in the output_path. Defaults to None"
        },
    )
    run_name: str = field(default="run", metadata={"help": "Name of the run. Defaults to 'run'"})
    project_name: str | None = field(default=None, metadata={"help": "Name of the project. Defaults to None"})
    run_description: str = field(
        default="🐸Coqui trainer run.",
        metadata={"help": "Notes and description about the run. Defaults to '🐸Coqui trainer run.'"},
    )
    # Fields for logging
    print_step: int = field(
        default=25, metadata={"help": "Print training stats on the terminal every print_step steps. Defaults to 25"}
    )
    plot_step: int = field(
        default=100, metadata={"help": "Plot training stats on the logger every plot_step steps. Defaults to 100"}
    )
    model_param_stats: bool = field(
        default=False, metadata={"help": "Log model parameters stats on the logger dashboard. Defaults to False"}
    )
    wandb_entity: str | None = field(default=None, metadata={"help": "Wandb entity to log the run. Defaults to None"})
    dashboard_logger: str = field(
        default="tensorboard", metadata={"help": "Logger to use for the tracking dashboard. Defaults to 'tensorboard'"}
    )
    # Fields for checkpointing
    save_on_interrupt: bool = field(
        default=True, metadata={"help": "Save checkpoint on interrupt (Ctrl+C). Defaults to True"}
    )
    log_model_step: int | None = field(
        default=None,
        metadata={
            "help": "Save checkpoint to the logger every log_model_step steps. If not defined `save_step == log_model_step`."
        },
    )
    save_step: int = field(
        default=10000, metadata={"help": "Save local checkpoint every save_step steps. Defaults to 10000"}
    )
    save_n_checkpoints: int = field(default=5, metadata={"help": "Keep n local checkpoints. Defaults to 5"})
    save_checkpoints: bool = field(default=True, metadata={"help": "Save checkpoints locally. Defaults to True"})
    save_all_best: bool = field(
        default=False, metadata={"help": "Save all best checkpoints and keep the older ones. Defaults to False"}
    )
    save_best_after: int = field(default=0, metadata={"help": "Wait N steps to save best checkpoints. Defaults to 0"})
    target_loss: str | None = field(
        default=None, metadata={"help": "Target loss name to select the best model. Defaults to None"}
    )
    # Fields for eval and test run
    print_eval: bool = field(default=False, metadata={"help": "Print eval steps on the terminal. Defaults to False"})
    test_delay_epochs: int = field(default=0, metadata={"help": "Wait N epochs before running the test. Defaults to 0"})
    run_eval: bool = field(
        default=True, metadata={"help": "Run evalulation epoch after training epoch. Defaults to True"}
    )
    run_eval_steps: int | None = field(
        default=None,
        metadata={
            "help": "Run evalulation epoch after N steps. If None, waits until training epoch is completed. Defaults to None"
        },
    )
    # Fields for distributed training
    distributed_backend: str = field(
        default="nccl", metadata={"help": "Distributed backend to use. Defaults to 'nccl'"}
    )
    distributed_url: str = field(
        default="tcp://localhost:54321",
        metadata={"help": "Distributed url to use. Defaults to 'tcp://localhost:54321'"},
    )
    # Fields for TPU training
    use_tpu: bool = field(
        default=False, metadata={"help": "Use TPU for training. Requires torch_xla to be installed. Defaults to False"}
    )
    tpu_cores: int = field(
        default=8, metadata={"help": "Number of TPU cores to use. Defaults to 8"}
    )
    tpu_metrics_debug: bool = field(
        default=False, metadata={"help": "Enable TPU metrics debugging. Defaults to False"}
    )
    tpu_profiler: bool = field(
        default=False, metadata={"help": "Enable TPU profiler. Defaults to False"}
    )
    tpu_profiler_steps: int = field(
        default=100, metadata={"help": "Number of steps to profile on TPU. Defaults to 100"}
    )
    # Fields for training specs
    mixed_precision: bool = field(default=False, metadata={"help": "Use mixed precision training. Defaults to False"})
    precision: str = field(
        default="fp16",
        metadata={
            "help": "Precision to use in mixed precision training. `fp16` for float16 and `bf16` for bfloat16. Defaults to 'f16'"
        },
    )
    epochs: int = field(default=1000, metadata={"help": "Number of epochs to train. Defaults to 1000"})
    batch_size: int = field(default=32, metadata={"help": "Batch size to use. Defaults to 32"})
    eval_batch_size: int = field(default=16, metadata={"help": "Batch size to use for eval. Defaults to 16"})
    grad_clip: float | list[float] = field(
        default=0.0,
        metadata={"help": "Gradient clipping value (for each optimizer if a list). Disabled if <= 0. Defaults to 0.0"},
    )
    scheduler_after_epoch: bool = field(
        default=True,
        metadata={"help": "Step the scheduler after each epoch else step after each iteration. Defaults to True"},
    )
    # Fields for optimzation
    lr: float | list[float] = field(
        default=0.001, metadata={"help": "Learning rate for each optimizer. Defaults to 0.001"}
    )
    optimizer: str | list[str] | None = field(default=None, metadata={"help": "Optimizer(s) to use. Defaults to None"})
    optimizer_params: dict[str, Any] | list[dict[str, Any]] = field(
        default_factory=dict, metadata={"help": "Optimizer(s) arguments. Defaults to {}"}
    )
    lr_scheduler: str | list[str] | None = field(
        default=None, metadata={"help": "Learning rate scheduler(s) to use. Defaults to None"}
    )
    lr_scheduler_params: dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Learning rate scheduler(s) arguments. Defaults to {}"}
    )
    use_grad_scaler: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable gradient scaler explicitly. It is enabled by default with AMP training. Defaults to False"
        },
    )
    allow_tf32: bool = field(
        default=False,
        metadata={
            "help": "A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs. Default to False."
        },
    )
    cudnn_enable: bool = field(default=True, metadata={"help": "Enable/disable cudnn explicitly. Defaults to True"})
    cudnn_deterministic: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable deterministic cudnn operations. Set this True for reproducibility but it slows down training significantly.  Defaults to False."
        },
    )
    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable cudnn benchmark explicitly. Set this False if your input size change constantly. Defaults to False"
        },
    )
    training_seed: int = field(
        default=54321,
        metadata={"help": "Global seed for torch, random and numpy random number generator. Defaults to 54321"},
    )
    # Deepspeed integration fields
    use_deepspeed: bool = field(
        default=False,
        metadata={"help": "Enable Deepspeed integration for large-scale training optimization. Defaults to False"}
    )
    deepspeed_config_file: str = field(
        default="",
        metadata={
            "help": "Path to custom Deepspeed configuration JSON file. If empty, auto-generates config. Defaults to ''"
        }
    )
    deepspeed_zero_stage: int = field(
        default=2,
        metadata={
            "help": "Deepspeed Zero optimization stage (0, 1, 2, or 3). Higher stages save more memory. Defaults to 2"
        }
    )
    deepspeed_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Enable Deepspeed CPU offloading for optimizer/parameters to save GPU memory. Defaults to False"
        }
    )


@dataclass
class BaseTrainingConfig(TrainerConfig):
    """Base configuration class for training that extends TrainerConfig with commonly used fields.
    
    This class provides a foundation for model-specific training configurations by adding
    common parameters that are frequently used across different model types, such as
    data loading, model identification, and training utilities.
    
    Inherit from this class to create model-specific configurations while maintaining
    consistent training parameter patterns across different models.
    
    Args:
        model (str):
            Name/identifier of the model being trained. This can be used for logging,
            model registry, or conditional logic. Defaults to None.
            
        num_loader_workers (int):
            Number of worker processes for training data loading. Set to 0 to disable
            multiprocessing. Higher values can improve training speed but consume more
            memory. Defaults to 0.
            
        num_eval_loader_workers (int):
            Number of worker processes for evaluation data loading. Can be different
            from training workers to optimize evaluation performance. Defaults to 0.
            
        data_path (str):
            Path to the root directory containing training data. This can be used
            as a base path for dataset loading. Defaults to "".
            
        use_data_cache (bool):
            Enable/disable caching of preprocessed data to speed up subsequent
            training runs. Defaults to False.
            
        cache_path (str):
            Directory path where cached data should be stored. If empty, defaults
            to a cache directory within the output path. Defaults to "".
            
        max_seq_len (int):
            Maximum sequence length for input data. Used for truncation or padding
            operations during data preprocessing. Defaults to None.
            
        min_seq_len (int):
            Minimum sequence length for input data. Sequences shorter than this
            may be filtered out or padded. Defaults to None.
            
        use_data_augmentation (bool):
            Enable/disable data augmentation techniques during training.
            Defaults to False.
            
        eval_split_size (float):
            Fraction of data to use for evaluation when automatic train/eval
            splitting is performed. Should be between 0.0 and 1.0. Defaults to 0.1.
            
        test_split_size (float):
            Fraction of data to use for testing when automatic train/test
            splitting is performed. Should be between 0.0 and 1.0. Defaults to 0.1.
            
        use_weighted_sampling (bool):
            Enable/disable weighted sampling for imbalanced datasets.
            Defaults to False.
            
        compute_linear_spec (bool):
            Enable/disable computation of linear spectrograms for audio models.
            Defaults to False.
            
        compute_mel_spec (bool):
            Enable/disable computation of mel-scale spectrograms for audio models.
            Defaults to True.
    
    Example:
        >>> from trainer.config import BaseTrainingConfig
        >>> class MyModelConfig(BaseTrainingConfig):
        ...     model: str = "my_custom_model"
        ...     num_loader_workers: int = 4
        ...     lr: float = 0.001
        ...     # Add model-specific parameters here
        >>> config = MyModelConfig()
        >>> trainer = Trainer(args, config, ...)
    """
    
    # Model identification
    model: str = field(
        default=None,
        metadata={"help": "Name/identifier of the model being trained. Used for logging and model registry."}
    )
    
    # Data loading configuration
    num_loader_workers: int = field(
        default=0,
        metadata={
            "help": "Number of worker processes for training data loading. Set to 0 to disable multiprocessing. Defaults to 0."
        }
    )
    num_eval_loader_workers: int = field(
        default=0,
        metadata={
            "help": "Number of worker processes for evaluation data loading. Defaults to 0."
        }
    )
    data_path: str = field(
        default="",
        metadata={"help": "Path to the root directory containing training data. Defaults to ''."}
    )
    
    # Data caching
    use_data_cache: bool = field(
        default=False,
        metadata={"help": "Enable/disable caching of preprocessed data to speed up training. Defaults to False."}
    )
    cache_path: str = field(
        default="",
        metadata={
            "help": "Directory path where cached data should be stored. If empty, uses output_path/cache. Defaults to ''."
        }
    )
    
    # Sequence configuration
    max_seq_len: int | None = field(
        default=None,
        metadata={
            "help": "Maximum sequence length for input data. Used for truncation/padding. Defaults to None."
        }
    )
    min_seq_len: int | None = field(
        default=None,
        metadata={
            "help": "Minimum sequence length for input data. Shorter sequences may be filtered. Defaults to None."
        }
    )
    
    # Data augmentation
    use_data_augmentation: bool = field(
        default=False,
        metadata={"help": "Enable/disable data augmentation techniques during training. Defaults to False."}
    )
    
    # Data splitting
    eval_split_size: float = field(
        default=0.1,
        metadata={
            "help": "Fraction of data for evaluation when automatic splitting is used. Range: 0.0-1.0. Defaults to 0.1."
        }
    )
    test_split_size: float = field(
        default=0.1,
        metadata={
            "help": "Fraction of data for testing when automatic splitting is used. Range: 0.0-1.0. Defaults to 0.1."
        }
    )
    
    # Sampling strategy
    use_weighted_sampling: bool = field(
        default=False,
        metadata={"help": "Enable/disable weighted sampling for imbalanced datasets. Defaults to False."}
    )
    
    # Audio/spectral processing (commonly used across audio models)
    compute_linear_spec: bool = field(
        default=False,
        metadata={"help": "Enable/disable computation of linear spectrograms for audio models. Defaults to False."}
    )
    compute_mel_spec: bool = field(
        default=True,
        metadata={"help": "Enable/disable computation of mel-scale spectrograms for audio models. Defaults to True."}
    )
