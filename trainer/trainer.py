import functools
import logging
import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data import DataLoader

from trainer._types import Callback, LossDict, LRScheduler
from trainer.callbacks import TrainerCallback
from trainer.config import TrainerArgs, TrainerConfig
from trainer.checkpoint_manager import CheckpointManager, CheckpointConfig
from trainer.generic_utils import (
    KeepAverage,
    count_parameters,
    get_experiment_folder_path,
    is_pytorch_at_least_2_3,
    is_pytorch_at_least_2_4,
)
from trainer.io import (
    copy_model_files,
)
from trainer.logging import ConsoleLogger
from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.model import TrainerModel
from trainer.utils.distributed import (
    init_distributed,
    rank_zero_logger_info,
)
from trainer.swa_utils import SWAManager, SWAConfig
from trainer.deepspeed_utils import DeepspeedManager, DeepspeedConfig, create_deepspeed_config, is_deepspeed_available

from trainer.core.base import Base
from trainer.core.data_loading import DataLoading
from trainer.core.fit_functions import FitFunctions
from trainer.core.testing import Testing
from trainer.core.eval_functions import EvalFunctions
from trainer.core.static_methods import StaticMethods
from trainer.core.helper_functions import HelperFunctions

# TPU support - import conditionally
try:
    from trainer.utils.tpu import (
        is_tpu_available, 
        setup_tpu_training_env, 
        get_tpu_device,
        mark_step,
        wait_for_tpu,
        all_reduce,
        get_tpu_world_size,
        save_model_on_tpu,
        print_tpu_memory_info
    )
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

logger = logging.getLogger("trainer")

if is_pytorch_at_least_2_3():
    GradScaler = functools.partial(torch.GradScaler, device="cuda")
else:
    GradScaler = torch.cuda.amp.GradScaler  # type: ignore[assignment]


class Trainer(Base, DataLoading, FitFunctions, Testing, EvalFunctions, StaticMethods, HelperFunctions):
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        args: TrainerArgs,
        config: TrainerConfig,
        output_path: str | os.PathLike[Any] | None = None,
        *,
        c_logger: ConsoleLogger | None = None,
        dashboard_logger: BaseDashboardLogger | None = None,
        model: TrainerModel | None = None,
        get_model: Callable[..., TrainerModel] | None = None,
        get_data_samples: Callable[..., list[Any]] | None = None,
        train_samples: list[Any] | None = None,
        eval_samples: list[Any] | None = None,
        test_samples: list[Any] | None = None,
        train_loader: DataLoader[Any] | None = None,
        eval_loader: DataLoader[Any] | None = None,
        training_assets: dict[str, Any] | None = None,
        parse_command_line_args: bool = True,
        callbacks: dict[str, Callback] | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        swa_config: SWAConfig | None = None,
        deepspeed_config = None,  # DeepspeedConfig | None
        gpu: int | None = None,
    ) -> None:
        """Simple yet powerful 🐸💬 TTS trainer for PyTorch.

        It can train all the available `tts` and `vocoder` models or easily be customized.

        Notes:
            Supports Automatic Mixed Precision training using PyTorch's native `amp` module.

        Args:
            args (TrainerArgs): Training arguments parsed either from console by `argparse` or `TrainerArgs`
                config object.

            config (TrainerConfig): Model config object. It includes all the values necessary for initializing, training, evaluating
                and testing the model.

            output_path (str or Path, optional): Path to the output training folder. All
                the files are saved under this path. Uses value from config if None.

            c_logger (ConsoleLogger, optional): Console logger for printing training status. If not provided, the default
                console logger is used. Defaults to None.

            dashboard_logger Union[TensorboardLogger, WandbLogger]: Dashboard logger. If not provided, the tensorboard logger is used.
                Defaults to None.

            model (TrainerModel, optional): Initialized and ready-to-train model. If it is not defined, `Trainer`
                initializes a model from the provided config. Defaults to None.

            get_model (Callable):
                A function that returns a model. It is used to initialize the model when `model` is not provided.
                It either takes the config as the only argument or does not take any argument.
                Defaults to None

            get_data_samples (Callable):
                A function that returns a list of training and evaluation samples. Used if `train_samples` and
                `eval_samples` are None. Defaults to None.

            train_samples (List):
                A list of training samples used by the model's `get_train_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            eval_samples (List):
                A list of evaluation samples used by the model's `get_eval_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            train_loader (DataLoader):
                A pytorch data loader object for training epochs. Leave as None if you want it to be made during training. Defaults to None.

            eval_loader (DataLoader):
                A pytorch data loader object for evaluation epochs. Leave as None to be generated during training. Defaults to None.

            test_samples (List):
                A list of test samples used by the model's `get_test_data_loader` to init the `dataset` and the
                `data_loader`. If None, the ```model.test_run()``` is expected to load the data. Defaults to None.

            training_assets (Dict):
                A dictionary of assets to be used at training and passed to the model's ```train_log(), eval_log(), get_data_loader()```
                during training. It can include  `AudioProcessor` or/and `Tokenizer`. Defaults to {}.

            parse_command_line_args (bool):
                If true, parse command-line arguments and update `TrainerArgs` and model `config` values. Set it
                to false if you parse the arguments yourself. Defaults to True.

            callbacks (Dict[str, Callable]):
                A dictionary of callbacks to be used during training. The keys are the callback names and the values

            gpu (int):
                GPU ID to use for training If "CUDA_VISIBLE_DEVICES" is not set. Defaults to None.

        Example::

            Running trainer with a model.

            >>> args = TrainerArgs(...)
            >>> config = ModelConfig(...)
            >>> model = Model(config)
            >>> trainer = Trainer(args, config, model=model)
            >>> trainer.fit()

        TODO:
                - Wrap model for not calling .module in DDP.
                - Deepspeed integration
                - Profiler integration.
                - Overfitting to a batch.
                - TPU training
        """
        if training_assets is None:
            training_assets = {}
        if callbacks is None:
            callbacks = {}

        if parse_command_line_args:
            # parse command-line arguments to override TrainerArgs()
            coqpit_overrides = args.parse_known_args(arg_prefix="")

            # get ready for training and parse command-line arguments to override the model config
            config, new_fields = self.init_training(args, coqpit_overrides, config)
        elif args.continue_path or args.restore_path:
            config, new_fields = self.init_training(args, [], config)
        else:
            new_fields = {}

        # set the output path
        if args.continue_path:
            # use the same path as the continuing run
            output_path = args.continue_path
        else:
            # override the output path if it is provided
            output_path = config.output_path if output_path is None else str(output_path)
            # create a new output folder name
            output_path = get_experiment_folder_path(output_path, config.run_name)
            output_path.mkdir(exist_ok=True, parents=True)

        # copy training assets to the output folder
        copy_model_files(config, output_path, new_fields)

        # init class members
        self.args = args
        self.config = config
        self.output_path = Path(output_path)
        self.training_assets = training_assets
        self.grad_accum_steps = args.grad_accum_steps
        self.overfit_batch = args.overfit_batch
        self.skip_train_epoch = args.skip_train_epoch
        self.start_with_eval = args.start_with_eval

        assert self.grad_accum_steps > 0, " [!] grad_accum_steps must be greater than 0."

        # setup logging
        log_file = os.path.join(self.output_path, f"trainer_{args.rank}_log.txt")
        self._setup_logger_config(log_file)

        # setup training environment
        self.use_cuda, self.num_gpus = self.setup_training_environment(args=args, config=config, gpu=gpu)

        # init loggers
        self.dashboard_logger, self.c_logger = self.init_loggers(self.config, output_path, dashboard_logger, c_logger)
        # self.c_logger.logger = logger

        # setup checkpoint manager
        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager
        else:
            # Create default checkpoint manager if not provided
            checkpoint_config = CheckpointConfig(
                keep_n_checkpoints=self.config.save_n_checkpoints,
                save_best_after=self.config.save_best_after,
                keep_all_best=self.config.save_all_best
            )
            self.checkpoint_manager = CheckpointManager(
                output_path=self.output_path,
                config=checkpoint_config,
                save_func=self.dashboard_logger.save_model if hasattr(self.dashboard_logger, 'save_model') else None
            )

        self.log_model_step = (
            self.config.log_model_step if self.config.log_model_step is not None else self.config.save_step
        )

        # make sure that start_with_eval is disabled if eval is disabled
        if not self.config.run_eval and self.start_with_eval:
            self.start_with_eval = False

        self.total_steps_done = 0
        self.epochs_done = 0
        self.restore_step = 0
        self.restore_epoch = 0
        self.best_loss: LossDict | float = {
            "train_loss": float("inf"),
            "eval_loss": float("inf") if self.config.run_eval else None,
        }
        self.train_loader: DataLoader[Any] | None = None
        self.test_loader: DataLoader[Any] | None = None
        self.eval_loader: DataLoader[Any] | None = None

        self.keep_avg_train: KeepAverage | None = None
        self.keep_avg_eval: KeepAverage | None = None

        self.use_amp_scaler = (
            self.use_cuda
            if self.config.mixed_precision and self.config.precision == "fp16"
            else self.config.use_grad_scaler
        )

        self.train_samples: list[Any] | None
        self.eval_samples: list[Any] | None
        self.test_samples: list[Any] | None
        if train_samples is not None:
            # use the provided samples
            self.train_samples = train_samples
            self.eval_samples = eval_samples
            self.test_samples = test_samples
        elif get_data_samples is not None:
            # run `get_data_samples` to init the data samples
            (
                self.train_samples,
                self.eval_samples,
                self.test_samples,
            ) = self.run_get_data_samples(config, get_data_samples)
        else:
            # expecting to load the samples in `model.get_data_loader()`
            self.train_samples = None
            self.eval_samples = None
            self.test_samples = None

        # define custom train and eval loader
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # only use a subset of the samples if small_run is set
        self.setup_small_run(args.small_run)

        # init the model
        if model is not None:
            self.model = model
        elif get_model is not None:
            self.run_get_model(self.config, get_model)
        else:
            msg = "`model` and `get_model` cannot both be None."
            raise ValueError(msg)

        # init model's training assets
        self.model.init_for_training()

        # setup criterion
        self.criterion = self.get_criterion(self.model)

        # DISTRUBUTED
        if self.use_pt_ddp:
            rank_zero_logger_info(" > Using PyTorch DDP", logger)
            init_distributed(
                args.rank,
                self.num_gpus,
                args.group_id,
                self.config.distributed_backend,
                self.config.distributed_url,
            )

        # Device setup - CUDA or TPU
        if self.config.use_tpu and TPU_AVAILABLE:
            # Move model to TPU
            self.device = get_tpu_device()
            self.model = self.model.to(self.device)
            if isinstance(self.criterion, list):
                for criterion in self.criterion:
                    if isinstance(criterion, nn.Module):
                        criterion.to(self.device)
            elif isinstance(self.criterion, nn.Module):
                self.criterion.to(self.device)
            rank_zero_logger_info(f" > Model moved to TPU device: {self.device}", logger)
            
        elif self.use_cuda and not self.config.use_deepspeed:
            # Skip CUDA move if using Deepspeed - let Deepspeed handle device placement
            self.device = torch.device("cuda")
            self.model.cuda()
            if isinstance(self.criterion, list):
                for criterion in self.criterion:
                    if isinstance(criterion, nn.Module):
                        criterion.cuda()
            elif isinstance(self.criterion, nn.Module):
                self.criterion.cuda()
        elif self.use_cuda and self.config.use_deepspeed:
            # Set device but don't move model - Deepspeed will handle this
            self.device = torch.device("cuda")
            rank_zero_logger_info(" > Delaying CUDA move for Deepspeed initialization", logger)
        else:
            self.device = torch.device("cpu")
            rank_zero_logger_info(" > Using CPU for training", logger)

        # setup optimizer
        self.optimizer = self.get_optimizer(self.model, self.config)

        # CALLBACK
        self.callbacks = TrainerCallback()
        self.callbacks.parse_callbacks_dict(callbacks)
        self.callbacks.on_init_start(self)

        # init AMP
        self.scaler = GradScaler() if self.use_amp_scaler else None

        # restore model
        if self.args.restore_path:
            (self.model, self.optimizer, self.scaler, self.restore_step, self.restore_epoch) = self.restore_model(
                self.config, args.restore_path, self.model, self.optimizer, self.scaler
            )

        # setup scheduler
        self.scheduler = self.get_scheduler(self.model, self.config, self.optimizer)
        self.scheduler = self.restore_scheduler(
            self.scheduler, self.args, self.config, self.restore_epoch, self.restore_step
        )

        # setup SWA manager
        if swa_config is not None:
            self.swa_manager = SWAManager(
                model=self.model,
                config=swa_config,
                optimizer=self.optimizer,
                output_path=self.output_path
            )
        else:
            self.swa_manager = None

        # setup Deepspeed manager
        if deepspeed_config is not None:
            self.deepspeed_manager = DeepspeedManager(
                config=deepspeed_config,
                model=self.model,
                optimizer=self.optimizer,
                trainer_config=self.config,
                output_path=self.output_path
            )
        elif self.config.use_deepspeed:
            # Create default Deepspeed config from trainer config
            auto_deepspeed_config = create_deepspeed_config(
                zero_stage=self.config.deepspeed_zero_stage,
                enable_mixed_precision=self.config.mixed_precision,
                enable_cpu_offload=self.config.deepspeed_cpu_offload,
                config_file=self.config.deepspeed_config_file,
            )
            self.deepspeed_manager = DeepspeedManager(
                config=auto_deepspeed_config,
                model=self.model,
                optimizer=self.optimizer,
                trainer_config=self.config,
                output_path=self.output_path
            )
        else:
            self.deepspeed_manager = None

        # DISTRIBUTED
        self.wrapped_model: TrainerModel | None = None
        
        # Initialize Deepspeed engine if configured
        if self.deepspeed_manager and self.deepspeed_manager.should_use_deepspeed():
            if self.use_pt_ddp or self.use_accelerate:
                logger.warning(" > Deepspeed is enabled. Disabling DDP and Accelerate for compatibility.")
                self.args.use_ddp = False
                # Note: we don't override use_accelerate property as it might be needed for other checks
            
            logger.info(" > Initializing Deepspeed engine...")
            engine = self.deepspeed_manager.initialize_engine(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                training_data=None  # Will be set later when data loader is available
            )
            
            # Update model and optimizer references to use Deepspeed
            self.model = self.deepspeed_manager.get_model()
            self.optimizer = self.deepspeed_manager.get_optimizer()
            self.wrapped_model = engine  # Use Deepspeed engine as wrapped model
            
        elif self.use_pt_ddp:
            ddp_model = DDP_th(self.model, device_ids=[args.rank], output_device=args.rank)
            self.wrapped_model = ddp_model  # Keep the DDP wrapper for training

        # setup accelerator
        self.setup_accelerate()

        # count model size
        num_params = count_parameters(self.model)
        rank_zero_logger_info(f"\n > Model has {num_params} parameters", logger)

        self.callbacks.on_init_end(self)
        self.dashboard_logger.add_config(config)
        self.save_training_script()
    ######################
    # TRAIN FUNCTIONS
    ######################
    
    def _model_train_step(
        self,
        batch: dict[str, Any] | list[Any],
        criterion: nn.Module | list[nn.Module],
        optimizer_idx: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform a training forward step. Compute model outputs and losses.

        Args:
            batch (Dict): [description]
            criterion (nn.Module): [description]
            optimizer_idx (int, optional): [description]. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses
        """
        input_args: list[Any] = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        
        # Handle Deepspeed training
        if self.use_deepspeed and self.deepspeed_manager and self.deepspeed_manager.engine is not None:
            # For Deepspeed, get the underlying model from the engine
            return self.deepspeed_manager.get_model().train_step(*input_args)
        # unwrap model in DDP training
        elif self.wrapped_model is not None:
            return self.wrapped_model.train_step(*input_args)
        return self.model.train_step(*input_args)

    def _get_autocast_args(self, *, mixed_precision: bool, precision: str) -> tuple[str, torch.dtype]:
        device = "cpu"
        dtype = torch.get_autocast_dtype("cpu") if is_pytorch_at_least_2_4() else torch.get_autocast_cpu_dtype()
        if self.use_cuda:
            device = "cuda"
            dtype = torch.float32
            if mixed_precision:
                if precision == "fp16":
                    dtype = torch.float16
                elif precision == "bf16":
                    dtype = torch.bfloat16
                else:
                    msg = f" ❗ Unknown precision {precision}"
                    raise ValueError(msg)
        elif mixed_precision:
            dtype = torch.bfloat16
        return device, dtype

    def detach_loss_dict(
        self,
        loss_dict: dict[str, Any],
        *,
        step_optimizer: bool,
        optimizer_idx: int | None = None,
        grad_norm: torch.Tensor | float | None = None,
    ) -> dict[str, Any]:
        # detach losses for logging
        loss_dict_detached = self._detach_loss_dict(loss_dict)
        # loss_dict_detached["loss"] = loss_di`ct_detached["loss"] * float(self.grad_accum_steps)

        if optimizer_idx is not None:
            loss_dict_detached[f"loss_{optimizer_idx}"] = loss_dict_detached.pop("loss")
            if step_optimizer and grad_norm is not None:
                loss_dict_detached[f"grad_norm_{optimizer_idx}"] = grad_norm
        elif step_optimizer and grad_norm is not None:
            loss_dict_detached["grad_norm"] = grad_norm
        return loss_dict_detached

    def _compute_loss(
        self,
        batch: dict[str, Any] | list[Any],
        criterion: nn.Module | list[nn.Module],
        optimizer_idx: int | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        device, dtype = self._get_autocast_args(
            mixed_precision=self.config.mixed_precision, precision=self.config.precision
        )
        with torch.autocast(device_type=device, dtype=dtype, enabled=self.config.mixed_precision):
            if optimizer_idx is not None:
                outputs, loss_dict = self._model_train_step(batch, criterion, optimizer_idx=optimizer_idx)
            else:
                outputs, loss_dict = self._model_train_step(batch, criterion)
        return outputs, loss_dict

    def _compute_grad_norm(self, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        return torch.norm(torch.cat([param.grad.view(-1) for param in self.master_params(optimizer)], dim=0), p=2)

    def _grad_clipping(
        self, grad_clip: float, optimizer: torch.optim.Optimizer, scaler: Optional["torch.GradScaler"]
    ) -> torch.Tensor:
        """Perform gradient clipping with improved stability checks.
        
        This implementation includes additional checks for NaN/Inf values and uses
        a more robust clipping method for better training stability.
        """
        if grad_clip is not None and grad_clip > 0:
            if scaler:
                scaler.unscale_(optimizer)
            self.callbacks.before_gradient_clipping(self)
            
            # Check for NaN/Inf values before clipping
            params = self.master_params(optimizer)
            for p in params:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=grad_clip, neginf=-grad_clip)
            
            # Compute and clip gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            
            # Additional stability check
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                grad_norm = torch.tensor(0.0, device=grad_norm.device)
                optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = self._compute_grad_norm(optimizer)
        return grad_norm

    def optimize(
        self,
        batch: dict[str, Any] | list[Any],
        optimizer: torch.optim.Optimizer,
        scaler: "torch.GradScaler | None",
        criterion: nn.Module | list[nn.Module],
        scheduler: LRScheduler | None,
        *,
        optimizer_idx: int | None = None,
        step_optimizer: bool = True,
        num_optimizers: int = 1,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        """Perform an optimized forward-backward pass with improved precision handling."""
        step_start_time = time.time()

        # Pre-validate input tensors
        def check_tensor_values(tensor_dict: dict[str, Any], prefix: str = "") -> bool:
            for k, v in tensor_dict.items():
                if isinstance(v, torch.Tensor):
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        logger.warning(f" [!] Found NaN/Inf in input {prefix}{k}")
                        return False
                elif isinstance(v, dict):
                    if not check_tensor_values(v, prefix=f"{k}."): 
                        return False
            return True

        if isinstance(batch, dict) and not check_tensor_values(batch, "batch."):
            step_time = time.time() - step_start_time
            return {}, {}, step_time

        # forward pass and loss computation with improved precision handling
        try:
            outputs, loss_dict = self._compute_loss(batch=batch, criterion=criterion, optimizer_idx=optimizer_idx)
        except RuntimeError as e:
            if "nan" in str(e).lower() or "infinity" in str(e).lower():
                logger.error(f" [!] NaN/Inf encountered during forward pass: {str(e)}")
            else:
                logger.error(f" [!] Error in forward pass: {str(e)}")
            step_time = time.time() - step_start_time
            return {}, {}, step_time

        # Skip if no valid outputs
        if not loss_dict:
            step_time = time.time() - step_start_time
            return outputs, {}, step_time

        grad_clip = self._set_grad_clip_per_optimizer(config=self.config, optimizer_idx=optimizer_idx)
        grad_norm: float | torch.Tensor = 0.0
        update_lr_scheduler = True

        # Pre-backward callback
        self.callbacks.before_backward_pass(self, loss_dict)

        # Gradient accumulation with improved scaling
        loss_dict["loss"] = loss_dict["loss"] / float(self.grad_accum_steps)

        # Enhanced loss validation with detailed diagnostics
        if torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"]):
            logger.warning(f" [!] Found NaN/Inf in loss. Value: {loss_dict['loss'].item()}")
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        if torch.isnan(v).any():
                            logger.warning(f" [!] NaN detected in output '{k}'")
                        if torch.isinf(v).any():
                            logger.warning(f" [!] Inf detected in output '{k}'")
            step_time = time.time() - step_start_time
            return outputs, {}, step_time

        if self.use_accelerate:
            with self.accelerator.accumulate(self.model):
                ctx_mgr = self.accelerator.autocast if self.config.mixed_precision else nullcontext
                with ctx_mgr():
                    self.accelerator.backward(loss_dict["loss"])
                    # TPU synchronization after backward pass
                    if self.config.use_tpu and TPU_AVAILABLE and step_optimizer:
                        mark_step()
                    grad_norm = self._compute_grad_norm(optimizer)
                    if self.accelerator.sync_gradients and grad_clip is not None and grad_clip > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()
                    if (
                        scheduler is not None
                        and not self.config.scheduler_after_epoch
                        and not self.accelerator.optimizer_step_was_skipped
                    ):
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    # TPU synchronization after optimizer step
                    if self.config.use_tpu and TPU_AVAILABLE and step_optimizer:
                        mark_step()
        elif self.use_deepspeed:
            # Deepspeed training path
            if self.deepspeed_manager and self.deepspeed_manager.is_initialized:
                # Backward pass through Deepspeed
                self.deepspeed_manager.backward(loss_dict["loss"])
                
                if step_optimizer and self.deepspeed_manager.is_gradient_accumulation_boundary():
                    # Gradient clipping is handled internally by Deepspeed if configured
                    grad_norm = 0.0  # Deepspeed doesn't expose grad norm easily
                    
                    # Step optimizer through Deepspeed
                    self.deepspeed_manager.step()
                    
                    # Step scheduler if configured and not after epoch
                    if (
                        scheduler is not None
                        and not self.config.scheduler_after_epoch
                    ):
                        scheduler.step()
                else:
                    grad_norm = 0.0
            else:
                logger.warning(" > Deepspeed manager not initialized properly, falling back to standard training")
                # Fallback to standard training
                loss_dict["loss"].backward()
                if step_optimizer:
                    if grad_clip > 0:
                        grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=None)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None and not self.config.scheduler_after_epoch:
                        scheduler.step()
                else:
                    grad_norm = 0.0
        else:
            if self.use_amp_scaler and scaler is not None:
                # Improved mixed precision training
                scaler.scale(loss_dict["loss"]).backward()
                # TPU synchronization after backward pass
                if self.config.use_tpu and TPU_AVAILABLE and step_optimizer:
                    mark_step()
                if step_optimizer:
                    grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=scaler)
                    scale_prev = scaler.get_scale()
                    # First step the optimizer
                    scaler.step(optimizer)
                    # Then update the scaler
                    if optimizer_idx is None or (optimizer_idx + 1 == num_optimizers):
                        scaler.update()
                        loss_dict["amp_scaler"] = scaler.get_scale()
                    update_lr_scheduler = scale_prev <= scaler.get_scale()
                    # Finally step the scheduler if needed
                    if (
                        scheduler is not None
                        and update_lr_scheduler
                        and not self.config.scheduler_after_epoch
                    ):
                        scheduler.step()
                    # TPU synchronization after optimizer step
                    if self.config.use_tpu and TPU_AVAILABLE:
                        mark_step()
            else:
                loss_dict["loss"].backward()
                # TPU synchronization after backward pass
                if self.config.use_tpu and TPU_AVAILABLE and step_optimizer:
                    mark_step()
                if step_optimizer:
                    self.callbacks.before_gradient_clipping(self)
                    if grad_clip > 0:
                        grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=None)
                    # First step the optimizer
                    optimizer.step()
                    # Then step the scheduler if needed
                    if (
                        scheduler is not None
                        and not self.config.scheduler_after_epoch
                    ):
                        scheduler.step()
                    # TPU synchronization after optimizer step
                    if self.config.use_tpu and TPU_AVAILABLE:
                        mark_step()

            if step_optimizer:
                optimizer.zero_grad(set_to_none=True)

        # Handle invalid gradient norms
        if isinstance(grad_norm, torch.Tensor) and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            grad_norm = torch.tensor(0.0, device=grad_norm.device if isinstance(grad_norm, torch.Tensor) else 'cpu')

        step_time = time.time() - step_start_time

        # Detach and prepare loss dict
        loss_dict_detached = self.detach_loss_dict(
            loss_dict, step_optimizer=step_optimizer, optimizer_idx=optimizer_idx, grad_norm=grad_norm
        )
        return outputs, loss_dict_detached, step_time

    def train_step(
        self, batch: dict[str, Any] | list[Any], batch_n_steps: int, step: int, loader_start_time: float
    ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Perform an enhanced training step with improved monitoring and stability."""
        self.callbacks.on_train_step_start(self)
        
        # Format and validate batch data
        batch = self.format_batch(batch)
        loader_time = time.time() - loader_start_time

        outputs: dict[str, Any] | list[dict[str, Any]]
        loss_dict = {}

        try:
            # Enhanced custom optimization
            step_time = time.time()
            device, dtype = self._get_autocast_args(
                mixed_precision=self.config.mixed_precision, precision=self.config.precision
            )
            
            with torch.autocast(device_type=device, dtype=dtype, enabled=self.config.mixed_precision):
                outputs, loss_dict_new = self.model.optimize(batch, self)
                
            step_time = time.time() - step_time
            
            # Skip invalid outputs
            if outputs is None:
                return None, None
                
            loss_dict_new = self.detach_loss_dict(loss_dict_new, step_optimizer=True)
            loss_dict.update(loss_dict_new)
            
        except NotImplementedError as e:
            # Enhanced gradient accumulation handling
            step_optimizer = True
            if ((step + 1) % self.grad_accum_steps != 0) and (step + 1 != batch_n_steps):
                step_optimizer = False

            if not isinstance(self.optimizer, list):
                if isinstance(self.scheduler, list):
                    msg = "Can't use list of schedulers with a single optimizer."
                    raise TypeError(msg) from e
                if isinstance(self.scheduler, dict):
                    msg = "Can only use dict of schedulers with custom `optimize()`"
                    raise TypeError(msg) from e
                    
                # Improved single optimizer training
                outputs, loss_dict_new, step_time = self.optimize(
                    batch,
                    self.optimizer,
                    self.scaler,
                    self.criterion,
                    self.scheduler,
                    step_optimizer=step_optimizer,
                    num_optimizers=1,
                )
                loss_dict.update(loss_dict_new)
            else:
                if self.grad_accum_steps != 1:
                    msg = " [!] Multiple optimizers require grad_accum_steps=1"
                    raise ValueError(msg) from e
                    
                # Enhanced multi-optimizer training
                outputs_per_optimizer = []
                total_step_time = 0.0
                
                for idx, optimizer in enumerate(self.optimizer):
                    optimizer_outputs, loss_dict_new, step_time = self.optimize(
                        batch,
                        optimizer,
                        self.scaler,
                        self.criterion,
                        self.scheduler[idx] if isinstance(self.scheduler, list) and self.scheduler is not None else None,
                        optimizer_idx=idx,
                        step_optimizer=step_optimizer,
                        num_optimizers=len(self.optimizer),
                    )
                    
                    total_step_time += step_time
                    outputs_per_optimizer.append(optimizer_outputs)
                    
                    if loss_dict_new is not None:
                        for k, v in loss_dict_new.items():
                            loss_dict[f"{k}-{idx}" if k in loss_dict else k] = v
                            
                    step_time = total_step_time
                outputs = outputs_per_optimizer

                if step_optimizer:
                    self.model.zero_grad(set_to_none=True)

        # Enhanced metrics tracking
        if self.keep_avg_train is not None:
            keep_avg_update = {
                "avg_loader_time": loader_time,
                "avg_step_time": step_time
            }
            self.keep_avg_train.update_values(keep_avg_update)

            update_eval_values = {
                f"avg_{key}": value for key, value in loss_dict.items()
            }
            self.keep_avg_train.update_values(update_eval_values)

        # Enhanced progress logging
        if self.total_steps_done % self.config.print_step == 0:
            lrs = self._get_current_learning_rates()
            loss_dict.update(lrs)
            loss_dict.update({
                "step_time": round(step_time, 4),
                "loader_time": round(loader_time, 4),
            })
            
            self.c_logger.print_train_step(
                batch_n_steps,
                step,
                self.total_steps_done,
                loss_dict,
                self.keep_avg_train.avg_values if self.keep_avg_train is not None else {},
            )

        # Enhanced checkpointing and logging
        if self.args.rank == 0:
            if self.total_steps_done % self.config.plot_step == 0:
                self.dashboard_logger.train_step_stats(self.total_steps_done, loss_dict)
                
            if (
                self.total_steps_done % self.config.save_step == 0
                and self.total_steps_done != 0
                and self.config.save_checkpoints
            ):
                self.save_checkpoint()

            if self.total_steps_done % self.log_model_step == 0:
                self.update_training_dashboard_logger(batch=batch, outputs=outputs)

            self.dashboard_logger.flush()

        self.total_steps_done += 1
        self.callbacks.on_train_step_end(self)
        return outputs, loss_dict

    def _get_current_learning_rates(self) -> dict[str, float]:
        """Helper method to get current learning rates for all optimizers."""
        if isinstance(self.optimizer, list):
            return {
                f"current_lr_{idx}": opt.param_groups[0]["lr"]
                for idx, opt in enumerate(self.optimizer)
            }
        elif isinstance(self.optimizer, dict):
            return {
                f"current_lr_{key}": opt.param_groups[0]["lr"]
                for key, opt in self.optimizer.items()
            }
        else:
            return {"current_lr": self.optimizer.param_groups[0]["lr"]}

    def train_epoch(self) -> None:
        """Main entry point for the training loop. Run training on the all training samples."""
        # initialize the data loader
        if self.train_loader is None:
            self.train_loader = self.get_train_dataloader(
                self.training_assets,
                self.train_samples,
                verbose=True,
            )
            self.train_loader = self.prepare_accelerate_loader(self.train_loader)
        # set model to training mode
        self.model.train()
        epoch_start_time = time.time()

        self.callbacks.on_train_epoch_start(self)

        self.c_logger.print_train_start()
        loader_start_time = time.time()
        
        # OVERFIT TO SINGLE BATCH -> for debugging purposes
        if self.overfit_batch:
            logger.info(" > Overfitting to a single batch for debugging...")
            try:
                # Get the first batch and reuse it throughout the epoch
                first_batch = next(iter(self.train_loader))
                batch_num_steps = len(self.train_loader)  # Keep original number of steps for logging
                
                for cur_step in range(batch_num_steps):
                    outputs, _ = self.train_step(first_batch, batch_num_steps, cur_step, loader_start_time)
                    if outputs is None:
                        logger.info(" [!] `train_step()` retuned `None` outputs. Skipping training step.")
                        continue
                    del outputs
                    loader_start_time = time.time()

                    # RUN EVAL -> run evaluation epoch in the middle of training. Useful for big datasets.
                    if self.config.run_eval_steps is not None and (self.total_steps_done % self.config.run_eval_steps == 0):
                        self.eval_epoch()
                        self.model.train()
            except StopIteration:
                logger.error(" [!] Cannot overfit to batch: training data loader is empty")
                return
            except Exception as e:
                logger.error(f" [!] Error during overfit batch training: {str(e)}")
                raise
        else:
            # TRAINING EPOCH -> iterate over the training samples
            batch_num_steps = len(self.train_loader)
            for cur_step, batch in enumerate(self.train_loader):
                outputs, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
                if outputs is None:
                    logger.info(" [!] `train_step()` retuned `None` outputs. Skipping training step.")
                    continue
                del outputs
                loader_start_time = time.time()

                # RUN EVAL -> run evaluation epoch in the middle of training. Useful for big datasets.
                if self.config.run_eval_steps is not None and (self.total_steps_done % self.config.run_eval_steps == 0):
                    self.eval_epoch()
                    self.model.train()

        epoch_time = time.time() - epoch_start_time
        self.callbacks.on_train_epoch_end(self)

        # scheduler step
        if self.scheduler is not None and self.config.scheduler_after_epoch:
            if isinstance(self.scheduler, list):
                for scheduler in self.scheduler:
                    if scheduler is not None:
                        scheduler.step()
            elif isinstance(self.scheduler, dict):  # only with `model.optimize()``
                for scheduler in self.scheduler.values():
                    if scheduler is not None:
                        scheduler.step()
            else:
                self.scheduler.step()
        # plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            if self.keep_avg_train is not None:
                epoch_stats.update(self.keep_avg_train.avg_values)
            self.dashboard_logger.train_epoch_stats(self.total_steps_done, epoch_stats)
            if self.config.model_param_stats:
                self.dashboard_logger.model_weights(self.model, self.total_steps_done)
        
        # Memory cleanup - TPU or CUDA
        if self.config.use_tpu and TPU_AVAILABLE:
            # TPU memory management and synchronization
            if self.config.tpu_metrics_debug:
                print_tpu_memory_info()
            mark_step()  # Final synchronization for the epoch
        else:
            torch.cuda.empty_cache()

