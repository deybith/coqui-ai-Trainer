import logging
import os
import platform
from collections.abc import Callable, Generator
from inspect import signature
from typing import Any, overload

import torch
from torch import nn
from torch.utils.data import DataLoader

from trainer._types import LRScheduler
from trainer.config import TrainerArgs, TrainerConfig
from trainer.generic_utils import (
    get_git_branch,
)
from trainer.io import (
    get_last_checkpoint,
)
from trainer.logging import ConsoleLogger, DummyLogger, logger_factory
from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.model import TrainerModel
from trainer.trainer_utils import (
    get_optimizer,
    get_scheduler,
    print_training_env,
    setup_torch_training_env,
)
from trainer.utils.distributed import (
    get_rank,
)

logger = logging.getLogger("trainer")

class StaticMethods:
    @staticmethod
    def init_loggers(
        config: TrainerConfig,
        output_path: str | os.PathLike[Any],
        dashboard_logger: BaseDashboardLogger | None = None,
        c_logger: ConsoleLogger | None = None,
    ) -> tuple[BaseDashboardLogger, ConsoleLogger]:
        """Init console and dashboard loggers.

        Use the given logger if passed externally else use config values to pick the right logger.
        Return a dashboard logger only for the rank 0 process in DDP
        Define a console logger for each process in DDP

        Args:
            config (TrainerConfig): Model config.
            output_path (str): Output path to save the training artifacts.
            dashboard_logger (DashboardLogger): Object passed to the trainer from outside.
            c_logger (ConsoleLogger): Object passed to the trained from outside.

        Returns:
            Initialized dashboard_logger and console_logger objects.
        """
        c_logger = ConsoleLogger() if c_logger is None else c_logger

        # only allow dashboard logging for the main process in DDP mode
        if get_rank() > 0:
            return DummyLogger(), c_logger
        if dashboard_logger is None:
            dashboard_logger = logger_factory(config, output_path)
        return dashboard_logger, c_logger

    @staticmethod
    def init_accelerate(
        model: TrainerModel,
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
        training_dataloader: DataLoader[Any] | None,
        scheduler: LRScheduler | list[LRScheduler] | dict[str, LRScheduler] | None,
        *,
        grad_accum_steps: int,
        mixed_precision: bool,
        precision: str,
    ) -> tuple:
        """Setup HF Accelerate for the training."""
        # check if accelerate is installed
        try:
            from accelerate import Accelerator  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            msg = "Please install accelerate to use this feature."
            raise ImportError(msg) from e

        _precision = precision if precision is not None else "f16" if mixed_precision else None
        if _precision == "float16":
            _precision = "f16"
        elif _precision == "float8":
            _precision = "f8"
        elif _precision == "bfloat16":
            _precision = "bf16"
        accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps, mixed_precision=_precision)
        if isinstance(model, nn.Module):
            model = accelerator.prepare_model(model)

        if isinstance(optimizer, dict):
            for key, optim in optimizer.items():
                optimizer[key] = accelerator.prepare_optimizer(optim)
        elif isinstance(optimizer, list):
            for i, optim in enumerate(optimizer):
                optimizer[i] = accelerator.prepare_optimizer(optim)
        elif optimizer is not None:
            optimizer = accelerator.prepare_optimizer(optimizer)

        if isinstance(training_dataloader, torch.utils.data.DataLoader):
            training_dataloader = accelerator.prepare_data_loader(training_dataloader)

        if isinstance(scheduler, dict):
            for key, sched in scheduler.items():
                scheduler[key] = accelerator.prepare_scheduler(sched)
        elif isinstance(scheduler, list):
            for i, sched in enumerate(scheduler):
                scheduler[i] = accelerator.prepare_scheduler(sched)
        elif scheduler is not None:
            scheduler = accelerator.prepare_scheduler(scheduler)

        return model, optimizer, training_dataloader, scheduler, accelerator

    @staticmethod
    def init_training(
        args: TrainerArgs, coqpit_overrides: list[str], config: TrainerConfig | None = None
    ) -> tuple[TrainerConfig, dict[str, str]]:
        """Initialize training and update model configs from command line arguments.

        Args:
            args: Parsed trainer arguments.
            config_overrides: Parsed config overriding arguments.
            config: Model config. If none, it is generated from `args`. Defaults to None.

        Returns:
            config (TrainerConfig): Config paramaters.
        """
        # set arguments for continuing training
        if args.continue_path:
            config_path = os.path.join(args.continue_path, "config.json")
            args.restore_path, best_model = get_last_checkpoint(args.continue_path)
            if not args.best_path:
                args.best_path = best_model
            # use the same config
            if config:
                config.load_json(config_path)
            else:
                config = TrainerConfig()
                config.load_json(config_path)

        if config is None:
            msg = "Config or continue_path containing Config not provided"
            raise ValueError(msg)

        # override config values from command-line args
        # TODO: Maybe it is better to do it outside
        if len(coqpit_overrides) > 0:
            config.parse_known_args(coqpit_overrides, relaxed_parser=True)

        # update the config.json fields and copy it to the output folder
        new_fields = {}
        if args.rank == 0:
            if args.restore_path:
                new_fields["restore_path"] = args.restore_path
            new_fields["github_branch"] = get_git_branch()
        return config, new_fields

    @staticmethod
    def setup_training_environment(args: TrainerArgs, config: TrainerConfig, gpu: int | None) -> tuple[bool, int]:
        if platform.system() != "Windows":
            # https://github.com/pytorch/pytorch/issues/973
            import resource  # pylint: disable=import-outside-toplevel

            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

        # TPU training setup
        if config.use_tpu:
            try:
                from trainer.utils.tpu import is_tpu_available, setup_tpu_training_env  # pylint: disable=import-outside-toplevel
                
                if not is_tpu_available():
                    raise RuntimeError("TPU training requested but torch_xla is not available. Please install torch_xla.")
                
                logger.info(" > Setting up TPU training environment...")
                device, world_size = setup_tpu_training_env(training_seed=config.training_seed)
                
                # For TPU, return False for use_cuda and world_size as num_devices
                use_cuda = False
                num_devices = world_size
                
                logger.info(f" > TPU training enabled with {num_devices} cores")
                print_training_env(args, config)
                return use_cuda, num_devices
                
            except ImportError as e:
                raise RuntimeError("TPU training requested but torch_xla is not installed. Please install torch_xla.") from e
        
        # Regular CUDA training setup
        use_cuda, num_gpus = setup_torch_training_env(
            args=args,
            cudnn_enable=config.cudnn_enable,
            cudnn_deterministic=config.cudnn_deterministic,
            cudnn_benchmark=config.cudnn_benchmark,
            use_ddp=args.use_ddp,
            training_seed=config.training_seed,
            allow_tf32=config.allow_tf32,
            gpu=gpu if args.gpu is None else args.gpu,
        )

        print_training_env(args, config)
        return use_cuda, num_gpus

    @staticmethod
    @overload
    def run_get_model(config: TrainerConfig, get_model: Callable[[TrainerConfig], TrainerModel]) -> TrainerModel: ...

    @staticmethod
    @overload
    def run_get_model(config: TrainerConfig, get_model: Callable[[], TrainerModel]) -> TrainerModel: ...

    @staticmethod
    def run_get_model(config: TrainerConfig, get_model: Callable[..., TrainerModel]) -> TrainerModel:
        """Run the `get_model` function and return the model.

        Args:
            config (TrainerConfig): Model config.

        Returns:
            TrainerModel: initialized model.
        """
        return get_model(config) if len(signature(get_model).parameters) == 1 else get_model()

    @staticmethod
    def run_get_data_samples(
        config: TrainerConfig, get_data_samples: Callable[..., list[Any]]
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        if callable(get_data_samples):
            if len(signature(get_data_samples).parameters) == 1:
                train_samples, eval_samples, test_samples = get_data_samples(config)
            else:
                train_samples, eval_samples, test_samples = get_data_samples()
            return train_samples, eval_samples, test_samples
        return None, None, None

    @staticmethod
    def master_params(optimizer: torch.optim.Optimizer) -> Generator[Any]:
        """Generator over parameters owned by the optimizer.

        Used to select parameters used by the optimizer for gradient clipping.

        Args:
            optimizer: Target optimizer.
        """
        for group in optimizer.param_groups:
            yield from group["params"]

    @staticmethod
    def _set_grad_clip_per_optimizer(config: TrainerConfig, optimizer_idx: int | None) -> float:
        # set gradient clipping threshold
        grad_clip: float = 0.0  # meaning no gradient clipping
        if "grad_clip" in config and config.grad_clip is not None:
            if optimizer_idx is not None:
                if isinstance(config.grad_clip, list):
                    grad_clip = config.grad_clip[optimizer_idx]
                else:
                    logger.warning(" [!] You are using multiple optimizers but `grad_clip` is not a list.")
            else:
                if isinstance(config.grad_clip, list):
                    msg = "`grad_clip` is a list, but no optimizer_idx specified"
                    raise ValueError(msg)
                grad_clip = config.grad_clip
        return grad_clip

    #####################
    # GET FUNCTIONS
    #####################

    @staticmethod
    def get_optimizer(
        model: TrainerModel, config: TrainerConfig
    ) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
        """Return the optimizer.

        From the model if model implements `get_optimizer()` else
        check the optimizer parameters in the config and try initiating the optimizer.

        Args:
            model (TrainerModel): Training model.
            config (TrainerConfig): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List]: A optimizer or a list of optimizers. GAN models define a list.
        """
        try:
            return model.get_optimizer()
        except NotImplementedError as e:
            if isinstance(config.optimizer, list):
                optimizers = []
                for i, optimizer_name in enumerate(config.optimizer):
                    optimizer_params = {} if config.optimizer_params is None else config.optimizer_params[i]  # type: ignore[index]
                    optimizers.append(get_optimizer(optimizer_name, optimizer_params, config.lr, model))  # type: ignore[arg-type]
                return optimizers
            if config.optimizer is None:
                msg = "No name specified in `optimizer`"
                raise ValueError(msg) from e
            optimizer_name = config.optimizer
            optimizer_params = {} if config.optimizer_params is None else config.optimizer_params
            return get_optimizer(optimizer_name, optimizer_params, config.lr, model)  # type: ignore[arg-type]

    @staticmethod
    def get_lr(model: TrainerModel, config: TrainerConfig) -> float | list[float] | dict[str, float]:
        """Set the initial learning rate.

        According to the model if model implements `get_lr()` else try setting
        the learning rate from the config.

        Args:
            model (TrainerModel): Training model.
            config (TrainerConfig): Training configuration.

        Returns:
            Union[float, List[float]]: A single learning rate or a list of learning rates, one for each optimzier.
        """
        try:
            return model.get_lr()
        except NotImplementedError:
            return config.lr

    @staticmethod
    def get_scheduler(
        model: TrainerModel,
        config: TrainerConfig,
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | dict[str, torch.optim.Optimizer],
    ) -> LRScheduler | list[LRScheduler] | dict[str, LRScheduler] | None:
        """Return the scheduler.

        From the model if model implements `get_scheduler()` else
        check the config and try initiating the scheduler.

        Args:
            model (TrainerModel): Training model.
            config (TrainerConfig): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List, Dict]: A scheduler or a list of schedulers, one for each optimizer.
        """
        try:
            return model.get_scheduler(optimizer)
        except NotImplementedError:
            lr_scheduler = config.lr_scheduler
            lr_scheduler_params = config.lr_scheduler_params
            return get_scheduler(lr_scheduler, lr_scheduler_params, optimizer)  # type: ignore[arg-type]

    @staticmethod
    def restore_scheduler(
        scheduler: LRScheduler | list[LRScheduler] | dict[str, LRScheduler] | None,
        args: TrainerArgs,
        config: TrainerConfig,
        restore_epoch: int,
        restore_step: int,
    ) -> LRScheduler | list[LRScheduler] | dict[str, LRScheduler] | None:
        """Restore scheduler wrt restored model."""
        if scheduler is not None and args.continue_path:
            if isinstance(scheduler, list):
                for s in scheduler:
                    if s is not None:
                        if config.scheduler_after_epoch:
                            s.last_epoch = restore_epoch
                        else:
                            s.last_epoch = restore_step
            elif isinstance(scheduler, dict):
                for s in scheduler.values():
                    if s is not None:
                        if config.scheduler_after_epoch:
                            s.last_epoch = restore_epoch
                        else:
                            s.last_epoch = restore_step
            elif config.scheduler_after_epoch:
                scheduler.last_epoch = restore_epoch
            else:
                scheduler.last_epoch = restore_step
        return scheduler

    @staticmethod
    def get_criterion(model: TrainerModel) -> nn.Module | list[nn.Module]:
        """Receive the criterion from the model. Model must implement `get_criterion()`.

        Args:
            model (TrainerModel): Training model.

        Returns:
            nn.Module: Criterion layer.
        """
        return model.get_criterion()
