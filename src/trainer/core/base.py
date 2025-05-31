import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from ..config import TrainerArgs, TrainerConfig
from ..generic_utils import (
    set_partial_state_dict,
)
from ..io import (
        load_fsspec,
)
from ..model import TrainerModel

logger = logging.getLogger("trainer")

class Base:
    @property
    def use_pt_ddp(self) -> bool:
        """Return True if using PyTorch DDP."""
        return self.num_gpus > 1 and not self.use_accelerate and not self.use_deepspeed

    @property
    def use_accelerate(self) -> bool:
        """Return True if using HF Accelerate."""
        return self.args.use_accelerate and not self.use_deepspeed

    @property
    def use_deepspeed(self) -> bool:
        """Return True if using Deepspeed."""
        return hasattr(self, 'deepspeed_manager') and self.deepspeed_manager and self.deepspeed_manager.should_use_deepspeed()

    def setup_accelerate(self) -> None:
        if self.use_accelerate and not self.use_deepspeed:
            self.model, self.optimizer, self.train_loader, self.scheduler, self.accelerator = self.init_accelerate(
                model=self.model,
                optimizer=self.optimizer,
                training_dataloader=self.train_loader,
                scheduler=self.scheduler,
                grad_accum_steps=self.grad_accum_steps,
                mixed_precision=self.config.mixed_precision,
                precision=self.config.precision,
            )

    def prepare_accelerate_loader(self, data_loader: DataLoader[Any]) -> DataLoader[Any]:
        """Prepare the accelerator for the training."""
        if self.use_accelerate:
            return self.accelerator.prepare_data_loader(data_loader)
        return data_loader

    def save_training_script(self) -> None:
        """Save the training script to tracking dashboard and output path."""
        file_path = Path(sys.argv[0])
        if file_path.is_file():
            file_name = file_path.name
            self.dashboard_logger.add_artifact(file_or_dir=file_path, name=file_name, artifact_type="file")
            with file_path.open(encoding="utf8") as f:
                self.dashboard_logger.add_text("training-script", f"{f.read()}", 0)
            shutil.copyfile(file_path, self.output_path / file_name)

    def setup_small_run(self, small_run: int | None = None) -> None:
        """Use a subset of samples for training, evaluation and testing."""
        if small_run is not None:
            logger.info("[!] Small Run, only using %i samples.", small_run)
            self.train_samples = None if self.train_samples is None else self.train_samples[:small_run]
            self.eval_samples = None if self.eval_samples is None else self.eval_samples[:small_run]
            self.test_samples = None if self.test_samples is None else self.test_samples[:small_run]

   
    def restore_model(
        self,
        config: TrainerConfig,
        restore_path: str | os.PathLike[Any],
        model: TrainerModel,
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
        scaler: Optional["torch.GradScaler"] = None,
    ) -> tuple[TrainerModel, torch.optim.Optimizer | list[torch.optim.Optimizer], "torch.GradScaler | None", int, int]:
        """Restore training from an old run. It restores model, optimizer, AMP scaler and training stats.

        Args:
            config (TrainerConfig): Model config.
            restore_path (str): Path to the restored training run.
            model (TrainerModel): Model to restored.
            optimizer (torch.optim.Optimizer): Optimizer to restore.
            scaler (torch.GradScaler, optional): AMP scaler to restore. Defaults to None.

        Returns:
            Tuple[TrainerModel, torch.optim.Optimizer, torch.GradScaler, int, int]: [description]
        """

        def _restore_list_objs(states: Any, obj: Any) -> Any:
            if isinstance(obj, list):
                for idx, state in enumerate(states):
                    obj[idx].load_state_dict(state)
            elif isinstance(obj, dict):
                for key, state in states.items():
                    obj[key].load_state_dict(state)
            else:
                obj.load_state_dict(states)
            return obj

        logger.info(" > Restoring from %s ...", os.path.basename(restore_path))
        
        # Check if this is a Deepspeed checkpoint restoration
        if self.use_deepspeed and hasattr(self, 'deepspeed_manager') and self.deepspeed_manager is not None:
            try:
                # Check for Deepspeed checkpoint directory
                restore_dir = os.path.dirname(restore_path)
                
                # Look for Deepspeed checkpoint directories
                import glob
                deepspeed_checkpoint_pattern = os.path.join(restore_dir, "deepspeed_checkpoint_*")
                deepspeed_checkpoints = glob.glob(deepspeed_checkpoint_pattern)
                
                if deepspeed_checkpoints:
                    # Use the most recent Deepspeed checkpoint
                    latest_deepspeed_checkpoint = max(deepspeed_checkpoints, key=os.path.getmtime)
                    logger.info(f" > Found Deepspeed checkpoint: {os.path.basename(latest_deepspeed_checkpoint)}")
                    
                    # Load Deepspeed checkpoint
                    checkpoint_path, client_state = self.deepspeed_manager.load_checkpoint(
                        load_dir=latest_deepspeed_checkpoint,
                        load_optimizer_states=True,
                        load_lr_scheduler_states=True
                    )
                    
                    # Also load traditional checkpoint for metadata
                    checkpoint = load_fsspec(restore_path, map_location="cpu")
                    
                    logger.info(" > Deepspeed checkpoint restored successfully")
                    restore_step = checkpoint["step"] + 1
                    restore_epoch = checkpoint["epoch"]
                    torch.cuda.empty_cache()
                    return model, optimizer, scaler, restore_step, restore_epoch
                    
            except Exception as e:
                logger.warning(f" > Failed to load Deepspeed checkpoint: {e}")
                logger.info(" > Falling back to standard checkpoint loading...")

        # Standard checkpoint loading
        checkpoint = load_fsspec(restore_path, map_location="cpu")

        try:
            logger.info(" > Restoring Model...")
            model.load_state_dict(checkpoint["model"])
            logger.info(" > Restoring Optimizer...")
            try:
                optimizer = _restore_list_objs(checkpoint["optimizer"], optimizer)
            except (KeyError, TypeError, RuntimeError):
                logger.info(" > Optimizer is not compatible with the restored model.")
            if "scaler" in checkpoint and self.use_amp_scaler and checkpoint["scaler"]:
                logger.info(" > Restoring Scaler...")
                scaler = _restore_list_objs(checkpoint["scaler"], scaler)
        except (KeyError, RuntimeError, ValueError):
            logger.info(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_partial_state_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        optimizer = self.restore_lr(config, self.args, model, optimizer)

        logger.info(" > Model restored from step %i", checkpoint["step"])
        restore_step = checkpoint["step"] + 1  # +1 not to immediately checkpoint if the model is restored
        restore_epoch = checkpoint["epoch"]
        torch.cuda.empty_cache()
        return model, optimizer, scaler, restore_step, restore_epoch

    def restore_lr(
        self,
        config: TrainerConfig,
        args: TrainerArgs,
        model: TrainerModel,
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
    ) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
        # use the same lr if continue training
        if not args.continue_path:
            if isinstance(optimizer, list):
                for idx, optim in enumerate(optimizer):
                    for group in optim.param_groups:
                        group["lr"] = self.get_lr(model, config)[idx]  # type: ignore[index]
            elif isinstance(optimizer, dict):
                for optim_name, optim in optimizer.items():
                    for group in optim.param_groups:
                        group["lr"] = self.get_lr(model, config)[optim_name]  # type: ignore[index]
            else:
                for group in optimizer.param_groups:
                    group["lr"] = self.get_lr(model, config)
        return optimizer
