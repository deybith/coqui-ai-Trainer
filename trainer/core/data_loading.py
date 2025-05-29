import logging
from contextlib import suppress
from typing import Any

import torch
from torch.utils.data import DataLoader

from trainer.config import TrainerConfig
from trainer.generic_utils import (
    to_cuda,
)
from trainer.model import TrainerModel

logger = logging.getLogger("trainer")

class DataLoading:    
    #########################
    # DATA LOADING FUNCTIONS
    #########################

    def _get_loader(
        self,
        model: TrainerModel,
        config: TrainerConfig,
        assets: dict[str, Any],
        samples: list[Any] | None,
        *,
        is_eval: bool,
        verbose: bool,
        num_gpus: int,
    ) -> DataLoader[Any]:
        loader = model.get_data_loader(
            config=config,
            assets=assets,
            is_eval=is_eval,
            samples=samples,
            verbose=verbose,
            num_gpus=num_gpus,
            rank=self.args.rank,
        )

        assert (
            len(loader) > 0
        ), " ❗ len(DataLoader) returns 0. Make sure your dataset is not empty or len(dataset) > 0. "
        return loader

    def _get_model(self) -> TrainerModel:
        """Get the model for training (DDP wrapped if available)."""
        if not hasattr(self, "wrapped_model") or self.wrapped_model is None:
            return self.model
        return self.wrapped_model
    
    def _get_unwrapped_model(self) -> TrainerModel:
        """Get the unwrapped model (without DDP wrapper)."""
        return self.model

    def get_train_dataloader(
        self, training_assets: dict[str, Any], samples: list[Any] | None, *, verbose: bool
    ) -> DataLoader[Any]:
        """Initialize and return a training data loader.

        Call ```model.get_train_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=False```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        model = self._get_model()
        try:
            return model.get_train_data_loader(
                self.config,
                self.training_assets,
                samples,
                verbose,
                self.num_gpus,
                self.args.rank,
            )
        except NotImplementedError:
            return self._get_loader(
                model,
                self.config,
                training_assets,
                samples,
                is_eval=False,
                verbose=verbose,
                num_gpus=self.num_gpus,
            )

    def get_eval_dataloader(
        self, training_assets: dict[str, Any], samples: list[Any] | None, *, verbose: bool
    ) -> DataLoader[Any]:
        """Initialize and return a evaluation data loader.

        Call ```model.get_eval_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        model = self._get_model()
        try:
            return model.get_eval_data_loader(
                self.config,
                self.training_assets,
                samples,
                verbose,
                self.num_gpus,
                self.args.rank,
            )
        except NotImplementedError:
            return self._get_loader(
                model,
                self.config,
                training_assets,
                samples,
                is_eval=True,
                verbose=verbose,
                num_gpus=self.num_gpus,
            )

    def get_test_dataloader(
        self, training_assets: dict[str, Any], samples: list[Any] | None, *, verbose: bool
    ) -> DataLoader[Any]:
        """Initialize and return a evaluation data loader.

        Call ```model.get_test_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        model = self._get_model()
        try:
            return model.get_test_data_loader(
                self.config,
                self.training_assets,
                samples,
                verbose,
                self.num_gpus,
                self.args.rank,
            )
        except NotImplementedError:
            return self._get_loader(
                model,
                self.config,
                training_assets,
                samples,
                is_eval=True,
                verbose=verbose,
                num_gpus=self.num_gpus,
            )

    def format_batch(self, batch: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
        """Format the dataloader output and return a batch.

        1. Call ```model.format_batch```.
        2. Pass the batch to the Device.
        3. Call ```model.format_batch_on_device```.

        Args:
            batch (List): Batch returned by the dataloader.

        Returns:
            Dict: Formatted batch.
        """
        with suppress(NotImplementedError):
            batch = (
                self.wrapped_model.format_batch(batch)
                if self.wrapped_model is not None
                else self.model.format_batch(batch)
            )

        # Get the correct device for moving tensors
        target_device = None
        if hasattr(self, 'device') and self.device is not None:
            target_device = self.device
            print(f"  [DEBUG] format_batch: Using trainer device: {target_device}")
        elif hasattr(self, 'use_deepspeed') and self.use_deepspeed and hasattr(self, 'deepspeed_manager'):
            # For Deepspeed, get the device from the engine if available
            if self.deepspeed_manager and self.deepspeed_manager.engine is not None:
                if hasattr(self.deepspeed_manager.engine, 'module'):
                    try:
                        model_device = next(self.deepspeed_manager.engine.module.parameters()).device
                        target_device = model_device
                        print(f"  [DEBUG] format_batch: Using Deepspeed model device: {target_device}")
                    except StopIteration:
                        # Fallback to default cuda device if no parameters found
                        target_device = torch.device("cuda") if torch.cuda.is_available() else None
                        print(f"  [DEBUG] format_batch: Using fallback device: {target_device}")
        
        print(f"  [DEBUG] format_batch: Final target device: {target_device}")

        if isinstance(batch, dict):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"  [DEBUG] format_batch: Moving {k} from {v.device} to {target_device}")
                batch[k] = to_cuda(v, device=target_device)
        elif isinstance(batch, list):
            batch = [to_cuda(v, device=target_device) for v in batch]

        with suppress(NotImplementedError):
            batch = (
                self.wrapped_model.format_batch_on_device(batch)
                if self.wrapped_model is not None
                else self.model.format_batch_on_device(batch)
            )
        return batch

