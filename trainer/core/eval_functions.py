import logging
import time
from contextlib import suppress
from inspect import signature
from typing import Any

import torch
from torch import nn

from trainer.generic_utils import (
    KeepAverage,
)
from trainer.model import TrainerModel

logger = logging.getLogger("trainer")

class EvalFunctions:
    #######################
    # EVAL FUNCTIONS
    #######################

    def _model_eval_step(
        self,
        batch: dict[str, Any],
        model: TrainerModel,
        criterion: nn.Module | list[nn.Module],
        optimizer_idx: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform a evaluation forward pass. Compute model outputs and losses with no gradients.

        Args:
            batch (Dict): IBatch of inputs.
            model (TrainerModel): Model to call evaluation.
            criterion (nn.Module): Model criterion.
            optimizer_idx (int, optional): Optimizer ID to define the closure in multi-optimizer training. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: model outputs and losses.
        """
        input_args: list[Any] = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)

        return self._get_model().eval_step(*input_args)

    def eval_step(
        self, batch: dict[str, Any], step: int
    ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Perform a evaluation step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            step (int): Current step number in this epoch.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        outputs: dict[str, Any] | list[dict[str, Any]]
        with torch.inference_mode():
            loss_dict: dict[str, Any] = {}
            model = self._get_model()
            if not isinstance(self.optimizer, list) or len(signature(model.eval_step).parameters) == 2:  # noqa: PLR2004
                outputs, loss_dict = model.eval_step(batch, self.criterion)
                if outputs is None:
                    return None, None
            else:
                optimizer_outputs = []
                for idx, _ in enumerate(self.optimizer):
                    outputs_, loss_dict_new = model.eval_step(batch, self.criterion, idx)
                    if outputs_ is None:
                        return None, None
                    optimizer_outputs.append(outputs_)

                    if loss_dict_new:
                        loss_dict_new[f"loss_{idx}"] = loss_dict_new.pop("loss")
                        loss_dict.update(loss_dict_new)
                outputs = optimizer_outputs

            loss_dict = self._detach_loss_dict(loss_dict)

            # update avg stats
            if self.keep_avg_eval is not None:
                update_eval_values = {}
                for key, value in loss_dict.items():
                    update_eval_values["avg_" + key] = value
                self.keep_avg_eval.update_values(update_eval_values)

            if self.config.print_eval:
                self.c_logger.print_eval_step(
                    step, loss_dict, self.keep_avg_eval.avg_values if self.keep_avg_eval is not None else {}
                )

        return outputs, loss_dict

    @torch.inference_mode()
    def eval_epoch(self) -> None:
        """Main entry point for the evaluation loop. Run evaluation on the all validation samples."""
        # initialize it when eval_epoch is called alone.
        self.keep_avg_eval = KeepAverage() if self.keep_avg_eval is None else self.keep_avg_eval

        if self.eval_loader is None:
            self.eval_loader = (
                self.get_eval_dataloader(
                    self.training_assets,
                    self.eval_samples,
                    verbose=True,
                )
                if self.config.run_eval
                else None
            )

        self.model.eval()
        self.c_logger.print_eval_start()
        loader_start_time = time.time()
        batch = None
        outputs = None
        for cur_step, batch in enumerate(self.eval_loader):  # type: ignore[arg-type]
            # format data
            batch = self.format_batch(batch)
            loader_time = time.time() - loader_start_time
            self.keep_avg_eval.update_values({"avg_loader_time": loader_time})
            outputs_, _ = self.eval_step(batch, cur_step)
            if outputs_ is None:
                logger.info(" [!] `eval_step()` retuned `None` outputs. Skipping evaluation step.")
                continue
            outputs = outputs_
            loader_start_time = time.time()
        # plot epoch stats, artifacts and figures
        if self.args.rank == 0 and outputs is not None:
            model = self._get_model()
            with suppress(NotImplementedError):
                model.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            self.dashboard_logger.eval_stats(self.total_steps_done, self.keep_avg_eval.avg_values)
        
        # Memory cleanup - TPU or CUDA
        if hasattr(self.config, 'use_tpu') and self.config.use_tpu:
            try:
                from trainer.utils.tpu import mark_step, print_tpu_memory_info
                if hasattr(self.config, 'tpu_metrics_debug') and self.config.tpu_metrics_debug:
                    print_tpu_memory_info()
                mark_step()  # Synchronize TPU operations
            except ImportError:
                pass  # TPU not available
        else:
            torch.cuda.empty_cache()
