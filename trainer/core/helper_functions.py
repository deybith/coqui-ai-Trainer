import logging
from typing import Any

from trainer.generic_utils import (
    KeepAverage,
)

logger = logging.getLogger("trainer")

class HelperFunctions:    
    ####################
    # HELPER FUNCTIONS
    ####################

    @staticmethod
    def _detach_loss_dict(loss_dict: dict[str, Any]) -> dict[str, Any]:
        """Detach loss values from autograp.

        Args:
            loss_dict (Dict): losses.

        Returns:
            Dict: losses detached from autograph.
        """
        loss_dict_detached = {}
        for key, value in loss_dict.items():
            if isinstance(value, (int | float)):
                loss_dict_detached[key] = value
            else:
                loss_dict_detached[key] = value.detach().cpu().item()
        return loss_dict_detached

    def _pick_target_avg_loss(self, keep_avg_target: KeepAverage | None) -> float | None:
        """Pick the target loss to compare models."""
        # if the keep_avg_target is None or empty return None
        if keep_avg_target is None or len(list(keep_avg_target.avg_values.keys())) == 0:
            return None

        # return if target loss defined in the model config
        # if not available in Dict use loss_1 as by default loss
        if "target_loss" in self.config and self.config.target_loss:
            if f"avg_{self.config.target_loss}" in keep_avg_target.avg_values:
                return keep_avg_target[f"avg_{self.config.target_loss}"]

            msg = " [!] Target loss not found in the keep_avg_target. You might be exiting the training loop before it is computed or set the target_loss in the model config incorrectly."
            raise ValueError(msg)

        # take the average of loss_{optimizer_idx} as the target loss when there are multiple optimizers
        if isinstance(self.optimizer, list):
            target_avg_loss = 0.0
            for idx in range(len(self.optimizer)):
                if f"avg_loss_{idx}" in keep_avg_target.avg_values:
                    target_avg_loss += keep_avg_target[f"avg_loss_{idx}"]
            target_avg_loss /= len(self.optimizer)
        else:
            target_avg_loss = keep_avg_target.avg_values.get("avg_loss", 0)
        return target_avg_loss

    def _setup_logger_config(self, log_file: str) -> None:
        """Set up the logger based on the process rank in DDP."""
        logger_new = logging.getLogger("trainer")
        handler = logging.FileHandler(log_file, mode="a")
        fmt = logging.Formatter("")
        handler.setFormatter(fmt)
        logger_new.addHandler(handler)

        # only log to a file if rank > 0 in DDP
        if self.args.rank > 0:
            logger_new.handlers = [h for h in logger_new.handlers if not isinstance(h, logging.StreamHandler)]
