import logging
import os
from contextlib import suppress
from typing import cast

from .._types import LossDict
from ..generic_utils import (
    KeepAverage,
)
from ..io import (
    load_fsspec,
)
from ..model import TrainerModel

logger = logging.getLogger("trainer")

class Testing:
    ##################################
    # TESTING
    ##################################
    def test_run(self) -> None:
        """Run model test.

        Test run is expected to pass over test samples and produce logging artifacts.

        If ```model.test_run()``` is defined, it will be called and it is expected to set and execute everything
        in the model.

        Else if  ```mode.test()``` is defined, it will be called and it takes an test data loader as an argument
        and iterate over it.
        """
        self.model.eval()
        model = self._get_model()
        test_outputs = None
        try:
            test_outputs = model.test_run(self.training_assets)
        except NotImplementedError:
            self.test_loader = self.get_test_dataloader(
                self.training_assets,
                self.test_samples if self.test_samples else self.eval_samples,
                verbose=True,
            )
            # use test_loader to load test samples
            with suppress(NotImplementedError):
                test_outputs = model.test(self.training_assets, self.test_loader, None)
        with suppress(NotImplementedError):
            model.test_log(test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done)

    def _restore_best_loss(self) -> None:
        """Restore the best loss.

        Restore from the args.best_path if provided else from the model
        (`args.continue_path`) used for resuming the training.
        """
        if self.args.continue_path and (self.restore_step != 0 or self.args.best_path):
            logger.info(" > Restoring best loss from %s ...", os.path.basename(self.args.best_path))
            ch = load_fsspec(self.args.restore_path, map_location="cpu")
            if "model_loss" in ch:
                if isinstance(ch["model_loss"], dict):
                    self.best_loss = cast(LossDict, ch["model_loss"])
                # For backwards-compatibility:
                elif isinstance(ch["model_loss"], float):
                    if self.config.run_eval:
                        self.best_loss = {"train_loss": float("inf"), "eval_loss": ch["model_loss"]}
                    else:
                        self.best_loss = {"train_loss": ch["model_loss"], "eval_loss": None}
            logger.info(" > Starting with loaded last best loss %s", self.best_loss)

    def test(self, model: TrainerModel | None = None, test_samples: list[str] | None = None) -> None:
        """Run evaluation steps on the test data split.

        You can either provide the model and the test samples
        explicitly or the trainer uses values from the initialization.

        Args:
            model (TrainerModel, optional): Model to use for testing. If None, use the model given in the initialization.
                Defaults to None.

            test_samples (List[str], optional): List of test samples to use for testing. If None, use the test samples
                given in the initialization. Defaults to None.
        """
        logger.info(" > USING TEST SET...")
        self.keep_avg_eval = KeepAverage()

        if model is not None:
            self.model = model

        eval_samples_cache = self.eval_samples
        if test_samples is not None:
            self.eval_samples = test_samples
        else:
            self.eval_samples = self.test_samples

        self.eval_epoch()
        self.c_logger.print_epoch_end(self.epochs_done, self.keep_avg_eval.avg_values)
        self.eval_samples = eval_samples_cache

