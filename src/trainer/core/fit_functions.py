import gc
import logging
import os
import sys
import traceback
from contextlib import suppress
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist

from ..generic_utils import (
    KeepAverage,
    remove_experiment_folder,
)
from ..io import (
    save_best_model,
    save_checkpoint,
)
from ..logging import DummyLogger
from ..utils.cuda_memory import cuda_meminfo, should_reduce_batch_size
from ..utils.distributed import (
    rank_zero_only,
)

if TYPE_CHECKING:
    from .batch_size_scaler import BatchSizeConfig, BatchSizeResult

logger = logging.getLogger("trainer")

class FitFunctions:

    ###################################
    # FIT FUNCTIONS
    ###################################

    def _fit(self) -> None:
        """🏃 train -> evaluate -> test for the number of epochs."""
        self._restore_best_loss()

        self.total_steps_done = self.restore_step

        for epoch in range(self.config.epochs):
            if self.num_gpus > 1:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
            self.callbacks.on_epoch_start(self)
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs, self.output_path)
            
            # SWA: Check if we should start SWA at this epoch
            if hasattr(self, 'swa_manager') and self.swa_manager is not None:
                if self.swa_manager.should_start_swa(epoch):
                    self.swa_manager.start_swa(epoch)
            
            if not self.skip_train_epoch and not self.start_with_eval:
                self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
                self.test_run()

            # SWA: Update SWA model and scheduler if active
            if hasattr(self, 'swa_manager') and self.swa_manager is not None:
                if self.swa_manager.should_update_swa(epoch):
                    self.swa_manager.update_swa_model(epoch)
                if self.swa_manager.is_active:
                    self.swa_manager.step_swa_scheduler()
                    # Save SWA model checkpoint if configured
                    if self.swa_manager.config.save_swa_model and epoch % self.config.save_step == 0:
                        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
                        train_loss = self._pick_target_avg_loss(self.keep_avg_train) or float("inf")
                        current_loss = eval_loss if eval_loss is not None else train_loss
                        self.swa_manager.save_swa_model(
                            current_epoch=epoch,
                            current_step=self.total_steps_done,
                            current_loss=current_loss
                        )

            self.c_logger.print_epoch_end(
                epoch,
                self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values,  # type: ignore[union-attr]
            )
            if self.args.rank in [None, 0]:
                self.save_best_model()
            self.callbacks.on_epoch_end(self)
            self.start_with_eval = False
        
        # SWA: Finalize SWA after training completion
        if hasattr(self, 'swa_manager') and self.swa_manager is not None and self.swa_manager.is_active:
            logger.info("🏁 Training completed, finalizing SWA...")
            # Use train_loader for final batch normalization statistics update
            train_loader = getattr(self, 'train_loader', None)
            self.swa_manager.finalize_swa(dataloader=train_loader)

    def fit_with_largest_batch_size(self, starting_batch_size: int = 2048) -> None:
        """Find and use the largest possible batch size for training.
        
        This method uses a simple halving strategy for backward compatibility.
        For more advanced batch size optimization, use fit_with_auto_batch_size().
        
        Args:
            starting_batch_size (int): Initial batch size to try. Defaults to 2048.
        """
        cuda_meminfo()
        bs = starting_batch_size
        
        def clear_memory():
            """Clear GPU memory and garbage collection."""
            gc.collect()
            torch.cuda.empty_cache()
            
        while True:
            clear_memory()
            try:
                self.config.batch_size = bs
                logger.info(" > current batch size: %i", self.config.batch_size)
                self._fit()
                break
            except (RuntimeError, Exception) as exception:  # catches RuntimeError and torch.cuda.OutOfMemoryError
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    clear_memory()
                else:
                    raise

    def fit_with_auto_batch_size(self, batch_size_config: Optional["BatchSizeConfig"] = None) -> "BatchSizeResult":
        """Find and use the optimal batch size with advanced scaling strategies.
        
        This method provides sophisticated batch size optimization with multiple strategies,
        safety features, and detailed reporting.
        
        Args:
            batch_size_config: Configuration for batch size scaling. Uses defaults if None.
            
        Returns:
            BatchSizeResult with optimization details and final batch size used.
        """
        from .batch_size_scaler import BatchSizeScaler, BatchSizeConfig
        
        if batch_size_config is None:
            batch_size_config = BatchSizeConfig()
        
        scaler = BatchSizeScaler(batch_size_config)
        result = scaler.find_optimal_batch_size(self)
        
        # Set the optimal batch size and train
        logger.info("🚀 Starting training with optimal batch size: %d", result.optimal_batch_size)
        self.config.batch_size = result.optimal_batch_size
        self._fit()
        
        return result

    def fit(self) -> None:
        """Start the training process.
        
        This method handles the main training loop, including error handling and cleanup.
        It manages:
        - Training execution
        - Keyboard interrupts (with optional checkpoint saving)
        - Distributed training cleanup
        - Dashboard logger finalization
        - Error logging and cleanup
        
        Raises:
            RuntimeError: If there's an error during training
            torch.cuda.OutOfMemoryError: If GPU runs out of memory
            Exception: For any other unexpected errors
        """
        try:
            self._fit()
            if self.args.rank == 0:
                self.dashboard_logger.finish()
        except KeyboardInterrupt:
            logger.info(" > Keyboard interrupt detected.")
            self._handle_interrupt()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(" > Training error: %s", str(e))
            remove_experiment_folder(self.output_path)
            logger.exception("Exception occurred during training:")
            sys.exit(1)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(" > Unexpected error: %s", str(e))
            remove_experiment_folder(self.output_path)
            logger.exception("Unexpected exception occurred:")
            sys.exit(1)
            
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt with graceful shutdown."""
        if self.config.save_on_interrupt:
            logger.info(" > Saving model before exiting...")
            self.save_checkpoint()
            self.update_training_dashboard_logger()
            
        # call the keyboard interrupt callback
        self.callbacks.on_keyboard_interrupt(self)
        
        # cleanup
        remove_experiment_folder(self.output_path)
        if self.num_gpus > 1:
            dist.destroy_process_group()
        if self.args.rank == 0:
            self.dashboard_logger.finish()
            
        # exit gracefully
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)  # pylint: disable=protected-access

    def profile_fit(
        self, torch_profiler: torch.profiler.profile, epochs: int | None = None, small_run: int | None = None
    ) -> torch.profiler.profile:
        """Run training under the PyTorch profiler to analyze performance.

        This method configures and runs the training process with profiling enabled,
        allowing detailed analysis of CPU, GPU, and memory usage.

        Args:
            torch_profiler (torch.profiler.profile): Configured PyTorch profiler instance
            epochs (int, optional): Number of epochs to profile. If None, uses config value
            small_run (int, optional): Number of samples to use for profiling. If None, uses full dataset

        Returns:
            torch.profiler.profile: The profiler instance with collected data

        Example::
            Profile CPU, GPU and memory usage with Tensorboard logging:

            >>> import torch
            >>> profiler = torch.profiler.profile(
            >>>     activities=[
            >>>         torch.profiler.ProfilerActivity.CPU,
            >>>         torch.profiler.ProfilerActivity.CUDA,
            >>>     ],
            >>>     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            >>>     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
            >>>     record_shapes=True,
            >>>     profile_memory=True,
            >>>     with_stack=True,
            >>> )
            >>> prof = trainer.profile_fit(profiler, epochs=1, small_run=64)
        """
        # Configure profiling environment
        self._setup_profiling_env(torch_profiler, epochs, small_run)
        
        try:
            # Run training with profiler
            self.torch_profiler.start()
            self.fit()
            self.torch_profiler.stop()
        except Exception as e:
            logger.error(" > Profiling error: %s", str(e))
            self.torch_profiler.stop()
            raise
            
        return self.torch_profiler
        
    def _setup_profiling_env(
        self, torch_profiler: torch.profiler.profile, epochs: int | None, small_run: int | None
    ) -> None:
        """Configure the environment for profiling.
        
        Args:
            torch_profiler: The PyTorch profiler instance
            epochs: Number of epochs to profile
            small_run: Number of samples for profiling
        """
        # Use dummy logger to avoid overhead
        self.dashboard_logger = DummyLogger()
        
        # Configure training parameters
        if epochs:
            self.config.epochs = epochs
        if small_run:
            self.setup_small_run(small_run)
            
        # Disable eval and testing to focus on training
        self.config.run_eval = False
        self.config.test_delay_epochs = 9999999
        
        # Setup profiler callbacks and instance
        self.callbacks_on_train_step_end = [  # pylint: disable=attribute-defined-outside-init
            lambda trainer: trainer.torch_profiler.step()
        ]
        self.torch_profiler = torch_profiler  # pylint: disable=attribute-defined-outside-init

    @rank_zero_only
    def save_best_model(self) -> None:
        """Save the best model. It only saves if the current target loss is smaller then the previous."""
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train) or float("inf")

        # Check if this is a better model
        current_loss = {"train_loss": train_loss, "eval_loss": eval_loss}
        
        # Determine if we should save based on target loss
        if eval_loss is not None and self.best_loss.get("eval_loss") is not None:
            is_better = eval_loss < self.best_loss["eval_loss"]
        else:
            is_better = train_loss < self.best_loss["train_loss"]
        
        if not is_better or self.total_steps_done <= self.config.save_best_after:
            return

        # Handle Deepspeed checkpoint saving
        if self.use_deepspeed and hasattr(self, 'deepspeed_manager') and self.deepspeed_manager is not None:
            try:
                # Save Deepspeed checkpoint
                deepspeed_checkpoint_dir = self.deepspeed_manager.save_checkpoint(
                    save_dir=self.output_path,
                    tag=f"best_model_{self.total_steps_done}"
                )
                
                # Also save a traditional checkpoint for compatibility
                best_model_name = f"best_model_{self.total_steps_done}.pth"
                checkpoint_path = os.path.join(self.output_path, best_model_name)
                
                from ..io import save_model
                save_model(
                    self.config,
                    self._get_model(),
                    self.optimizer,
                    self.scaler if self.use_amp_scaler else None,
                    self.total_steps_done,
                    self.epochs_done,
                    checkpoint_path,
                    model_loss=current_loss,
                    deepspeed_checkpoint_dir=deepspeed_checkpoint_dir,
                    save_func=self.dashboard_logger.save_model,
                )
                
                # Update best_loss tracking
                if isinstance(self.best_loss, dict):
                    self.best_loss["train_loss"] = train_loss
                    if eval_loss is not None:
                        self.best_loss["eval_loss"] = eval_loss
                else:
                    self.best_loss = eval_loss if eval_loss is not None else train_loss
                    
                logger.info(f" > BEST DEEPSPEED MODEL saved at step {self.total_steps_done}")
                return
                
            except Exception as e:
                logger.warning(f" > Failed to save Deepspeed best model checkpoint: {e}")
                # Fall through to standard checkpoint saving

        # Use CheckpointManager for enhanced checkpoint management
        if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager is not None:
            was_saved = self.checkpoint_manager.save_best_model(
                current_loss=current_loss,
                model=self._get_model(),
                optimizer=self.optimizer,
                step=self.total_steps_done,
                epoch=self.epochs_done,
                config=self.config,
                scaler=self.scaler if self.use_amp_scaler else None,
            )
            if was_saved:
                # Update best_loss tracking for backward compatibility
                target_loss = eval_loss if eval_loss is not None else train_loss
                if isinstance(self.best_loss, dict):
                    self.best_loss["train_loss"] = train_loss
                    if eval_loss is not None:
                        self.best_loss["eval_loss"] = eval_loss
                else:
                    self.best_loss = target_loss
        else:
            # Fallback to original implementation
            from ..io import save_best_model
            self.best_loss = save_best_model(
                {"train_loss": train_loss, "eval_loss": eval_loss},
                self.best_loss,
                self.config,
                self._get_model(),
                self.optimizer,
                self.scaler if self.use_amp_scaler else None,
                self.total_steps_done,
                self.epochs_done,
                self.output_path,
                keep_all_best=self.config.save_all_best,
                keep_after=self.config.save_best_after,
                save_func=self.dashboard_logger.save_model,
            )

    @rank_zero_only
    def save_checkpoint(self) -> None:
        """Save the current model checkpoint."""
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)

        # Handle Deepspeed checkpoint saving
        if self.use_deepspeed and hasattr(self, 'deepspeed_manager') and self.deepspeed_manager is not None:
            try:
                # Save Deepspeed checkpoint
                deepspeed_checkpoint_dir = self.deepspeed_manager.save_checkpoint(
                    save_dir=self.output_path,
                    tag=f"checkpoint_{self.total_steps_done}"
                )
                
                # Also save a traditional checkpoint for compatibility
                checkpoint_name = f"checkpoint_{self.total_steps_done}.pth"
                checkpoint_path = os.path.join(self.output_path, checkpoint_name)
                
                from ..io import save_model
                save_model(
                    self.config,
                    self._get_model(),
                    self.optimizer,
                    self.scaler if self.use_amp_scaler else None,
                    self.total_steps_done,
                    self.epochs_done,
                    checkpoint_path,
                    model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
                    deepspeed_checkpoint_dir=deepspeed_checkpoint_dir,
                    save_func=self.dashboard_logger.save_model,
                )
                
                # Clean up old Deepspeed checkpoints
                if self.config.save_n_checkpoints > 0:
                    self._cleanup_deepspeed_checkpoints()
                    
                logger.info(f" > DEEPSPEED CHECKPOINT saved at step {self.total_steps_done}")
                return
                
            except Exception as e:
                logger.warning(f" > Failed to save Deepspeed checkpoint: {e}")
                # Fall through to standard checkpoint saving

        # Use CheckpointManager for enhanced checkpoint management
        if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager is not None:
            self.checkpoint_manager.save_checkpoint(
                model=self._get_model(),
                optimizer=self.optimizer,
                step=self.total_steps_done,
                epoch=self.epochs_done,
                config=self.config,
                scaler=self.scaler if self.use_amp_scaler else None,
                train_loss=train_loss,
                eval_loss=eval_loss,
            )
            # Automatic cleanup of old checkpoints
            self.checkpoint_manager.cleanup_old_checkpoints()
        else:
            # Fallback to original implementation
            from ..io import save_checkpoint
            save_checkpoint(
                self.config,
                self._get_model(),
                self.optimizer,
                self.scaler if self.use_amp_scaler else None,
                self.total_steps_done,
                self.epochs_done,
                self.output_path,
                model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
                save_n_checkpoints=self.config.save_n_checkpoints,
                save_func=self.dashboard_logger.save_model,
            )

    @rank_zero_only
    def update_training_dashboard_logger(
        self, batch: dict[str, Any] | list[Any] | None = None, outputs: dict[str, Any] | None = None
    ) -> None:
        aliases = [
            f"epoch-{self.epochs_done}",
            f"step-{self.total_steps_done}",
        ]
        self.dashboard_logger.add_artifact(
            file_or_dir=self.output_path, name="checkpoint", artifact_type="model", aliases=aliases
        )

        # training visualizations
        if batch is not None and outputs is not None:
            model = self._get_model()
            with suppress(NotImplementedError):
                model.train_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )

    def _cleanup_deepspeed_checkpoints(self) -> None:
        """Clean up old Deepspeed checkpoint directories."""
        try:
            import glob
            import shutil
            
            # Find all Deepspeed checkpoint directories
            checkpoint_pattern = os.path.join(self.output_path, "deepspeed_checkpoint_checkpoint_*")
            checkpoint_dirs = glob.glob(checkpoint_pattern)
            
            if len(checkpoint_dirs) <= self.config.save_n_checkpoints:
                return
            
            # Sort by modification time and keep only the most recent
            checkpoint_dirs.sort(key=os.path.getmtime)
            dirs_to_remove = checkpoint_dirs[:-self.config.save_n_checkpoints]
            
            for dir_path in dirs_to_remove:
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f" > Removed old Deepspeed checkpoint: {os.path.basename(dir_path)}")
                except Exception as e:
                    logger.warning(f" > Failed to remove old Deepspeed checkpoint {dir_path}: {e}")
                    
        except Exception as e:
            logger.warning(f" > Failed to cleanup old Deepspeed checkpoints: {e}")
