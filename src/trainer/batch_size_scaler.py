"""Auto-scaling batch size finder for optimal training performance."""

import gc
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from .utils.cuda_memory import cuda_meminfo, should_reduce_batch_size
from .utils.cpu_memory import get_available_cpu_memory, is_out_of_cpu_memory

logger = logging.getLogger("trainer")


class ScalingStrategy(Enum):
    """Different strategies for batch size scaling."""
    BINARY_SEARCH = "binary_search"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSERVATIVE = "conservative"


@dataclass
class BatchSizeConfig:
    """Configuration for batch size scaling."""
    strategy: ScalingStrategy = ScalingStrategy.BINARY_SEARCH
    starting_batch_size: int = 2048
    min_batch_size: int = 1
    max_batch_size: int = 16384
    tolerance: int = 1
    max_iterations: int = 20
    safety_factor: float = 0.9
    test_steps: int = 3
    clear_cache_between_attempts: bool = True
    monitor_memory_usage: bool = True
    memory_reserve_mb: int = 512
    enable_warmup: bool = False
    warmup_steps: int = 100
    warmup_factor: float = 0.5


@dataclass
class BatchSizeResult:
    """Result of batch size scaling."""
    optimal_batch_size: int
    max_tested_batch_size: int
    iterations_used: int
    total_time_seconds: float
    memory_usage_peak_mb: float
    memory_usage_final_mb: float
    scaling_history: List[Tuple[int, bool, str]]
    strategy_used: ScalingStrategy


class BatchSizeScaler:
    """Advanced batch size scaling with multiple strategies."""
    
    def __init__(self, config: Optional[BatchSizeConfig] = None):
        self.config = config or BatchSizeConfig()
        self.scaling_history: List[Tuple[int, bool, str]] = []
        self.start_time = 0.0
        
    def find_optimal_batch_size(self, trainer, **fit_kwargs) -> BatchSizeResult:
        """Find the optimal batch size for the given trainer."""
        logger.info("🔍 Starting batch size optimization with %s strategy", self.config.strategy.value)
        self.start_time = time.time()
        self.scaling_history = []
        
        if self.config.monitor_memory_usage:
            self._log_initial_memory_state()
        
        # Choose scaling strategy
        if self.config.strategy == ScalingStrategy.BINARY_SEARCH:
            result = self._binary_search_scaling(trainer, **fit_kwargs)
        elif self.config.strategy == ScalingStrategy.EXPONENTIAL:
            result = self._exponential_scaling(trainer, **fit_kwargs)
        else:
            result = self._conservative_scaling(trainer, **fit_kwargs)
        
        # Apply safety factor
        safe_batch_size = max(
            self.config.min_batch_size,
            int(result.optimal_batch_size * self.config.safety_factor)
        )
        
        if safe_batch_size != result.optimal_batch_size:
            logger.info("🛡️ Applying safety factor %.1f: %d -> %d", 
                       self.config.safety_factor, result.optimal_batch_size, safe_batch_size)
            result.optimal_batch_size = safe_batch_size
        
        self._log_final_results(result)
        return result
    
    def _binary_search_scaling(self, trainer, **fit_kwargs) -> BatchSizeResult:
        """Use binary search to find optimal batch size."""
        logger.info("🎯 Using binary search strategy")
        
        low = self.config.min_batch_size
        high = self.config.max_batch_size
        best_working = self.config.min_batch_size
        peak_memory = 0.0
        
        for iteration in range(self.config.max_iterations):
            if high - low <= self.config.tolerance:
                break
                
            mid = (low + high) // 2
            logger.info("🔍 Testing batch size %d (iteration %d)", mid, iteration + 1)
            
            success, error_msg, memory_used = self._test_batch_size(trainer, mid, **fit_kwargs)
            peak_memory = max(peak_memory, memory_used)
            
            if success:
                best_working = mid
                low = mid + 1
                logger.info("✅ Batch size %d works!", mid)
            else:
                high = mid - 1
                logger.info("❌ Batch size %d failed: %s", mid, error_msg)
        
        return self._create_result(best_working, peak_memory)
    
    def _exponential_scaling(self, trainer, **fit_kwargs) -> BatchSizeResult:
        """Use exponential growth to find optimal batch size."""
        logger.info("📈 Using exponential growth strategy")
        
        batch_size = self.config.starting_batch_size
        best_working = self.config.min_batch_size
        peak_memory = 0.0
        
        # First, find a working starting point
        while batch_size >= self.config.min_batch_size:
            success, error_msg, memory_used = self._test_batch_size(trainer, batch_size, **fit_kwargs)
            peak_memory = max(peak_memory, memory_used)
            
            if success:
                best_working = batch_size
                break
            else:
                batch_size //= 2
        
        # Now grow exponentially
        growth_factor = 2
        while (batch_size * growth_factor <= self.config.max_batch_size and 
               len(self.scaling_history) < self.config.max_iterations):
            
            test_batch_size = min(batch_size * growth_factor, self.config.max_batch_size)
            success, error_msg, memory_used = self._test_batch_size(trainer, test_batch_size, **fit_kwargs)
            peak_memory = max(peak_memory, memory_used)
            
            if success:
                best_working = test_batch_size
                batch_size = test_batch_size
                logger.info("✅ Batch size %d works!", batch_size)
            else:
                logger.info("❌ Batch size %d failed", test_batch_size)
                break
        
        return self._create_result(best_working, peak_memory)
    
    def _conservative_scaling(self, trainer, **fit_kwargs) -> BatchSizeResult:
        """Use conservative scaling that prefers stability."""
        logger.info("🛡️ Using conservative strategy")
        
        batch_size = min(self.config.starting_batch_size // 4, 32)
        best_working = self.config.min_batch_size
        peak_memory = 0.0
        
        while (batch_size <= self.config.max_batch_size and 
               len(self.scaling_history) < self.config.max_iterations):
            
            success, error_msg, memory_used = self._test_batch_size(trainer, batch_size, **fit_kwargs)
            peak_memory = max(peak_memory, memory_used)
            
            if success:
                best_working = batch_size
                batch_size = int(batch_size * 1.25)  # Conservative 25% growth
                logger.info("✅ Batch size %d works!", best_working)
            else:
                logger.info("❌ Batch size %d failed", batch_size)
                break
        
        return self._create_result(best_working, peak_memory)
    
    def _test_batch_size(self, trainer, batch_size: int, **fit_kwargs) -> Tuple[bool, str, float]:
        """Test if a specific batch size works."""
        if self.config.clear_cache_between_attempts:
            self._clear_memory()
        
        original_batch_size = trainer.config.batch_size
        memory_before = self._get_current_memory_usage()
        
        try:
            trainer.config.batch_size = batch_size
            self._run_test_steps(trainer, **fit_kwargs)
            
            memory_after = self._get_current_memory_usage()
            memory_used = memory_after - memory_before
            
            self.scaling_history.append((batch_size, True, ""))
            return True, "", memory_used
            
        except Exception as e:
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            memory_after = self._get_current_memory_usage()
            memory_used = memory_after - memory_before
            
            self.scaling_history.append((batch_size, False, error_msg))
            return False, error_msg, memory_used
            
        finally:
            trainer.config.batch_size = original_batch_size
    
    def _run_test_steps(self, trainer, **fit_kwargs):
        """Run a few test training steps to verify batch size works."""
        original_epochs = trainer.config.epochs
        original_print_step = trainer.config.print_step
        
        try:
            trainer.config.epochs = 1
            trainer.config.print_step = 999999
            
            data_loader = trainer.get_train_dataloader()
            
            for step, batch in enumerate(data_loader):
                if step >= self.config.test_steps:
                    break
                    
                outputs, loss_dict, step_time = trainer.optimize(
                    batch=batch,
                    optimizer=trainer.optimizer,
                    scaler=trainer.scaler,
                    criterion=trainer.criterion,
                    scheduler=trainer.scheduler,
                )
                
                if loss_dict and "loss" in loss_dict:
                    loss_val = loss_dict["loss"]
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        raise RuntimeError(f"NaN/Inf loss detected: {loss_val}")
                        
        finally:
            trainer.config.epochs = original_epochs
            trainer.config.print_step = original_print_step
    
    def _create_result(self, best_working: int, peak_memory: float) -> BatchSizeResult:
        """Create a BatchSizeResult."""
        total_time = time.time() - self.start_time
        final_memory = self._get_current_memory_usage()
        
        return BatchSizeResult(
            optimal_batch_size=best_working,
            max_tested_batch_size=max(bs for bs, _, _ in self.scaling_history) if self.scaling_history else best_working,
            iterations_used=len(self.scaling_history),
            total_time_seconds=total_time,
            memory_usage_peak_mb=peak_memory,
            memory_usage_final_mb=final_memory,
            scaling_history=self.scaling_history.copy(),
            strategy_used=self.config.strategy
        )
    
    def _clear_memory(self):
        """Clear GPU and CPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0
    
    def _log_initial_memory_state(self):
        """Log initial memory state."""
        if torch.cuda.is_available():
            logger.info("🖥️ Initial GPU memory: %.2f GB allocated",
                       torch.cuda.memory_allocated() / (1024 ** 3))
    
    def _log_final_results(self, result: BatchSizeResult):
        """Log final results of batch size scaling."""
        logger.info("🎉 Batch size optimization completed!")
        logger.info("📊 Results: optimal=%d, iterations=%d, time=%.2fs", 
                   result.optimal_batch_size, result.iterations_used, result.total_time_seconds)
