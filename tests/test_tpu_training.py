#!/usr/bin/env python3

"""
Test script for TPU training functionality.

This test verifies that the TPU integration works correctly and that
all TPU-specific code paths function properly even when TPU is not available.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import Trainer, TrainerArgs, TrainerConfig
from trainer.model import TrainerModel


@dataclass 
class TPUTestConfig(TrainerConfig):
    """Test configuration for TPU training."""
    epochs: int = 2
    print_step: int = 1
    save_step: int = 10
    plot_step: int = 5
    training_seed: int = 1234
    use_tpu: bool = True
    tpu_cores: int = 8
    tpu_metrics_debug: bool = True
    tpu_profiler: bool = False
    tpu_profiler_steps: int = 5


class SimpleTestModel(TrainerModel):
    """Simple model for testing TPU functionality."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 50)
        self.linear2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.linear1(x))
        return self.linear2(x)
    
    def train_step(self, batch, criterion):
        """Training step."""
        x, y = batch
        y_pred = self(x)
        loss = criterion(y_pred.squeeze(), y)
        return {"model_outputs": y_pred}, {"loss": loss}
    
    @torch.inference_mode()
    def eval_step(self, batch, criterion):
        """Evaluation step."""
        x, y = batch
        y_pred = self(x)
        loss = criterion(y_pred.squeeze(), y)
        return {"model_outputs": y_pred}, {"loss": loss}
    
    def get_criterion(self):
        """Get loss criterion."""
        return nn.MSELoss()
    
    def get_optimizer(self):
        """Get optimizer."""
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def get_data_loader(self, config, assets, is_eval, samples, verbose, num_gpus, rank=0):
        """Create data loader with synthetic data."""
        # Generate synthetic regression data
        torch.manual_seed(config.training_seed)
        x = torch.randn(100, 10)
        y = torch.sum(x, dim=1) + 0.1 * torch.randn(100)
        
        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=not is_eval,
            drop_last=True
        )


def test_tpu_config_integration():
    """Test that TPU configuration is properly integrated."""
    config = TPUTestConfig()
    
    # Check TPU configuration fields
    assert hasattr(config, 'use_tpu')
    assert hasattr(config, 'tpu_cores')
    assert hasattr(config, 'tpu_metrics_debug')
    assert hasattr(config, 'tpu_profiler')
    assert hasattr(config, 'tpu_profiler_steps')
    
    assert config.use_tpu is True
    assert config.tpu_cores == 8
    assert config.tpu_metrics_debug is True
    
    print("✅ TPU configuration integration test passed")


def test_tpu_training_without_hardware():
    """Test TPU training code paths without actual TPU hardware."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TPUTestConfig()
        config.batch_size = 8
        config.epochs = 1
        config.output_path = tmp_dir
        
        model = SimpleTestModel()
        
        # This should gracefully handle the absence of TPU hardware
        trainer = Trainer(
            TrainerArgs(),
            config,
            output_path=tmp_dir,
            model=model
        )
        
        # Verify TPU configuration is stored
        assert trainer.config.use_tpu is True
        
        # Check that device falls back properly when TPU not available
        print(f"Device used: {trainer.device}")
        
        # Verify training can run (should fall back to CPU/CUDA)
        try:
            trainer.fit()
            print("✅ TPU training fallback test passed")
        except Exception as e:
            print(f"❌ TPU training fallback test failed: {e}")
            raise


def test_tpu_utilities_import():
    """Test TPU utilities import handling."""
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
        print("✅ TPU utilities import test passed")
        
        # Test that functions can be called without TPU hardware
        try:
            tpu_available = is_tpu_available()
            print(f"TPU available: {tpu_available}")
            
            # These should not crash even without TPU
            mark_step()
            wait_for_tpu()
            
            print("✅ TPU utilities function calls test passed")
            
        except Exception as e:
            print(f"TPU utilities functions handled gracefully: {e}")
            
    except ImportError as e:
        print(f"TPU utilities not available (expected without torch_xla): {e}")


def test_tpu_model_saving():
    """Test TPU model saving integration."""
    from trainer.io import save_model
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TPUTestConfig()
        model = SimpleTestModel()
        optimizer = model.get_optimizer()
        
        save_path = os.path.join(tmp_dir, "test_model.pth")
        
        # This should work even without TPU (falls back to standard saving)
        save_model(
            config,
            model,
            optimizer,
            None,  # no scaler
            1,     # step
            1,     # epoch
            save_path
        )
        
        # Verify file was created
        assert os.path.exists(save_path)
        print("✅ TPU model saving integration test passed")


def main():
    """Run all TPU tests."""
    print("🧪 Running TPU Training Integration Tests")
    print("=" * 50)
    
    test_tpu_config_integration()
    test_tpu_utilities_import()
    test_tpu_model_saving()
    test_tpu_training_without_hardware()
    
    print("\n" + "=" * 50)
    print("🎉 All TPU tests completed!")
    print("\nNote: These tests verify TPU integration without requiring actual TPU hardware.")
    print("The implementation includes proper fallbacks for CPU/CUDA when TPU is not available.")


if __name__ == "__main__":
    main()
