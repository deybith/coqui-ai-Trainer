"""Tests for Stochastic Weight Averaging (SWA) integration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from trainer import Trainer, TrainerConfig, TrainerArgs
from trainer.swa_utils import SWAConfig, SWAManager


class SimpleMockModel(nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    def train_step(self, batch, criterion, *args, **kwargs):
        x, y = batch
        output = self(x)
        loss = criterion(output, y)
        return {"model_outputs": output}, {"loss": loss}


class TestSWAIntegration(unittest.TestCase):
    """Test SWA integration with Trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)
        
        # Create simple model and data
        self.model = SimpleMockModel()
        
        # Create simple dataset
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(x, y)
        self.train_loader = DataLoader(dataset, batch_size=10)
        
        # Create basic config
        self.config = TrainerConfig(
            epochs=5,
            run_eval=False,
            save_step=1000,  # Don't save during test
            print_step=1000,  # Don't print during test
        )
        
        self.args = TrainerArgs(
            rank=0,
            group_id="test_group"
        )
        
    def test_swa_config_creation(self):
        """Test SWA configuration creation."""
        swa_config = SWAConfig(
            enabled=True,
            swa_start=2,
            swa_freq=1,
            swa_lr=0.01
        )
        
        self.assertTrue(swa_config.enabled)
        self.assertEqual(swa_config.swa_start, 2)
        self.assertEqual(swa_config.swa_freq, 1)
        self.assertEqual(swa_config.swa_lr, 0.01)
        
    def test_swa_manager_initialization(self):
        """Test SWA manager initialization."""
        swa_config = SWAConfig(enabled=True, swa_start=2)
        optimizer = optim.Adam(self.model.parameters())
        
        swa_manager = SWAManager(
            model=self.model,
            config=swa_config,
            optimizer=optimizer,
            output_path=self.output_path
        )
        
        self.assertIsNotNone(swa_manager)
        self.assertTrue(swa_manager.config.enabled)
        self.assertIsNotNone(swa_manager.swa_model)
        self.assertIsNotNone(swa_manager.swa_scheduler)
        
    def test_swa_should_start_logic(self):
        """Test SWA start logic."""
        swa_config = SWAConfig(enabled=True, swa_start=2)
        optimizer = optim.Adam(self.model.parameters())
        
        swa_manager = SWAManager(
            model=self.model,
            config=swa_config,
            optimizer=optimizer
        )
        
        # Should not start before swa_start epoch
        self.assertFalse(swa_manager.should_start_swa(0))
        self.assertFalse(swa_manager.should_start_swa(1))
        
        # Should start at swa_start epoch
        self.assertTrue(swa_manager.should_start_swa(2))
        
        # Should not start again after already started
        swa_manager.start_swa(2)
        self.assertFalse(swa_manager.should_start_swa(3))
        
    def test_swa_update_logic(self):
        """Test SWA update logic."""
        swa_config = SWAConfig(enabled=True, swa_start=2, swa_freq=2)
        optimizer = optim.Adam(self.model.parameters())
        
        swa_manager = SWAManager(
            model=self.model,
            config=swa_config,
            optimizer=optimizer
        )
        
        # Start SWA
        swa_manager.start_swa(2)
        
        # Should update at start epoch and then every swa_freq epochs
        self.assertTrue(swa_manager.should_update_swa(2))  # Start epoch
        self.assertFalse(swa_manager.should_update_swa(3))  # +1 from start
        self.assertTrue(swa_manager.should_update_swa(4))   # +2 from start (swa_freq=2)
        self.assertFalse(swa_manager.should_update_swa(5))  # +3 from start
        self.assertTrue(swa_manager.should_update_swa(6))   # +4 from start (swa_freq=2)
        
    def test_trainer_swa_integration(self):
        """Test SWA integration with Trainer class."""
        # Create SWA config
        swa_config = SWAConfig(
            enabled=True,
            swa_start=2,
            swa_freq=1,
            swa_lr=0.01,
            save_swa_model=False  # Don't save during test
        )
        
        # Mock training assets and samples
        with patch('trainer.trainer.TrainerModel'), \
             patch('trainer.trainer.KeepAverage'), \
             patch('trainer.trainer.ConsoleLogger'), \
             patch('trainer.trainer.DummyLogger'):
            
            # Create trainer with SWA config
            trainer = Trainer(
                config=self.config,
                model=self.model,
                train_samples=list(range(100)),
                eval_samples=None,
                training_assets={},
                parse_command_line_args=False,
                args=self.args,
                output_path=self.output_path,
                swa_config=swa_config
            )
            
            # Check that SWA manager was created
            self.assertIsNotNone(trainer.swa_manager)
            self.assertTrue(trainer.swa_manager.config.enabled)
            
    def test_swa_model_averaging(self):
        """Test SWA model averaging functionality."""
        swa_config = SWAConfig(enabled=True, swa_start=0)
        optimizer = optim.Adam(self.model.parameters())
        
        swa_manager = SWAManager(
            model=self.model,
            config=swa_config,
            optimizer=optimizer
        )
        
        # Get initial weights
        initial_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Start SWA
        swa_manager.start_swa(0)
        
        # Modify model weights to simulate training
        for param in self.model.parameters():
            param.data += torch.randn_like(param.data) * 0.1
            
        # Update SWA model
        swa_manager.update_swa_model(0)
        
        # Get SWA model
        swa_model = swa_manager.get_swa_model()
        self.assertIsNotNone(swa_model)
        
        # SWA model should have averaged weights (should be different from both initial and current)
        for name, swa_param in swa_model.named_parameters():
            initial_param = initial_weights[name]
            current_param = dict(self.model.named_parameters())[name]
            
            # SWA weights should be different from current weights
            self.assertFalse(torch.equal(swa_param.data, current_param.data))
            
    def test_swa_metrics_tracking(self):
        """Test SWA metrics tracking."""
        swa_config = SWAConfig(enabled=True, swa_start=0)
        optimizer = optim.Adam(self.model.parameters())
        
        swa_manager = SWAManager(
            model=self.model,
            config=swa_config,
            optimizer=optimizer
        )
        
        # Start SWA and update
        swa_manager.start_swa(0)
        swa_manager.update_swa_model(0)
        
        # Get metrics
        metrics = swa_manager.get_swa_metrics()
        
        self.assertIn('updates_count', metrics)
        self.assertIn('epochs_active', metrics)
        self.assertIn('is_active', metrics)
        self.assertTrue(metrics['is_active'])
        self.assertEqual(metrics['updates_count'], 1)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
