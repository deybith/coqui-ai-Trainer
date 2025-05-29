"""Tests for BaseTrainingConfig functionality."""

import os
import tempfile
from dataclasses import dataclass, field
from unittest import TestCase

from trainer import BaseTrainingConfig, TrainerArgs, Trainer
from trainer.model import TrainerModel
import torch
import torch.nn as nn


@dataclass
class TestModelConfig(BaseTrainingConfig):
    """Test configuration extending BaseTrainingConfig."""
    
    # Model-specific parameters
    model: str = "test_model"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    
    # Override some base config values for testing
    num_loader_workers: int = 2
    batch_size: int = 8
    epochs: int = 2
    lr: float = 0.001
    print_step: int = 1
    save_step: int = 10
    use_data_cache: bool = True
    max_seq_len: int = 100
    eval_split_size: float = 0.2


class TestModel(TrainerModel):
    """Simple test model for testing BaseTrainingConfig."""
    
    def __init__(self, config: TestModelConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return self.dropout(self.linear(x))
    
    def train_step(self, batch, criterion):
        """Simple training step for testing."""
        # Create dummy loss for testing
        x = torch.randn(self.config.batch_size, self.config.hidden_size)
        output = self.forward(x)
        loss = torch.mean(output ** 2)  # Simple loss for testing
        
        return {"model_output": output}, {"loss": loss}
    
    def eval_step(self, batch, criterion):
        """Simple evaluation step for testing."""
        return self.train_step(batch, criterion)
    
    def get_optimizer(self):
        """Return optimizer for testing."""
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    
    def get_criterion(self):
        """Return criterion for testing."""
        return nn.MSELoss()


class TestBaseTrainingConfig(TestCase):
    """Test cases for BaseTrainingConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TestModelConfig()
        self.args = TrainerArgs()
        
    def test_base_training_config_inheritance(self):
        """Test that BaseTrainingConfig properly inherits from TrainerConfig."""
        # Check that it has TrainerConfig fields
        self.assertTrue(hasattr(self.config, 'epochs'))
        self.assertTrue(hasattr(self.config, 'batch_size'))
        self.assertTrue(hasattr(self.config, 'lr'))
        self.assertTrue(hasattr(self.config, 'output_path'))
        
        # Check that it has BaseTrainingConfig fields
        self.assertTrue(hasattr(self.config, 'model'))
        self.assertTrue(hasattr(self.config, 'num_loader_workers'))
        self.assertTrue(hasattr(self.config, 'data_path'))
        self.assertTrue(hasattr(self.config, 'use_data_cache'))
        self.assertTrue(hasattr(self.config, 'max_seq_len'))
        
    def test_base_training_config_defaults(self):
        """Test default values of BaseTrainingConfig fields."""
        base_config = BaseTrainingConfig()
        
        # Test BaseTrainingConfig specific defaults
        self.assertIsNone(base_config.model)
        self.assertEqual(base_config.num_loader_workers, 0)
        self.assertEqual(base_config.num_eval_loader_workers, 0)
        self.assertEqual(base_config.data_path, "")
        self.assertFalse(base_config.use_data_cache)
        self.assertEqual(base_config.cache_path, "")
        self.assertIsNone(base_config.max_seq_len)
        self.assertIsNone(base_config.min_seq_len)
        self.assertFalse(base_config.use_data_augmentation)
        self.assertEqual(base_config.eval_split_size, 0.1)
        self.assertEqual(base_config.test_split_size, 0.1)
        self.assertFalse(base_config.use_weighted_sampling)
        self.assertFalse(base_config.compute_linear_spec)
        self.assertTrue(base_config.compute_mel_spec)
        
        # Test that TrainerConfig defaults are preserved
        self.assertEqual(base_config.epochs, 1000)
        self.assertEqual(base_config.batch_size, 32)
        self.assertEqual(base_config.lr, 0.001)
        
    def test_config_override(self):
        """Test that config values can be properly overridden."""
        self.assertEqual(self.config.model, "test_model")
        self.assertEqual(self.config.hidden_size, 128)
        self.assertEqual(self.config.num_layers, 2)
        self.assertEqual(self.config.dropout, 0.1)
        self.assertEqual(self.config.num_loader_workers, 2)
        self.assertEqual(self.config.batch_size, 8)
        self.assertEqual(self.config.epochs, 2)
        self.assertTrue(self.config.use_data_cache)
        self.assertEqual(self.config.max_seq_len, 100)
        self.assertEqual(self.config.eval_split_size, 0.2)
        
    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        # Convert to dict
        config_dict = self.config.to_dict()
        
        # Check that all fields are present
        self.assertIn('model', config_dict)
        self.assertIn('hidden_size', config_dict)
        self.assertIn('num_loader_workers', config_dict)
        self.assertIn('epochs', config_dict)
        self.assertIn('batch_size', config_dict)
        
        # Check values
        self.assertEqual(config_dict['model'], "test_model")
        self.assertEqual(config_dict['hidden_size'], 128)
        self.assertEqual(config_dict['num_loader_workers'], 2)
        
    def test_config_with_trainer_integration(self):
        """Test that BaseTrainingConfig works with Trainer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config for test
            self.config.output_path = temp_dir
            self.config.epochs = 1
            self.config.save_step = 1000  # Don't save during test
            
            # Create model
            model = TestModel(self.config)
            
            # Test that trainer can be initialized with BaseTrainingConfig
            try:
                trainer = Trainer(
                    args=self.args,
                    config=self.config,
                    model=model,
                    output_path=temp_dir,
                    parse_command_line_args=False
                )
                
                # Verify trainer initialized correctly
                self.assertIsNotNone(trainer)
                self.assertEqual(trainer.config.model, "test_model")
                self.assertEqual(trainer.config.num_loader_workers, 2)
                self.assertEqual(trainer.config.use_data_cache, True)
                
            except Exception as e:
                self.fail(f"Trainer initialization failed with BaseTrainingConfig: {e}")
                
    def test_config_field_types(self):
        """Test that config fields have correct types."""
        self.assertIsInstance(self.config.model, str)
        self.assertIsInstance(self.config.num_loader_workers, int)
        self.assertIsInstance(self.config.num_eval_loader_workers, int)
        self.assertIsInstance(self.config.data_path, str)
        self.assertIsInstance(self.config.use_data_cache, bool)
        self.assertIsInstance(self.config.cache_path, str)
        self.assertIsInstance(self.config.use_data_augmentation, bool)
        self.assertIsInstance(self.config.eval_split_size, float)
        self.assertIsInstance(self.config.test_split_size, float)
        self.assertIsInstance(self.config.use_weighted_sampling, bool)
        self.assertIsInstance(self.config.compute_linear_spec, bool)
        self.assertIsInstance(self.config.compute_mel_spec, bool)
        
    def test_config_validation_ranges(self):
        """Test that config values are within expected ranges."""
        # Test split sizes are valid probabilities
        self.assertTrue(0.0 <= self.config.eval_split_size <= 1.0)
        self.assertTrue(0.0 <= self.config.test_split_size <= 1.0)
        
        # Test worker counts are non-negative
        self.assertTrue(self.config.num_loader_workers >= 0)
        self.assertTrue(self.config.num_eval_loader_workers >= 0)
        
    def test_config_json_export_import(self):
        """Test JSON export and import functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "test_config.json")
            
            # Save config to JSON
            self.config.save_json(json_path)
            self.assertTrue(os.path.exists(json_path))
            
            # Load config from JSON
            loaded_config = TestModelConfig()
            loaded_config.load_json(json_path)
            
            # Verify values match
            self.assertEqual(loaded_config.model, self.config.model)
            self.assertEqual(loaded_config.hidden_size, self.config.hidden_size)
            self.assertEqual(loaded_config.num_loader_workers, self.config.num_loader_workers)
            self.assertEqual(loaded_config.epochs, self.config.epochs)
            self.assertEqual(loaded_config.batch_size, self.config.batch_size)
            self.assertEqual(loaded_config.use_data_cache, self.config.use_data_cache)


if __name__ == '__main__':
    import unittest
    unittest.main()
