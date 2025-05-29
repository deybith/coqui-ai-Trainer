"""Test CheckpointManager integration with Trainer class."""

import tempfile
import torch
import torch.nn as nn
from pathlib import Path

from trainer import Trainer, TrainerArgs, TrainerConfig, CheckpointManager, CheckpointConfig
from trainer.model import TrainerModel


class MockModel(TrainerModel):
    """Mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
    
    def train_step(self, batch, criterion):
        x = batch["x"]
        y = batch["y"]
        outputs = self.forward(x)
        loss = criterion(outputs, y)
        return {"outputs": outputs}, {"loss": loss}
    
    def eval_step(self, batch, criterion):
        return self.train_step(batch, criterion)
    
    def get_data_loader(self, config, assets, is_eval, samples, verbose, num_gpus, rank=None):
        """Mock data loader implementation."""
        # Create a simple mock dataset
        import torch.utils.data as data
        
        class MockDataset(data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {
                    "x": torch.randn(10),
                    "y": torch.randn(1)
                }
        
        dataset = MockDataset()
        return data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    def get_criterion(self):
        """Return the loss criterion for the model."""
        return nn.MSELoss()
    
    def get_optimizer(self):
        """Return the optimizer for the model."""
        return torch.optim.Adam(self.parameters(), lr=0.001)


def test_checkpoint_manager_integration():
    """Test that CheckpointManager integrates properly with Trainer."""
    print("Testing CheckpointManager integration with Trainer...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_run"
        
        # Create trainer config
        config = TrainerConfig(
            epochs=1,
            save_step=5,
            save_n_checkpoints=2,
            save_all_best=True,
            save_best_after=0,
            print_step=1,
            plot_step=5,
            run_eval=False,
            mixed_precision=False,
            optimizer="Adam",
            lr=0.001
        )
        
        # Create trainer args
        args = TrainerArgs(
            restore_path=None,
            continue_path=None,
            rank=0,
            grad_accum_steps=1
        )
        
        # Create custom checkpoint config
        checkpoint_config = CheckpointConfig(
            keep_n_checkpoints=3,
            save_best_after=0,
            keep_all_best=True,
            validate_on_save=True,
            create_backup=False
        )
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            output_path=output_path,
            config=checkpoint_config
        )
        
        # Create trainer with checkpoint manager
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=MockModel(),
            checkpoint_manager=checkpoint_manager,
            parse_command_line_args=False
        )
        
        # Verify checkpoint manager is properly integrated
        assert hasattr(trainer, 'checkpoint_manager')
        assert trainer.checkpoint_manager is checkpoint_manager
        
        # Test saving a checkpoint
        trainer.total_steps_done = 10
        trainer.epochs_done = 1
        
        # Mock some losses for testing
        if trainer.keep_avg_train is None:
            from trainer.generic_utils import KeepAverage
            trainer.keep_avg_train = KeepAverage()
            trainer.keep_avg_train.update_values({"avg_loss": 0.5})
        
        # Mock eval losses for best model saving
        if trainer.keep_avg_eval is None:
            from trainer.generic_utils import KeepAverage
            trainer.keep_avg_eval = KeepAverage()
            trainer.keep_avg_eval.update_values({"avg_loss": 0.4})
        
        # Set a mock eval loss for best model comparison
        trainer.checkpoint_manager.best_loss = 1.0  # Start with high loss
        
        # Test checkpoint saving
        try:
            trainer.save_checkpoint()
            print("✓ Checkpoint saving works")
            
            # Check that checkpoint was created
            checkpoint_files = list(output_path.glob("checkpoint_*.pth"))
            assert len(checkpoint_files) > 0, "No checkpoint files created"
            print(f"✓ Checkpoint file created: {checkpoint_files[0].name}")
            
            # Test best model saving
            trainer.save_best_model()
            print("✓ Best model saving works")
            
            # Check that best model was created
            best_model_files = list(output_path.glob("best_model_*.pth"))
            assert len(best_model_files) > 0, "No best model files created"
            print(f"✓ Best model file created: {best_model_files[0].name}")
            
            # Test metadata functionality
            metadata = trainer.checkpoint_manager.get_checkpoint_metadata()
            assert len(metadata) > 0, "No checkpoint metadata found"
            print(f"✓ Checkpoint metadata contains {len(metadata)} entries")
            
            # Test checkpoint discovery
            latest = trainer.checkpoint_manager.get_latest_checkpoint()
            best = trainer.checkpoint_manager.get_best_checkpoint()
            assert latest is not None, "Could not find latest checkpoint"
            assert best is not None, "Could not find best checkpoint"
            print(f"✓ Latest checkpoint: {Path(latest).name}")
            print(f"✓ Best checkpoint: {Path(best).name}")
            
            # Test checkpoint summary
            summary = trainer.checkpoint_manager.create_checkpoint_summary()
            assert "total_checkpoints" in summary
            assert "total_best_models" in summary
            print(f"✓ Summary: {summary['total_checkpoints']} checkpoints, {summary['total_best_models']} best models")
            
        except Exception as e:
            print(f"✗ Error during checkpoint operations: {e}")
            raise
        
        print("✓ All CheckpointManager integration tests passed!")


def test_default_checkpoint_manager():
    """Test that Trainer creates a default CheckpointManager when none provided."""
    print("Testing default CheckpointManager creation...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_default"
        
        config = TrainerConfig(
            epochs=1,
            save_n_checkpoints=5,
            save_all_best=False,
            save_best_after=10,
            optimizer="Adam",
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        
        # Create trainer without checkpoint manager
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=MockModel(),
            parse_command_line_args=False
        )
        
        # Verify default checkpoint manager was created
        assert hasattr(trainer, 'checkpoint_manager')
        assert trainer.checkpoint_manager is not None
        print("✓ Default CheckpointManager created")
        
        # Verify it uses config from TrainerConfig
        cm_config = trainer.checkpoint_manager.config
        assert cm_config.keep_n_checkpoints == config.save_n_checkpoints
        assert cm_config.save_best_after == config.save_best_after
        assert cm_config.keep_all_best == config.save_all_best
        print("✓ Default config properly mapped from TrainerConfig")
        
        print("✓ Default CheckpointManager creation test passed!")


if __name__ == "__main__":
    test_checkpoint_manager_integration()
    test_default_checkpoint_manager()
    print("\n🎉 All CheckpointManager integration tests completed successfully!")
