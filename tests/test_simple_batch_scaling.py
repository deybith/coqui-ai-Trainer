"""Simple test for enhanced batch size scaling."""

import tempfile
import torch
import torch.nn as nn
from pathlib import Path

from trainer import Trainer, TrainerArgs, TrainerConfig
from trainer.batch_size_scaler import BatchSizeScaler, BatchSizeConfig, ScalingStrategy
from trainer.model import TrainerModel


class SimpleTestModel(TrainerModel):
    """Simple model for testing batch size scaling."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
    
    def train_step(self, batch, criterion):
        x, y = batch
        outputs = self.forward(x)
        loss = criterion(outputs, y)
        return {"outputs": outputs}, {"loss": loss}
    
    def eval_step(self, batch, criterion):
        return self.train_step(batch, criterion)
    
    def get_criterion(self):
        return nn.MSELoss()
    
    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def get_data_loader(self, config, assets, is_eval, samples, verbose, num_gpus, rank=None):
        import torch.utils.data as data
        
        class MockDataset(data.Dataset):
            def __len__(self):
                return 50
            
            def __getitem__(self, idx):
                return torch.randn(10), torch.randn(1)
        
        dataset = MockDataset()
        return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def test_simple_batch_scaling():
    """Test basic batch size scaling functionality."""
    print("Testing simple batch size scaling...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_scaling"
        
        config = TrainerConfig(
            epochs=1,
            print_step=999999,
            plot_step=999999,
            save_step=999999,
            optimizer="Adam",
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        model = SimpleTestModel()
        
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=model,
            parse_command_line_args=False
        )
        
        # Test batch size scaling
        batch_config = BatchSizeConfig(
            strategy=ScalingStrategy.BINARY_SEARCH,
            starting_batch_size=64,
            max_batch_size=128,
            min_batch_size=1,
            test_steps=1,
            max_iterations=5
        )
        
        scaler = BatchSizeScaler(batch_config)
        result = scaler.find_optimal_batch_size(trainer)
        
        print(f"✓ Found optimal batch size: {result.optimal_batch_size}")
        print(f"✓ Used {result.iterations_used} iterations")
        print(f"✓ Strategy: {result.strategy_used}")
        print(f"✓ Time taken: {result.total_time_seconds:.2f}s")
        
        assert result.optimal_batch_size >= 1
        assert result.optimal_batch_size <= 128
        assert result.iterations_used > 0
        
        print("✅ Simple batch size scaling test passed!\n")


def test_fit_with_auto_batch_size():
    """Test the trainer's new fit_with_auto_batch_size method."""
    print("Testing fit_with_auto_batch_size...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_auto"
        
        config = TrainerConfig(
            epochs=1,
            print_step=10,
            plot_step=999999,
            save_step=999999,
            optimizer="Adam",
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        model = SimpleTestModel()
        
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=model,
            parse_command_line_args=False
        )
        
        # Test with custom config
        batch_config = BatchSizeConfig(
            strategy=ScalingStrategy.EXPONENTIAL,
            starting_batch_size=8,
            max_batch_size=32,
            test_steps=1
        )
        
        # This should find optimal batch size and then train
        result = trainer.fit_with_auto_batch_size(batch_config)
        
        print(f"✓ Auto training completed with batch size: {result.optimal_batch_size}")
        print(f"✓ Final trainer batch size: {trainer.config.batch_size}")
        
        assert result.optimal_batch_size == trainer.config.batch_size
        assert hasattr(trainer, 'keep_avg_train')
        
        print("✅ fit_with_auto_batch_size test passed!\n")


if __name__ == "__main__":
    print("🧪 Testing Enhanced Batch Size Scaling (Simple)\n")
    
    test_simple_batch_scaling()
    test_fit_with_auto_batch_size()
    
    print("🎉 All simple batch size scaling tests passed!")
