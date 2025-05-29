"""Test enhanced batch size scaling functionality."""

import tempfile
import torch
import torch.nn as nn
from pathlib import Path

from trainer import Trainer, TrainerArgs, TrainerConfig, BatchSizeScaler, BatchSizeConfig, ScalingStrategy
from trainer.model import TrainerModel


class MockModelForBatchTesting(TrainerModel):
    """Mock model optimized for batch size testing."""
    
    def __init__(self, complexity_factor=1):
        super().__init__()
        # Create a model with configurable complexity
        hidden_size = 128 * complexity_factor
        self.layers = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )
    
    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))
    
    def train_step(self, batch, criterion):
        x, y = batch
        outputs = self.forward(x)
        loss = criterion(outputs, y)
        return {"outputs": outputs}, {"loss": loss}
    
    def eval_step(self, batch, criterion):
        return self.train_step(batch, criterion)
    
    def get_criterion(self):
        return nn.CrossEntropyLoss()
    
    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def get_data_loader(self, config, assets, is_eval, samples, verbose, num_gpus, rank=None):
        import torch.utils.data as data
        
        class MockDataset(data.Dataset):
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                return torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).long()
        
        dataset = MockDataset()
        return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def test_binary_search_strategy():
    """Test binary search batch size scaling."""
    print("Testing binary search strategy...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_binary"
        
        config = TrainerConfig(
            epochs=1,
            print_step=999999,
            plot_step=999999,
            save_step=999999,
            optimizer="Adam",
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        model = MockModelForBatchTesting(complexity_factor=1)
        
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=model,
            parse_command_line_args=False
        )
        
        # Configure batch size scaling
        batch_config = BatchSizeConfig(
            strategy=ScalingStrategy.BINARY_SEARCH,
            starting_batch_size=128,
            max_batch_size=512,
            min_batch_size=1,
            test_steps=2,
            max_iterations=10
        )
        
        scaler = BatchSizeScaler(batch_config)
        result = scaler.find_optimal_batch_size(trainer)
        
        print(f"✓ Binary search found optimal batch size: {result.optimal_batch_size}")
        print(f"✓ Used {result.iterations_used} iterations in {result.total_time_seconds:.2f}s")
        print(f"✓ Peak memory usage: {result.memory_usage_peak_mb:.2f} MB")
        
        assert result.optimal_batch_size >= batch_config.min_batch_size
        assert result.optimal_batch_size <= batch_config.max_batch_size
        assert result.strategy_used == ScalingStrategy.BINARY_SEARCH
        assert len(result.scaling_history) > 0


def test_exponential_strategy():
    """Test exponential growth batch size scaling."""
    print("Testing exponential strategy...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_exponential"
        
        config = TrainerConfig(
            epochs=1,
            print_step=999999,
            plot_step=999999, 
            save_step=999999,
            optimizer="Adam",
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        model = MockModelForBatchTesting(complexity_factor=1)
        
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=model,
            parse_command_line_args=False
        )
        
        batch_config = BatchSizeConfig(
            strategy=ScalingStrategy.EXPONENTIAL,
            starting_batch_size=8,
            max_batch_size=256,
            test_steps=2,
            max_iterations=8
        )
        
        scaler = BatchSizeScaler(batch_config)
        result = scaler.find_optimal_batch_size(trainer)
        
        print(f"✓ Exponential strategy found optimal batch size: {result.optimal_batch_size}")
        print(f"✓ Used {result.iterations_used} iterations")
        
        assert result.optimal_batch_size >= batch_config.min_batch_size
        assert result.strategy_used == ScalingStrategy.EXPONENTIAL


def test_conservative_strategy():
    """Test conservative batch size scaling."""
    print("Testing conservative strategy...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_conservative"
        
        config = TrainerConfig(
            epochs=1,
            print_step=999999,
            plot_step=999999,
            save_step=999999,
            optimizer="Adam", 
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        model = MockModelForBatchTesting(complexity_factor=1)
        
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=model,
            parse_command_line_args=False
        )
        
        batch_config = BatchSizeConfig(
            strategy=ScalingStrategy.CONSERVATIVE,
            starting_batch_size=64,
            max_batch_size=256,
            test_steps=2,
            max_iterations=10,
            safety_factor=0.8  # Extra conservative
        )
        
        scaler = BatchSizeScaler(batch_config)
        result = scaler.find_optimal_batch_size(trainer)
        
        print(f"✓ Conservative strategy found optimal batch size: {result.optimal_batch_size}")
        
        assert result.strategy_used == ScalingStrategy.CONSERVATIVE


def test_fit_with_auto_batch_size():
    """Test the trainer's fit_with_auto_batch_size method."""
    print("Testing fit_with_auto_batch_size integration...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_auto_fit"
        
        config = TrainerConfig(
            epochs=1,
            print_step=5,
            plot_step=999999,
            save_step=999999,
            optimizer="Adam",
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        model = MockModelForBatchTesting(complexity_factor=1)
        
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=model,
            parse_command_line_args=False
        )
        
        # Test with custom config
        batch_config = BatchSizeConfig(
            strategy=ScalingStrategy.BINARY_SEARCH,
            starting_batch_size=32,
            max_batch_size=128,
            test_steps=1,
            safety_factor=0.9
        )
        
        # This should find optimal batch size and then train
        result = trainer.fit_with_auto_batch_size(batch_config)
        
        print(f"✓ Auto batch size training completed with batch size: {result.optimal_batch_size}")
        print(f"✓ Final trainer batch size: {trainer.config.batch_size}")
        
        assert result.optimal_batch_size == trainer.config.batch_size
        assert hasattr(trainer, 'keep_avg_train')


def test_batch_size_scaling_with_memory_constraints():
    """Test batch size scaling under memory constraints."""
    print("Testing batch size scaling with memory constraints...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_memory"
        
        config = TrainerConfig(
            epochs=1,
            print_step=999999,
            plot_step=999999,
            save_step=999999,
            optimizer="Adam",
            lr=0.001
        )
        
        args = TrainerArgs(rank=0, grad_accum_steps=1)
        # Use a more complex model to create memory pressure
        model = MockModelForBatchTesting(complexity_factor=3)
        
        trainer = Trainer(
            args=args,
            config=config,
            output_path=output_path,
            model=model,
            parse_command_line_args=False
        )
        
        batch_config = BatchSizeConfig(
            strategy=ScalingStrategy.BINARY_SEARCH,
            starting_batch_size=1024,  # Start high to trigger memory issues
            max_batch_size=2048,
            min_batch_size=1,
            test_steps=2,
            monitor_memory_usage=True,
            clear_cache_between_attempts=True
        )
        
        scaler = BatchSizeScaler(batch_config)
        result = scaler.find_optimal_batch_size(trainer)
        
        print(f"✓ Found batch size under memory constraints: {result.optimal_batch_size}")
        print(f"✓ Peak memory usage: {result.memory_usage_peak_mb:.2f} MB")
        
        # Should find a reasonable batch size (not 1)
        assert result.optimal_batch_size > 1
        assert result.optimal_batch_size <= batch_config.max_batch_size


def test_batch_size_warmup_schedule():
    """Test batch size warmup schedule creation."""
    print("Testing batch size warmup schedule...")
    
    from trainer.batch_size_scaler import create_batch_size_warmup_schedule
    
    # Test normal warmup
    schedule = create_batch_size_warmup_schedule(
        target_batch_size=128,
        warmup_steps=5,
        warmup_factor=0.5
    )
    
    print(f"✓ Warmup schedule: {schedule}")
    
    assert len(schedule) == 5
    assert schedule[0] == 64  # 50% of 128
    assert schedule[-1] == 128  # Final target
    assert all(schedule[i] <= schedule[i+1] for i in range(len(schedule)-1))  # Monotonic
    
    # Test edge cases
    single_step = create_batch_size_warmup_schedule(100, 1)
    assert single_step == [100]
    
    zero_steps = create_batch_size_warmup_schedule(100, 0)
    assert zero_steps == [100]


if __name__ == "__main__":
    print("🧪 Testing Enhanced Batch Size Scaling\n")
    
    test_binary_search_strategy()
    print()
    
    test_exponential_strategy()
    print()
    
    test_conservative_strategy()
    print()
    
    test_fit_with_auto_batch_size()
    print()
    
    test_batch_size_scaling_with_memory_constraints()
    print()
    
    test_batch_size_warmup_schedule()
    print()
    
    print("🎉 All enhanced batch size scaling tests passed!")
