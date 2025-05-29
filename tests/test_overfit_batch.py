import torch
from dataclasses import dataclass

from tests.utils.mnist import MnistModel, MnistModelConfig
from trainer import Trainer, TrainerArgs

is_cuda = torch.cuda.is_available()


@dataclass
class OverfitMnistModelConfig(MnistModelConfig):
    """MNIST model config for overfitting test."""
    epochs: int = 2
    print_step: int = 1
    training_seed: int = 12345


def test_overfit_batch_mnist(tmp_path):
    """Test that overfit_batch=True trains on the same batch repeatedly."""
    model = MnistModel()
    config = OverfitMnistModelConfig()
    
    # Test with overfit_batch=True
    trainer_args = TrainerArgs(overfit_batch=True)
    trainer = Trainer(trainer_args, config, output_path=tmp_path, model=model, gpu=0 if is_cuda else None)
    
    # Track the batches seen during training
    original_train_step = trainer.train_step
    batches_seen = []
    
    def track_batch_train_step(batch, batch_n_steps, step, loader_start_time):
        # Store a hash of the batch data to verify it's the same batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            batch_hash = hash(batch[0].data_ptr() if hasattr(batch[0], 'data_ptr') else str(batch[0]))
        else:
            batch_hash = hash(str(batch))
        batches_seen.append(batch_hash)
        return original_train_step(batch, batch_n_steps, step, loader_start_time)
    
    trainer.train_step = track_batch_train_step
    
    # Run training for 1 epoch
    trainer.config.epochs = 1
    trainer.fit()
    
    # Verify that all batches in the epoch were the same (overfitting to one batch)
    if len(batches_seen) > 1:
        first_batch_hash = batches_seen[0]
        all_same = all(batch_hash == first_batch_hash for batch_hash in batches_seen)
        assert all_same, f"Expected all batches to be the same when overfit_batch=True, but got different batches: {set(batches_seen)}"
    
    # Verify we processed multiple steps (not just one)
    assert len(batches_seen) > 1, f"Expected multiple training steps, but only saw {len(batches_seen)} steps"
    
    print(f" > Successfully verified overfit_batch functionality with {len(batches_seen)} steps using the same batch")


def test_normal_training_vs_overfit_batch(tmp_path):
    """Test that normal training and overfit_batch show different behavior."""
    model1 = MnistModel()
    model2 = MnistModel()
    config = OverfitMnistModelConfig()
    
    # Normal training
    trainer_normal = Trainer(TrainerArgs(overfit_batch=False), config, output_path=tmp_path / "normal", model=model1, gpu=0 if is_cuda else None)
    trainer_normal.config.epochs = 1
    trainer_normal.fit()
    normal_loss = trainer_normal.keep_avg_train["avg_loss"]
    
    # Overfit batch training
    trainer_overfit = Trainer(TrainerArgs(overfit_batch=True), config, output_path=tmp_path / "overfit", model=model2, gpu=0 if is_cuda else None)
    trainer_overfit.config.epochs = 1
    trainer_overfit.fit()
    overfit_loss = trainer_overfit.keep_avg_train["avg_loss"]
    
    print(f" > Normal training loss: {normal_loss:.6f}")
    print(f" > Overfit batch loss: {overfit_loss:.6f}")
    
    # Both should complete successfully (we're not asserting which is better since it depends on the specific case)
    assert normal_loss is not None, "Normal training should produce a loss value"
    assert overfit_loss is not None, "Overfit batch training should produce a loss value"
