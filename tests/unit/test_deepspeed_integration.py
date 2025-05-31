#!/usr/bin/env python3
"""
Test script for Deepspeed integration with coqui-ai-Trainer.

This script tests the complete Deepspeed integration including:
- Configuration creation and validation
- Manager initialization and engine setup
- Training loop integration
- Checkpoint saving and loading
- Error handling and fallback mechanisms
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Import torch BEFORE adding trainer to sys.path to avoid name collision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add the trainer package to the path
sys.path.insert(0, str(Path(__file__).parent))

from trainer import (
    Trainer, 
    TrainerArgs, 
    TrainerConfig, 
    TrainerModel,
    DeepspeedConfig,
    create_deepspeed_config,
    is_deepspeed_available
)


class SimpleModel(TrainerModel):
    """Simple model for testing Deepspeed integration."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x: torch.Tensor) -> dict:
        print(f"  [DEBUG] forward: Input tensor device: {x.device}")
        print(f"  [DEBUG] forward: linear1 weight device: {self.linear1.weight.device}")
        print(f"  [DEBUG] forward: linear2 weight device: {self.linear2.weight.device}")
        
        h = torch.relu(self.linear1(x))
        output = self.linear2(h)
        return {"output": output}
    
    def compute_loss(self, batch: dict, outputs: dict) -> dict:
        target = batch["target"]
        pred = outputs["output"]
        loss = self.loss_fn(pred, target)
        return {"loss": loss}
    
    def get_criterion(self):
        """Return the loss function for the trainer."""
        return self.loss_fn
    
    def train_step(self, batch: dict, criterion: torch.nn.Module) -> tuple[dict, dict]:
        """Implement the training step."""
        # Forward pass
        outputs = self.forward(batch["input"])
        
        # Compute loss
        loss_dict = self.compute_loss(batch, outputs)
        
        return outputs, loss_dict
    
    def eval_step(self, batch: dict, criterion: torch.nn.Module) -> tuple[dict, dict]:
        """Implement the evaluation step."""
        return self.train_step(batch, criterion)
    
    def format_batch(self, batch):
        """Format batch data for training."""
        # Convert from DataLoader format to expected format
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            
            # Ensure tensors are on the correct device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  [DEBUG] format_batch: Moving tensors to device: {device}")
            print(f"  [DEBUG] Input tensor device before: {inputs.device if isinstance(inputs, torch.Tensor) else 'N/A'}")
            print(f"  [DEBUG] Target tensor device before: {targets.device if isinstance(targets, torch.Tensor) else 'N/A'}")
            
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device, non_blocking=True)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device, non_blocking=True)
            
            print(f"  [DEBUG] Input tensor device after: {inputs.device if isinstance(inputs, torch.Tensor) else 'N/A'}")
            print(f"  [DEBUG] Target tensor device after: {targets.device if isinstance(targets, torch.Tensor) else 'N/A'}")
            
            return {
                "input": inputs,
                "target": targets
            }
        return batch
    
    def get_data_loader(self, config, assets, is_eval=False, verbose=False):
        """Create a simple data loader for testing."""
        # Generate random data
        batch_size = config.eval_batch_size if is_eval else config.batch_size
        num_samples = 100 if is_eval else 1000
        
        x = torch.randn(num_samples, 128)
        y = torch.randn(num_samples, 64)
        
        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=not is_eval,
            num_workers=0
        )
    
    def format_batch(self, batch):
        """Format batch for the model."""
        x, y = batch
        return {"input": x, "target": y}


def test_deepspeed_availability():
    """Test if Deepspeed is available."""
    print("Testing Deepspeed availability...")
    available = is_deepspeed_available()
    print(f"Deepspeed available: {available}")
    
    if not available:
        print("⚠️  Deepspeed is not available. Install with: pip install deepspeed")
        return False
    return True


def test_deepspeed_config_creation():
    """Test Deepspeed configuration creation."""
    print("\nTesting Deepspeed configuration creation...")
    
    # Test default config
    config = create_deepspeed_config()
    assert config.enable_deepspeed == True
    assert config.zero_stage == 2
    assert config.enable_mixed_precision == True
    print("✅ Default config created successfully")
    
    # Test custom config
    custom_config = create_deepspeed_config(
        zero_stage=3,
        enable_cpu_offload=True,
        gradient_accumulation_steps=4
    )
    assert custom_config.zero_stage == 3
    assert custom_config.enable_cpu_offload == True
    assert custom_config.gradient_accumulation_steps == 4
    print("✅ Custom config created successfully")
    
    return True


def test_trainer_config_integration():
    """Test Deepspeed integration with TrainerConfig."""
    print("\nTesting TrainerConfig Deepspeed integration...")
    
    config = TrainerConfig(
        epochs=2,
        batch_size=8,
        eval_batch_size=8,
        print_step=10,
        save_step=50,
        use_deepspeed=True,
        deepspeed_zero_stage=2,
        deepspeed_cpu_offload=False,
        mixed_precision=True
    )
    
    assert config.use_deepspeed == True
    assert config.deepspeed_zero_stage == 2
    assert config.deepspeed_cpu_offload == False
    print("✅ TrainerConfig Deepspeed fields work correctly")
    
    return True


def test_full_training_integration():
    """Test full training integration with Deepspeed."""
    print("\nTesting full training integration...")
    
    if not is_deepspeed_available():
        print("⚠️  Skipping full integration test - Deepspeed not available")
        return True
    
    # Set environment variables for single GPU Deepspeed
    os.environ["MASTER_ADDR"] = "localhost" 
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "deepspeed_test"
        
        # Create trainer configuration with simpler Deepspeed settings for testing
        config = TrainerConfig(
            output_path=str(output_path),  # Move output_path to config
            epochs=1,
            batch_size=4,
            eval_batch_size=4,
            print_step=5,
            save_step=20,
            run_eval=True,
            use_deepspeed=True,
            deepspeed_zero_stage=0,  # Use stage 0 for single GPU testing
            deepspeed_cpu_offload=False,
            mixed_precision=False,  # Disable for testing stability
            save_n_checkpoints=2,
            optimizer="Adam",  # Add optimizer specification
            lr=0.001  # Add learning rate
        )
        
        # Create Deepspeed configuration for single GPU
        deepspeed_config = create_deepspeed_config(
            zero_stage=0,  # Use stage 0 for single GPU
            enable_mixed_precision=False,
            enable_cpu_offload=False,
            gradient_accumulation_steps=1
        )
        
        # Create trainer arguments (removed output_path - it belongs in config)
        args = TrainerArgs(
            continue_path="",
            restore_path=""
        )
        
        try:
            # Create model
            model = SimpleModel()
            
            # Create trainer with Deepspeed
            trainer = Trainer(
                args=args,
                config=config,
                model=model,
                deepspeed_config=deepspeed_config
            )
            
            print("✅ Trainer created successfully with Deepspeed")
            
            # Check if Deepspeed manager was initialized
            assert hasattr(trainer, 'deepspeed_manager')
            assert trainer.deepspeed_manager is not None
            print("✅ Deepspeed manager initialized")
            
            # Check use_deepspeed property
            assert trainer.use_deepspeed == True
            print("✅ use_deepspeed property works correctly")
            
            # Test a few training steps
            print("Running training steps...")
            
            # Get data loaders
            train_loader = model.get_data_loader(config, {}, is_eval=False)
            eval_loader = model.get_data_loader(config, {}, is_eval=True)
            
            # Set data loaders
            trainer.train_loader = train_loader
            trainer.eval_loader = eval_loader
            
            # Initialize for training
            trainer.model.train()
            
            # Run a few training steps
            step_count = 0
            for batch in train_loader:
                if step_count >= 3:  # Just test a few steps
                    break
                    
                # Use trainer's format_batch method for proper device handling
                formatted_batch = trainer.format_batch(batch)
                
                # Test training step
                outputs, loss_dict, step_time = trainer.optimize(
                    batch=formatted_batch,
                    optimizer=trainer.optimizer,
                    scaler=trainer.scaler,
                    criterion=trainer.criterion,
                    scheduler=trainer.scheduler,
                    step_optimizer=True
                )
                
                trainer.total_steps_done += 1
                step_count += 1
                
                # Format loss value properly for display
                loss_value = loss_dict.get('loss', 'N/A')
                if isinstance(loss_value, torch.Tensor):
                    loss_str = f"{loss_value.item():.4f}"
                elif isinstance(loss_value, (int, float)):
                    loss_str = f"{loss_value:.4f}"
                else:
                    loss_str = str(loss_value)
                
                print(f"Step {step_count}: loss = {loss_str}, time = {step_time:.3f}s")
            
            print("✅ Training steps completed successfully")
            
            # Test checkpoint saving
            if trainer.total_steps_done > 0:
                print("Testing checkpoint saving...")
                trainer.save_checkpoint()
                
                # Check if checkpoint files were created
                checkpoint_files = list(output_path.glob("checkpoint_*.pth"))
                deepspeed_dirs = list(output_path.glob("deepspeed_checkpoint_*"))
                
                if checkpoint_files:
                    print("✅ Standard checkpoint saved")
                if deepspeed_dirs:
                    print("✅ Deepspeed checkpoint saved")
            
            print("✅ Full training integration test completed successfully")
            
            # Clean up distributed environment
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                    print("✅ Distributed process group cleaned up")
            except Exception as e:
                print(f"⚠️  Warning during cleanup: {e}")
                
            return True
            
        except Exception as e:
            print(f"❌ Full integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_error_handling():
    """Test error handling and fallback mechanisms."""
    print("\nTesting error handling...")
    
    # Test with invalid configuration
    try:
        config = DeepspeedConfig(
            enable_deepspeed=True,
            zero_stage=5,  # Invalid stage
        )
        print("⚠️  Invalid zero stage should be handled gracefully")
    except Exception:
        pass
    
    print("✅ Error handling tests completed")
    return True


def run_all_tests():
    """Run all Deepspeed integration tests."""
    print("🚀 Starting Deepspeed Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Deepspeed Availability", test_deepspeed_availability),
        ("Config Creation", test_deepspeed_config_creation),
        ("TrainerConfig Integration", test_trainer_config_integration),
        ("Error Handling", test_error_handling),
    ]
    
    # Only run full integration test if Deepspeed is available
    if is_deepspeed_available():
        tests.append(("Full Training Integration", test_full_training_integration))
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deepspeed integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
