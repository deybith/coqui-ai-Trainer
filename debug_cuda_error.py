#!/usr/bin/env python3

import torch
import traceback
import sys
import os
from pathlib import Path

# Set CUDA error handling
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def debug_training_step():
    """Run a single training step with detailed error tracking"""
    
    try:
        # Import necessary modules
        sys.path.append('/home/ubuntu/projects/coqui-ai-Trainer')
        
        from trainer.config.config import Config
        from trainer.io import save_checkpoint
        from trainer.model import ModelManager
        from trainer.trainer import Trainer
        from trainer.optimizers import OptimizerManager
        from trainer.config.shared_configs import BaseDatasetConfig
        from trainer.utils.distribute import init_distributed
        
        print("Loading configuration...")
        config_path = "/home/ubuntu/projects/coqui-ai-Trainer/config/demo_config.json"
        config = Config(config_path)
        config.set_output_dir("/home/ubuntu/projects/coqui-ai-Trainer/output/debug_training")
        
        print("Initializing model...")
        model = ModelManager(config, samples=None)
        
        print("Setting up training...")
        trainer = Trainer(
            model=model,
            config=config,
            output_dir=config.output_dir,
            c=config.c
        )
        
        print("Loading data...")
        # Try to get a single batch
        trainer.setup_cuda()
        trainer.setup_loaders(verbose=True)
        
        print("Getting first batch...")
        batch = next(iter(trainer.train_loader))
        
        print("Batch keys:", list(batch.keys()) if isinstance(batch, dict) else "Non-dict batch")
        
        # Print batch details
        if isinstance(batch, dict):
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    if key == 'text_tokens' or 'text' in key:
                        print(f"    text tensor min={value.min()}, max={value.max()}")
                    if key == 'mel_tokens' or 'mel' in key:
                        print(f"    mel tensor min={value.min()}, max={value.max()}")
                else:
                    print(f"  {key}: {type(value)}")
        
        print("\nTrying forward pass...")
        model.train()
        
        # Move batch to device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        print("Running model forward...")
        with torch.cuda.device(0):
            output = model(**batch)
        
        print("Forward pass successful!")
        
    except Exception as e:
        print(f"\nCUDA Error Details:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        # Try to get more CUDA details
        try:
            print(f"\nCUDA Details:")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name()}")
                print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        except:
            pass

if __name__ == "__main__":
    debug_training_step()
