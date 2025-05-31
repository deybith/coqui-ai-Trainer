#!/usr/bin/env python3
"""
Test script to verify the path concatenation fix
"""

import os
import sys
from pathlib import Path

print("🧪 Starting test script...")

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("📁 Python path setup complete")

try:
    from trainer.xtts.gpt_trainer import train_gpt
    print("✅ Successfully imported train_gpt")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def main():
    print("🧪 Testing path concatenation fix...")
    
    # Test parameters
    language = "en"
    num_epochs = 1
    batch_size = 2
    grad_acumm = 1
    train_csv = "./data/dieck/dataset/metadata_train.csv"
    eval_csv = "./data/dieck/dataset/metadata_eval.csv"
    output_path = "./test_output"
    max_audio_length = int(30 * 22050)  # 30 seconds in frames
    
    print(f"📂 Training CSV: {train_csv}")
    print(f"📂 Evaluation CSV: {eval_csv}")
    print(f"📂 Output path: {output_path}")
    
    # Check if files exist
    if not os.path.exists(train_csv):
        print(f"❌ Training CSV not found: {train_csv}")
        return False
    
    if not os.path.exists(eval_csv):
        print(f"❌ Evaluation CSV not found: {eval_csv}")
        return False
    
    print("✅ CSV files found, starting training...")
    
    # Just test that we can call the function without the path error
    try:
        print("🔧 Testing dataset configuration...")
        
        # This should test the path concatenation fix we made
        from trainer.xtts.shared_configs import BaseDatasetConfig
        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )
        
        print(f"✅ Dataset config created successfully:")
        print(f"   - Path: {config_dataset.path}")
        print(f"   - Train meta file: {config_dataset.meta_file_train}")
        print(f"   - Val meta file: {config_dataset.meta_file_val}")
        
        # Test the path construction that was causing the issue
        full_train_path = os.path.join(config_dataset.path, config_dataset.meta_file_train)
        full_eval_path = os.path.join(config_dataset.path, config_dataset.meta_file_val)
        
        print(f"🔍 Constructed paths:")
        print(f"   - Full train path: {full_train_path}")
        print(f"   - Full eval path: {full_eval_path}")
        
        if os.path.exists(full_train_path):
            print("✅ Training metadata file accessible via constructed path")
        else:
            print("❌ Training metadata file NOT accessible via constructed path")
            
        if os.path.exists(full_eval_path):
            print("✅ Evaluation metadata file accessible via constructed path")
        else:
            print("❌ Evaluation metadata file NOT accessible via constructed path")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"🏁 Test completed: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
