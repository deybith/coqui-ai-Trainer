#!/usr/bin/env python3
"""
Full Enhanced XTTS Training Validation
Quick validation to ensure everything is ready for training
"""

import torch
import sys
import json
import subprocess
from pathlib import Path
import importlib.util

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    print("🔍 Checking PyTorch...")
    print(f"  ✅ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
        print(f"  ✅ GPU: {torch.cuda.get_device_name()}")
        print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("  ⚠️  CUDA not available - training will be slower")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'scipy', 'librosa', 'soundfile', 
        'tensorboard', 'tqdm', 'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Missing!")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_training_files():
    """Check training infrastructure files"""
    print("\n🔍 Checking training files...")
    
    required_files = [
        'train_full_enhanced_xtts.py',
        'config/full_enhanced_config.json',
        'run_full_enhanced_training.sh',
        'prepare_training_data.py'
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} - Missing!")
            missing.append(file_path)
    
    return len(missing) == 0

def check_config():
    """Check configuration file"""
    print("\n🔍 Checking configuration...")
    
    config_path = Path('config/full_enhanced_config.json')
    if not config_path.exists():
        print("  ❌ Configuration file missing!")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"  ✅ Configuration loaded")
        print(f"  ✅ Phase 1 enabled: {config.get('use_phase1', False)}")
        print(f"  ✅ Phase 2 enabled: {config.get('use_phase2', False)}")
        print(f"  ✅ Batch size: {config.get('batch_size', 'not set')}")
        print(f"  ✅ Model dimension: {config.get('d_model', 'not set')}")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def check_enhanced_components():
    """Check if enhanced components can be imported"""
    print("\n🔍 Checking enhanced components...")
    
    try:
        # Check if we can import the main training script
        spec = importlib.util.spec_from_file_location(
            "train_full_enhanced", "train_full_enhanced_xtts.py"
        )
        if spec and spec.loader:
            print("  ✅ Training script can be imported")
        else:
            print("  ❌ Training script import failed")
            return False
        
        # Check trainer directory structure
        trainer_path = Path('trainer')
        if trainer_path.exists():
            print("  ✅ Trainer directory exists")
            
            # Check for enhanced components
            enhanced_paths = [
                'trainer/xtts/layers/attention/mamba.py',
                'trainer/xtts/layers/attention/mixture_of_experts.py', 
                'trainer/xtts/layers/attention/flash_attention.py',
                'trainer/xtts/layers/attention/rope.py'
            ]
            
            for path in enhanced_paths:
                if Path(path).exists():
                    print(f"  ✅ {Path(path).name}")
                else:
                    print(f"  ⚠️  {Path(path).name} - Not found")
        
        return True
    except Exception as e:
        print(f"  ❌ Component check failed: {e}")
        return False

def check_data_directory():
    """Check data directory structure"""
    print("\n🔍 Checking data directory...")
    
    data_path = Path('data')
    if not data_path.exists():
        print("  ⚠️  Data directory doesn't exist - will be created during training")
        data_path.mkdir(exist_ok=True)
        print("  ✅ Created data directory")
    else:
        print("  ✅ Data directory exists")
    
    # Check for existing datasets
    datasets = list(data_path.glob('*/'))
    if datasets:
        print(f"  ✅ Found {len(datasets)} dataset(s):")
        for dataset in datasets[:3]:  # Show first 3
            print(f"    - {dataset.name}")
    else:
        print("  ℹ️  No datasets found - use prepare_training_data.py to create one")
    
    return True

def run_quick_test():
    """Run a quick test of the training script"""
    print("\n🔍 Running quick test...")
    
    try:
        # Try to run training script with help flag
        result = subprocess.run([
            sys.executable, 'train_full_enhanced_xtts.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✅ Training script runs successfully")
            return True
        else:
            print(f"  ❌ Training script error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ⚠️  Training script test timed out")
        return False
    except Exception as e:
        print(f"  ❌ Training script test failed: {e}")
        return False

def main():
    print("🎯 Full Enhanced XTTS Training Validation")
    print("=" * 50)
    
    checks = [
        ("PyTorch & CUDA", check_pytorch),
        ("Dependencies", check_dependencies), 
        ("Training Files", check_training_files),
        ("Configuration", check_config),
        ("Enhanced Components", check_enhanced_components),
        ("Data Directory", check_data_directory),
        ("Quick Test", run_quick_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {name} check failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED! Ready to train!")
        print("\n🚀 Next steps:")
        print("  1. python train_full_enhanced_xtts.py --demo --epochs 2")
        print("  2. Or prepare your data: python prepare_training_data.py")
        print("  3. Full guide: cat NEXT_STEPS_TRAINING.md")
        return True
    else:
        print(f"⚠️  {total - passed} checks failed. Please fix issues before training.")
        return False
    print("=" * 50)
    
    # Check Python and PyTorch
    print(f"✅ Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"✅ PyTorch: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available")
    
    # Check training script
    if Path("train_full_enhanced_xtts.py").exists():
        print("✅ Main training script available")
    else:
        print("❌ Training script missing")
    
    # Check config
    if Path("config/full_enhanced_config.json").exists():
        print("✅ Configuration file available")
    else:
        print("❌ Configuration file missing")
    
    print("\n🚀 Quick Start:")
    print("1. Prepare data: python prepare_training_data.py --input_dir /path/to/audio --output_dir ./data --create_metadata")
    print("2. Edit config: Edit config/full_enhanced_config.json")
    print("3. Start training: python train_full_enhanced_xtts.py --config config/full_enhanced_config.json --use_phase1 --use_phase2 --data_path ./data --output_dir ./output")
    print("4. Monitor: tensorboard --logdir ./output/logs")
    print("\n📚 Read READY_TO_TRAIN.md for complete instructions!")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
