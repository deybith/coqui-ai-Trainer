#!/usr/bin/env python3
"""
Full Enhanced XTTS Quick Start and Validation

This script performs a complete validation of the full enhanced XTTS training setup
and provides guidance for getting started with training.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import torch
import json

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_status(message: str, status: str = "info"):
    """Print colored status messages."""
    color_map = {
        "success": Colors.OKGREEN,
        "error": Colors.FAIL,
        "warning": Colors.WARNING,
        "info": Colors.OKBLUE,
        "header": Colors.HEADER + Colors.BOLD
    }
    color = color_map.get(status, Colors.ENDC)
    print(f"{color}{message}{Colors.ENDC}")

def check_python_environment():
    """Check Python environment and dependencies."""
    print_status("🐍 Checking Python Environment", "header")
    
    # Check Python version
    python_version = sys.version_info
    print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print_status("❌ Python 3.8+ required", "error")
        return False
    else:
        print_status("✅ Python version OK", "success")
    
    # Check critical dependencies
    dependencies = [
        ("torch", "2.0.0"),
        ("torchaudio", "2.0.0"),
        ("transformers", "4.30.0"),
        ("numpy", "1.21.0"),
        ("tensorboard", None),
        ("tqdm", None)
    ]
    
    missing_deps = []
    for dep_name, min_version in dependencies:
        try:
            module = importlib.import_module(dep_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   {dep_name}: {version}")
            
            if min_version and version != 'unknown':
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    print_status(f"⚠️  {dep_name} version {min_version}+ recommended", "warning")
            
        except ImportError:
            missing_deps.append(dep_name)
            print_status(f"❌ {dep_name}: Not installed", "error")
    
    if missing_deps:
        print_status(f"❌ Missing dependencies: {', '.join(missing_deps)}", "error")
        print("   Install with: pip install torch torchaudio transformers tensorboard tqdm numpy")
        return False
    
    print_status("✅ All dependencies available", "success")
    return True

def check_gpu_setup():
    """Check GPU setup and CUDA availability."""
    print_status("\n🖥️  Checking GPU Setup", "header")
    
    if not torch.cuda.is_available():
        print_status("❌ CUDA not available", "error")
        print("   GPU training is highly recommended for enhanced XTTS")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"   Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 6:
            print_status(f"⚠️  GPU {i} has limited memory (<6GB)", "warning")
        elif gpu_memory < 12:
            print_status(f"⚠️  GPU {i} may need reduced batch size (<12GB)", "warning")
    
    print_status("✅ CUDA GPU(s) available", "success")
    return True

def check_phase1_components():
    """Check Phase 1 enhanced components."""
    print_status("\n🎵 Checking Phase 1 Components", "header")
    
    components = [
        ("trainer.xtts.models.enhanced_xtts", "EnhancedXtts"),
        ("trainer.xtts.layers.encodec", "create_encodec_for_xtts"),
        ("trainer.xtts.layers.streaming", "create_quality_monitor")
    ]
    
    available_count = 0
    for module_name, component_name in components:
        try:
            module = importlib.import_module(module_name)
            component = getattr(module, component_name)
            print(f"   ✅ {module_name}.{component_name}")
            available_count += 1
        except (ImportError, AttributeError) as e:
            print(f"   ❌ {module_name}.{component_name}: {e}")
    
    if available_count == len(components):
        print_status("✅ All Phase 1 components available", "success")
        return True
    elif available_count > 0:
        print_status(f"⚠️  Partial Phase 1 components ({available_count}/{len(components)})", "warning")
        return True
    else:
        print_status("❌ No Phase 1 components available", "error")
        return False

def check_phase2_components():
    """Check Phase 2 enhanced components."""
    print_status("\n🧠 Checking Phase 2 Components", "header")
    
    components = [
        ("trainer.xtts.layers.xtts.phase2_enhanced_gpt", "Phase2EnhancedXTTSGPT"),
        ("trainer.xtts.layers.attention.phase2_integration", "Phase2Config"),
        ("trainer.xtts.layers.attention.mamba", "MambaBlock"),
        ("trainer.xtts.layers.attention.mixture_of_experts", "MoELayer"),
        ("trainer.xtts.layers.attention.flash_attention", "FlashMultiHeadAttention"),
        ("trainer.xtts.layers.attention.rope", "RoPEAttention")
    ]
    
    available_count = 0
    for module_name, component_name in components:
        try:
            module = importlib.import_module(module_name)
            component = getattr(module, component_name)
            print(f"   ✅ {module_name}.{component_name}")
            available_count += 1
        except (ImportError, AttributeError) as e:
            print(f"   ❌ {module_name}.{component_name}: {e}")
    
    if available_count == len(components):
        print_status("✅ All Phase 2 components available", "success")
        return True
    elif available_count > 0:
        print_status(f"⚠️  Partial Phase 2 components ({available_count}/{len(components)})", "warning")
        return True
    else:
        print_status("❌ No Phase 2 components available", "error")
        return False

def check_training_scripts():
    """Check training scripts availability."""
    print_status("\n📝 Checking Training Scripts", "header")
    
    scripts = [
        "train_full_enhanced_xtts.py",
        "prepare_training_data.py",
        "full_enhanced_training_demo.py"
    ]
    
    all_available = True
    for script in scripts:
        if Path(script).exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script}: Not found")
            all_available = False
    
    if all_available:
        print_status("✅ All training scripts available", "success")
    else:
        print_status("❌ Some training scripts missing", "error")
    
    return all_available

def check_config_files():
    """Check configuration files."""
    print_status("\n⚙️  Checking Configuration Files", "header")
    
    config_dir = Path("config")
    if not config_dir.exists():
        config_dir.mkdir(exist_ok=True)
    
    configs = [
        "config/full_enhanced_config.json",
        "config/enhanced_xtts_config.json"
    ]
    
    all_available = True
    for config_file in configs:
        if Path(config_file).exists():
            print(f"   ✅ {config_file}")
            # Validate JSON
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
                print(f"      Valid JSON format")
            except json.JSONDecodeError as e:
                print(f"      ❌ Invalid JSON: {e}")
                all_available = False
        else:
            print(f"   ❌ {config_file}: Not found")
            all_available = False
    
    if all_available:
        print_status("✅ All configuration files available", "success")
    else:
        print_status("⚠️  Some configuration files missing", "warning")
    
    return all_available

def run_quick_validation():
    """Run a quick validation test."""
    print_status("\n🧪 Running Quick Validation Test", "header")
    
    try:
        # Try importing the main training module
        sys.path.insert(0, '.')
        
        # Test Phase 1 imports
        try:
            from trainer.xtts.models.enhanced_xtts import EnhancedXtts, EnhancedXttsConfig
            print("   ✅ Phase 1 models import successful")
            phase1_ok = True
        except ImportError as e:
            print(f"   ❌ Phase 1 models import failed: {e}")
            phase1_ok = False
        
        # Test Phase 2 imports
        try:
            from trainer.xtts.layers.xtts.phase2_enhanced_gpt import Phase2EnhancedXTTSGPT
            from trainer.xtts.layers.attention.phase2_integration import Phase2Config
            print("   ✅ Phase 2 models import successful")
            phase2_ok = True
        except ImportError as e:
            print(f"   ❌ Phase 2 models import failed: {e}")
            phase2_ok = False
        
        # Test training script import
        try:
            import train_full_enhanced_xtts
            print("   ✅ Training script import successful")
            training_ok = True
        except ImportError as e:
            print(f"   ❌ Training script import failed: {e}")
            training_ok = False
        
        if phase1_ok and phase2_ok and training_ok:
            print_status("✅ Quick validation passed", "success")
            return True
        else:
            print_status("⚠️  Validation completed with issues", "warning")
            return False
            
    except Exception as e:
        print_status(f"❌ Validation error: {e}", "error")
        return False

def generate_usage_guide():
    """Generate a usage guide based on available components."""
    print_status("\n📚 Usage Guide", "header")
    
    print("🚀 Getting Started with Full Enhanced XTTS Training:")
    print()
    
    print("1. 📊 Prepare Your Training Data:")
    print("   python prepare_training_data.py \\")
    print("       --input_dir /path/to/your/audio/files \\")
    print("       --output_dir ./data \\")
    print("       --create_metadata \\")
    print("       --validate")
    print()
    
    print("2. ⚙️  Configure Training:")
    print("   Edit config/full_enhanced_config.json:")
    print("   - Set data_path to your data directory")
    print("   - Adjust batch_size based on GPU memory")
    print("   - Enable/disable Phase 1 and Phase 2 features")
    print()
    
    print("3. 🎯 Start Training:")
    print("   # Full enhanced training (Phase 1 + Phase 2)")
    print("   python train_full_enhanced_xtts.py \\")
    print("       --config config/full_enhanced_config.json \\")
    print("       --use_phase1 --use_phase2 \\")
    print("       --data_path ./data \\")
    print("       --output_dir ./output")
    print()
    
    print("   # Phase 2 only (Advanced neural architecture)")
    print("   python train_full_enhanced_xtts.py \\")
    print("       --config config/full_enhanced_config.json \\")
    print("       --use_phase2 \\")
    print("       --data_path ./data \\")
    print("       --output_dir ./output_phase2")
    print()
    
    print("4. 📈 Monitor Training:")
    print("   tensorboard --logdir ./output/logs")
    print("   # Open browser to http://localhost:6006")
    print()
    
    print("5. 🧪 Run Demo (Optional):")
    print("   python full_enhanced_training_demo.py")
    print()
    
    print("📖 Documentation:")
    print("   - FULL_ENHANCED_TRAINING_GUIDE.md: Comprehensive training guide")
    print("   - PHASE2_COMPLETION_REPORT.md: Phase 2 features documentation")
    print("   - IMPLEMENTATION_COMPLETE.md: Complete implementation status")
    print()

def main():
    """Main validation and setup function."""
    print_status("🎯 Full Enhanced XTTS Training Validation", "header")
    print_status("=" * 50, "header")
    
    # Run all checks
    checks = [
        ("Python Environment", check_python_environment),
        ("GPU Setup", check_gpu_setup),
        ("Phase 1 Components", check_phase1_components),
        ("Phase 2 Components", check_phase2_components),
        ("Training Scripts", check_training_scripts),
        ("Configuration Files", check_config_files),
        ("Quick Validation", run_quick_validation)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_status(f"❌ {check_name} check failed: {e}", "error")
            results[check_name] = False
    
    # Summary
    print_status("\n📋 Validation Summary", "header")
    print_status("=" * 30, "header")
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
    
    print()
    print(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print_status("🎉 All checks passed! Ready for training!", "success")
    elif passed >= total * 0.8:
        print_status("⚠️  Most checks passed. Training should work with some limitations.", "warning")
    else:
        print_status("❌ Multiple issues detected. Please resolve before training.", "error")
    
    # Generate usage guide
    generate_usage_guide()
    
    # Offer to run demo
    if results.get("Training Scripts", False):
        print_status("\n❓ Would you like to run the training demonstration? (y/n): ", "info")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                print_status("🚀 Starting training demonstration...", "info")
                subprocess.run([sys.executable, "full_enhanced_training_demo.py"])
        except KeyboardInterrupt:
            print_status("\n⏹️  Demo cancelled by user.", "info")

if __name__ == "__main__":
    main()
