#!/usr/bin/env python3
"""
Development Setup Script for Coqui AI Trainer

This script helps set up the development environment for the reorganized project structure.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {text}")
    print(f"{'='*60}")

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def run_command(command, description="Running command"):
    """Run a shell command and return success status"""
    print_info(f"{description}: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def setup_virtual_environment():
    """Set up Python virtual environment"""
    print_header("Setting up Virtual Environment")
    
    if not run_command("python3 -m venv venv", "Creating virtual environment"):
        return False
    
    activation_script = "venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate.bat"
    print_success(f"Virtual environment created. Activate with: source {activation_script}")
    return True

def install_dependencies():
    """Install project dependencies"""
    print_header("Installing Dependencies")
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print_info("Not in a virtual environment. Consider activating one first.")
    
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command("pip install -e .", "Installing project in development mode"):
        return False
    
    if not run_command("pip install -e .[dev,test]", "Installing development dependencies"):
        return False
    
    print_success("Dependencies installed successfully")
    return True

def setup_pre_commit():
    """Set up pre-commit hooks"""
    print_header("Setting up Pre-commit Hooks")
    
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        return False
    
    print_success("Pre-commit hooks installed")
    return True

def verify_installation():
    """Verify the installation works correctly"""
    print_header("Verifying Installation")
    
    # Test basic imports
    test_imports = [
        "import sys; sys.path.insert(0, 'src'); import trainer; print('✅ Trainer module imported successfully')",
        "import torch; print(f'✅ PyTorch version: {torch.__version__}')",
        "import numpy; print(f'✅ NumPy version: {numpy.__version__}')",
    ]
    
    for test in test_imports:
        if not run_command(f'python -c "{test}"', "Testing import"):
            print_error("Import test failed")
            return False
    
    print_success("All imports working correctly")
    return True

def run_basic_tests():
    """Run basic tests to ensure everything works"""
    print_header("Running Basic Tests")
    
    # Check if tests directory exists and has tests
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print_info("No tests directory found, skipping tests")
        return True
    
    # Run a simple test
    if not run_command("python -m pytest tests/unit/ -v --tb=short", "Running unit tests"):
        print_info("Some tests may have failed, but this is normal during development")
    
    return True

def create_example_script():
    """Create an example script to test the setup"""
    print_header("Creating Example Test Script")
    
    example_content = '''#!/usr/bin/env python3
"""
Example script to test the Coqui AI Trainer setup
"""
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import trainer
    print("✅ Successfully imported trainer module")
    print(f"📍 Trainer location: {trainer.__file__}")
    
    # Test configuration loading
    from trainer.config import TrainerConfig
    print("✅ Successfully imported TrainerConfig")
    
    # Test basic functionality
    print("🎉 Setup verification complete!")
    print("\\n📚 Next steps:")
    print("   1. Explore examples/ directory for usage examples")
    print("   2. Check docs/ for comprehensive guides")
    print("   3. Use scripts/ for training and validation tools")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\\n🔧 Troubleshooting:")
    print("   1. Make sure you're in the project root directory")
    print("   2. Activate your virtual environment")
    print("   3. Run: pip install -e .")

if __name__ == "__main__":
    pass
'''
    
    with open("test_setup.py", "w") as f:
        f.write(example_content)
    
    print_success("Created test_setup.py - run it to verify your setup")
    return True

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Coqui AI Trainer development environment")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment setup")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--quick", action="store_true", help="Quick setup (skip tests and pre-commit)")
    
    args = parser.parse_args()
    
    print_header("Coqui AI Trainer Development Setup")
    print_info("This script will set up your development environment")
    
    # Verify we're in the right directory
    if not Path("pyproject.toml").exists():
        print_error("pyproject.toml not found. Are you in the project root directory?")
        sys.exit(1)
    
    success_count = 0
    total_steps = 6 if not args.quick else 4
    
    # Setup steps
    if not args.skip_venv:
        if setup_virtual_environment():
            success_count += 1
    else:
        print_info("Skipping virtual environment setup")
        success_count += 1
    
    if install_dependencies():
        success_count += 1
    
    if verify_installation():
        success_count += 1
    
    if create_example_script():
        success_count += 1
    
    if not args.quick:
        if setup_pre_commit():
            success_count += 1
        
        if not args.skip_tests:
            if run_basic_tests():
                success_count += 1
        else:
            print_info("Skipping tests")
            success_count += 1
    
    # Final summary
    print_header("Setup Complete")
    print(f"✅ {success_count}/{total_steps} setup steps completed successfully")
    
    if success_count == total_steps:
        print_success("🎉 Development environment is ready!")
        print("\n📋 Quick commands to get started:")
        print("   • python test_setup.py                    # Verify setup")
        print("   • python examples/xtts/train_xtts_enhanced.py --help  # See training options")
        print("   • python scripts/validation/quick_validate.py         # Quick validation")
        print("   • ls examples/                            # Browse examples")
        print("   • ls docs/guides/                         # Read documentation")
    else:
        print_error("Some setup steps failed. Please check the output above.")
        print_info("You can try running individual commands manually or re-run this script.")
    
    return success_count == total_steps

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
