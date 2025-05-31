#!/usr/bin/env python3
"""
Project Structure Validation Script

This script validates that the reorganized project structure is working correctly.
"""

import sys
import os
from pathlib import Path

def check_directory_structure():
    """Check that all expected directories exist"""
    print("🔍 Checking directory structure...")
    
    required_dirs = [
        "src/trainer",
        "examples/xtts",
        "examples/demos", 
        "scripts/training",
        "scripts/validation",
        "scripts/utilities",
        "configs/xtts",
        "tests/unit",
        "docs/guides",
        "docs/roadmaps",
        "data",
        "output",
        "logs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print(f"✅ All {len(required_dirs)} required directories exist")
        return True

def check_key_files():
    """Check that key files exist in the right places"""
    print("\n📁 Checking key files...")
    
    key_files = [
        "src/trainer/__init__.py",
        "src/trainer/trainer.py",
        "examples/xtts/train_xtts_enhanced.py",
        "scripts/validation/validate_enhanced_xtts.py",
        "configs/xtts/enhanced_xtts_config.json",
        "docs/README.md",
        "PROJECT_STRUCTURE.md",
        "setup_dev.py",
        "Makefile"
    ]
    
    missing_files = []
    for file_path in key_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✅ All {len(key_files)} key files found in correct locations")
        return True

def test_imports():
    """Test that imports work with the new structure"""
    print("\n🐍 Testing Python imports...")
    
    # Add src to path for testing
    sys.path.insert(0, str(Path("src")))
    
    try:
        import trainer
        print("✅ Core trainer module imports successfully")
        
        from trainer.config import TrainerConfig
        print("✅ TrainerConfig imports successfully")
        
        print(f"📍 Trainer module location: {trainer.__file__}")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def validate_makefile():
    """Check that Makefile has essential targets"""
    print("\n🔨 Validating Makefile...")
    
    try:
        with open("Makefile", "r") as f:
            makefile_content = f.read()
        
        required_targets = ["help", "setup", "test", "train-basic", "validate", "clean"]
        missing_targets = []
        
        for target in required_targets:
            if f"{target}:" not in makefile_content:
                missing_targets.append(target)
        
        if missing_targets:
            print(f"❌ Missing Makefile targets: {missing_targets}")
            return False
        else:
            print(f"✅ All {len(required_targets)} essential Makefile targets found")
            return True
            
    except FileNotFoundError:
        print("❌ Makefile not found")
        return False

def check_documentation():
    """Check that documentation is properly organized"""
    print("\n📚 Checking documentation organization...")
    
    doc_categories = [
        ("docs/guides", ["XTTS_ENHANCEMENT_GUIDE.md", "README.md"]),
        ("docs/roadmaps", ["XTTS_PHASE2_ROADMAP.md"]),
        ("PROJECT_STRUCTURE.md", None),
        ("REORGANIZATION_COMPLETE.md", None)
    ]
    
    issues = []
    
    for category, files in doc_categories:
        if files is None:  # Single file check
            if not Path(category).exists():
                issues.append(f"Missing: {category}")
        else:  # Directory with files
            category_path = Path(category)
            if not category_path.exists():
                issues.append(f"Missing directory: {category}")
                continue
            
            for file in files:
                if not (category_path / file).exists():
                    issues.append(f"Missing: {category}/{file}")
    
    if issues:
        print(f"❌ Documentation issues: {issues}")
        return False
    else:
        print("✅ Documentation is properly organized")
        return True

def main():
    """Main validation function"""
    print("🏗️  Project Structure Validation")
    print("=" * 50)
    
    # Change to project root if needed
    if not Path("pyproject.toml").exists():
        print("❌ Not in project root directory (pyproject.toml not found)")
        return False
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Key Files", check_key_files),
        ("Python Imports", test_imports),
        ("Makefile", validate_makefile),
        ("Documentation", check_documentation)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                print(f"❌ {check_name} check failed")
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Project structure validation PASSED!")
        print("\n🚀 Your project is properly organized and ready for development!")
        print("\n📋 Next steps:")
        print("   1. Run: python setup_dev.py")
        print("   2. Try: make help")
        print("   3. Explore: examples/xtts/")
        print("   4. Read: docs/README.md")
        return True
    else:
        print(f"❌ Project structure validation FAILED ({total - passed} issues)")
        print("\n🔧 Please fix the issues above and run the validation again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
