#!/usr/bin/env python3
"""
Comprehensive script to fix all import issues after reorganization.
"""
import os
import re
from pathlib import Path

def fix_all_imports():
    """Fix all import issues in the trainer module."""
    
    # Define the fixes needed for different subdirectories
    fixes = {
        # Core module needs to import from parent trainer
        'src/trainer/core': [
            (r'from \.logging\.base_dash_logger import', r'from ..logging.base_dash_logger import'),
            (r'from \.logging import', r'from ..logging import'),
        ],
        
        # Utils module - no changes needed for internal imports
        'src/trainer/utils': [
            # These are already correct
        ],
        
        # Logging module - already fixed
        'src/trainer/logging': [
            # These are already correct
        ]
    }
    
    # Apply fixes directory by directory
    for directory, patterns in fixes.items():
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Directory {directory} not found, skipping")
            continue
            
        print(f"Processing directory: {directory}")
        
        for py_file in dir_path.glob("*.py"):
            print(f"  Processing file: {py_file}")
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply all patterns for this directory
                for pattern, replacement in patterns:
                    content = re.sub(pattern, replacement, content)
                
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"    ✅ Fixed imports in {py_file}")
                else:
                    print(f"    ℹ️  No changes needed in {py_file}")
                    
            except Exception as e:
                print(f"    ❌ Error processing {py_file}: {e}")

def test_import():
    """Test if the trainer module can be imported."""
    try:
        import sys
        sys.path.insert(0, '.')
        import src.trainer
        print("✅ Trainer module import successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing remaining import issues...")
    fix_all_imports()
    print("\n🧪 Testing import...")
    test_import()
