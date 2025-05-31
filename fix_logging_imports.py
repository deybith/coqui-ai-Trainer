#!/usr/bin/env python3
"""
Script to fix logging module imports specifically.
"""
import os
import re
from pathlib import Path

def fix_logging_imports():
    """Fix specific logging module import issues."""
    logging_dir = Path("src/trainer/logging")
    
    # Define replacement patterns for logging module
    patterns = [
        (r'from \._types import', r'from .._types import'),
        (r'from \.config import', r'from ..config import'),
        (r'from \.logging\.base_dash_logger import', r'from .base_dash_logger import'),
        (r'from \.logging\.tensorboard_logger import', r'from .tensorboard_logger import'),
        (r'from \.utils\.distributed import', r'from ..utils.distributed import'),
    ]
    
    for py_file in logging_dir.glob("*.py"):
        print(f"Processing: {py_file}")
        if py_file.name == "__init__.py":
            print("  Skipping __init__.py")
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply all patterns
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed imports in {py_file}")
            else:
                print(f"ℹ️  No changes needed in {py_file}")
                
        except Exception as e:
            print(f"❌ Error processing {py_file}: {e}")

if __name__ == "__main__":
    fix_logging_imports()
