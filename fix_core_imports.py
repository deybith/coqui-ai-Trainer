#!/usr/bin/env python3
"""
Script to fix core module imports.
"""
import os
import re
from pathlib import Path

def fix_core_imports():
    """Fix specific core module import issues."""
    core_dir = Path("src/trainer/core")
    
    for py_file in core_dir.glob("*.py"):
        print(f"Processing: {py_file}")
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace relative imports that should go up one level
            patterns = [
                (r'from \._types import', r'from .._types import'),
                (r'from \.config import', r'from ..config import'),
                (r'from \.generic_utils import', r'from ..generic_utils import'),
                (r'from \.io import', r'from ..io import'),
                (r'from \.logging import', r'from ..logging import'),
                (r'from \.logging\.base_dash_logger import', r'from ..logging.base_dash_logger import'),
                (r'from \.model import', r'from ..model import'),
                (r'from \.trainer_utils import', r'from ..trainer_utils import'),
                (r'from \.utils\.distributed import', r'from ..utils.distributed import'),
                (r'from \.utils\.tpu import', r'from ..utils.tpu import'),
            ]
            
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
    fix_core_imports()
