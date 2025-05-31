#!/usr/bin/env python3
"""
Script to fix absolute imports to relative imports in the trainer module.
"""
import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix absolute trainer imports to relative imports in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace absolute imports with relative imports
        # Pattern: from trainer.xxx import yyy
        content = re.sub(r'from trainer\.', 'from .', content)
        
        # Pattern: import trainer.xxx
        content = re.sub(r'import trainer\.([^\s]+)', r'from . import \1', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in {file_path}")
            return True
        else:
            print(f"ℹ️  No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Fix all absolute imports in the trainer module."""
    trainer_dir = Path("src/trainer")
    if not trainer_dir.exists():
        print("❌ src/trainer directory not found!")
        return
    
    print("🔧 Fixing absolute imports to relative imports...")
    
    fixed_count = 0
    total_count = 0
    
    # Process all Python files in the trainer module
    for py_file in trainer_dir.rglob("*.py"):
        print(f"Processing: {py_file}")
        total_count += 1
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files processed: {total_count}")
    print(f"   Files modified: {fixed_count}")
    print(f"   Files unchanged: {total_count - fixed_count}")

if __name__ == "__main__":
    main()
