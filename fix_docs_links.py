#!/usr/bin/env python3
"""
Fix documentation links to reflect the new src/trainer/ structure
"""

import os
import re
from pathlib import Path

def fix_trainer_paths_in_file(file_path):
    """Fix trainer/ paths to src/trainer/ in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace trainer/ with src/trainer/ in various contexts
        patterns = [
            # File paths in backticks or quotes
            (r'`trainer/', '`src/trainer/'),
            (r'"trainer/', '"src/trainer/'),
            (r"'trainer/", "'src/trainer/"),
            
            # Directory paths
            (r'(\s)trainer/([a-zA-Z_])', r'\1src/trainer/\2'),
            (r'^trainer/([a-zA-Z_])', r'src/trainer/\1'),
            
            # But don't replace things like "coqui-ai-Trainer/"
            # This should be handled by making patterns more specific above
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all documentation files"""
    
    # Find all markdown files
    project_root = Path('/home/ubuntu/projects/coqui-ai-Trainer')
    md_files = list(project_root.rglob('*.md'))
    
    print(f"Found {len(md_files)} markdown files to check...")
    
    fixed_count = 0
    checked_count = 0
    
    for md_file in md_files:
        # Skip files that shouldn't be modified
        if any(skip in str(md_file) for skip in ['.git/', '__pycache__/', '.pytest_cache/']):
            continue
            
        checked_count += 1
        if fix_trainer_paths_in_file(md_file):
            fixed_count += 1
    
    print(f"\n--- Documentation Link Fix Complete ---")
    print(f"Files checked: {checked_count}")
    print(f"Files fixed: {fixed_count}")

if __name__ == "__main__":
    main()
