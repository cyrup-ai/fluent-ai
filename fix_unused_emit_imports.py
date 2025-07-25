#!/usr/bin/env python3
"""
Script to remove unused emit imports across the fluent-ai codebase.
This fixes the widespread unused import warnings for fluent_ai_async::emit.
"""

import os
import re
import subprocess
from pathlib import Path

def find_files_with_emit_import():
    """Find all Rust files that import emit from fluent_ai_async."""
    result = subprocess.run(
        ["find", ".", "-name", "*.rs", "-exec", "grep", "-l", "use.*emit", "{}", ";"],
        cwd="/Volumes/samsung_t9/fluent-ai",
        capture_output=True,
        text=True
    )
    return [f.strip() for f in result.stdout.split('\n') if f.strip()]

def check_emit_macro_usage(file_path):
    """Check if a file actually uses the emit! macro."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for emit! macro calls
            return 'emit!' in content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return True  # Conservative - assume it's used if we can't read

def remove_unused_emit_import(file_path):
    """Remove unused emit import from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Remove 'emit' from compound import
        # use fluent_ai_async::{AsyncStream, emit, handle_error}; -> use fluent_ai_async::{AsyncStream, handle_error};
        content = re.sub(r'(use fluent_ai_async::\{[^}]*),\s*emit,\s*([^}]*\})', r'\1, \2', content)
        content = re.sub(r'(use fluent_ai_async::\{)\s*emit,\s*([^}]*\})', r'\1\2', content)
        content = re.sub(r'(use fluent_ai_async::\{[^}]*),\s*emit\s*(\})', r'\1\2', content)
        
        # Pattern 2: Remove standalone emit import
        # use fluent_ai_async::emit;
        content = re.sub(r'use fluent_ai_async::emit;\s*\n', '', content)
        
        # Pattern 3: Handle cases where emit is the only import in braces
        # use fluent_ai_async::{emit}; -> remove entire line
        content = re.sub(r'use fluent_ai_async::\{\s*emit\s*\};\s*\n', '', content)
        
        # Clean up double commas and empty braces
        content = re.sub(r',\s*,', ',', content)
        content = re.sub(r'\{\s*,', '{', content)
        content = re.sub(r',\s*\}', '}', content)
        content = re.sub(r'use fluent_ai_async::\{\s*\};', '', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Removed unused emit import from {file_path}")
            return True
        else:
            print(f"🔍 No emit import found in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to remove unused emit imports."""
    print("🔍 Finding files with emit imports...")
    files = find_files_with_emit_import()
    
    print(f"📁 Found {len(files)} files with emit imports")
    
    fixed_count = 0
    skipped_count = 0
    
    for file_path in files:
        full_path = f"/Volumes/samsung_t9/fluent-ai/{file_path.lstrip('./')}"
        
        # Check if the file actually uses emit! macro
        if check_emit_macro_usage(full_path):
            print(f"⚠️  Skipping {file_path} - emit! macro is actually used")
            skipped_count += 1
            continue
        
        # Remove unused import
        if remove_unused_emit_import(full_path):
            fixed_count += 1
        
    print(f"\n🎉 Summary:")
    print(f"   ✅ Fixed: {fixed_count} files")
    print(f"   ⚠️  Skipped: {skipped_count} files (emit! macro in use)")
    print(f"   📁 Total processed: {len(files)} files")

if __name__ == "__main__":
    main()