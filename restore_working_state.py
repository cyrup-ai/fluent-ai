#!/usr/bin/env python3
"""
Restore working state by replacing TODO comments with working async_stream_channel() calls.
This gets us back to a compiling state so we can work systematically.
"""

import os
import re

def restore_working_state(content):
    """
    Replace TODO comments with working async_stream_channel() calls
    """
    # Replace TODO comments with the actual channel call
    content = re.sub(
        r'(\s*)// TODO: Convert async_stream_channel to AsyncStream::with_channel pattern',
        r'\1let (sender, stream) = async_stream_channel();',
        content
    )
    
    return content

def process_file(filepath):
    """Process a single Rust file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        content = restore_working_state(content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Restored: {filepath}")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Main function."""
    packages_dir = "/Volumes/samsung_t9/fluent-ai/packages"
    
    # Find all Rust files
    rust_files = []
    for root, dirs, files in os.walk(packages_dir):
        for file in files:
            if file.endswith('.rs'):
                rust_files.append(os.path.join(root, file))
    
    print(f"Restoring working state in {len(rust_files)} files...")
    
    restored_count = 0
    for filepath in rust_files:
        if "TODO: Convert async_stream_channel" in open(filepath, 'r').read():
            process_file(filepath)
            restored_count += 1
    
    print(f"Restored {restored_count} files to working state")

if __name__ == "__main__":
    main()