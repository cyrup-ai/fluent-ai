#!/usr/bin/env python3
"""
Simple and safe global find/replace for async_stream_channel -> AsyncStream::with_channel
"""

import os
from pathlib import Path

def safe_replace_in_file(file_path: Path) -> bool:
    """Safely replace async_stream_channel with AsyncStream::with_channel in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Simple string replacement
        content = content.replace('async_stream_channel', 'AsyncStream::with_channel')
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"✓ Updated {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    """Find and replace in all Rust files."""
    root_dir = Path("/Volumes/samsung_t9/fluent-ai")
    
    if not root_dir.exists():
        print(f"❌ Root directory not found: {root_dir}")
        return 1
    
    print("🔄 Performing global find/replace: async_stream_channel → AsyncStream::with_channel")
    
    modified_count = 0
    total_files = 0
    
    # Find all .rs files
    for rust_file in root_dir.rglob("*.rs"):
        total_files += 1
        if safe_replace_in_file(rust_file):
            modified_count += 1
    
    print(f"\n✅ Processed {total_files} files, modified {modified_count} files")
    return 0

if __name__ == "__main__":
    exit(main())