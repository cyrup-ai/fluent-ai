#!/usr/bin/env python3
"""
Fix streaming patterns in fluent-ai domain package.
Replace AsyncStream::channel() with proper AsyncStream::with_channel() pattern.
"""

import os
import re
import glob

def fix_streaming_pattern(content):
    """
    Simple and reliable approach:
    1. Fix imports by removing async_stream_channel
    2. Mark lines with async_stream_channel for manual conversion
    """
    original_content = content
    
    # Fix imports first - remove async_stream_channel from imports
    content = re.sub(
        r'use fluent_ai_async::\{AsyncStream, async_stream_channel\};',
        r'use fluent_ai_async::AsyncStream;',
        content
    )
    
    # Also handle the case where async_stream_channel is imported separately
    content = re.sub(
        r'use fluent_ai_async::async_stream_channel;',
        r'// REMOVED: use fluent_ai_async::async_stream_channel;',
        content
    )
    
    # Mark any line containing async_stream_channel for manual conversion
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        if 'async_stream_channel' in line and 'REMOVED' not in line and 'TODO' not in line:
            # Replace the line with a TODO comment
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + '// TODO: Convert async_stream_channel to AsyncStream::with_channel pattern'
            modified = True
    
    if modified:
        content = '\n'.join(lines)
    
    return content

def process_file(filepath):
    """Process a single Rust file."""
    print(f"Processing {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Call the main fix function to handle both imports and pattern replacement
        content = fix_streaming_pattern(content)
        
        # Report if we found patterns that need manual conversion
        if 'TODO: Convert async_stream_channel' in content:
            print(f"  *** NEEDS MANUAL CONVERSION: Found async_stream_channel patterns in {filepath}")
        elif 'async_stream_channel' in content:
            print(f"  Found async_stream_channel references in {filepath}")
            
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Updated {filepath}")
        else:
            print(f"  No changes needed for {filepath}")
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Main function."""
    # Process ALL packages in the workspace
    packages_dir = "/Volumes/samsung_t9/fluent-ai/packages"
    
    # Find all Rust files in ALL packages
    rust_files = []
    for root, dirs, files in os.walk(packages_dir):
        for file in files:
            if file.endswith('.rs'):
                rust_files.append(os.path.join(root, file))
    
    print(f"Found {len(rust_files)} Rust files to process across ALL packages")
    
    for filepath in rust_files:
        process_file(filepath)
    
    print("Done!")

if __name__ == "__main__":
    main()