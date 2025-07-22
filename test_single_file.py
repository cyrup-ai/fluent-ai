#!/usr/bin/env python3
"""
Test the fix on a single file to debug the issue.
"""

import re

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
            print(f"  -> Replacing line {i+1}: {line.strip()}")
    
    if modified:
        content = '\n'.join(lines)
    
    return content

# Test on resolver.rs
filepath = '/Volumes/samsung_t9/fluent-ai/packages/domain/src/model/resolver.rs'
print(f"Testing {filepath}")

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

original_content = content
print(f"Original content length: {len(content)}")
print(f"Contains async_stream_channel: {'async_stream_channel' in content}")

# Fix imports
content = re.sub(
    r'use fluent_ai_async::\{AsyncStream, async_stream_channel\};',
    r'use fluent_ai_async::AsyncStream;',
    content
)

print(f"After import fix, contains async_stream_channel: {'async_stream_channel' in content}")

content = fix_streaming_pattern(content)

if content != original_content:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {filepath}")
else:
    print(f"No changes needed for {filepath}")