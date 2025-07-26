#!/usr/bin/env python3
"""
Extract all test code from src/ files and move to tests/ directory.
This script ensures no test code remains in src/ files.
"""

import os
import re
import shutil
from pathlib import Path

# Base paths
src_path = Path("/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src")
tests_path = Path("/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/tests")

def extract_test_module(file_path):
    """Extract test module from a file and return the test code and cleaned source code."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find #[cfg(test)] mod tests { ... } blocks with proper brace matching
    test_blocks = []
    cleaned_content = content
    
    # Look for #[cfg(test)] patterns - handle various whitespace and variations
    # Including doc comments with test code
    cfg_patterns = [
        r'#\[cfg\(test\)\]\s*mod\s+tests\s*\{',
        r'#\[cfg\(test\)\]\s*pub\s+mod\s+tests\s*\{',
        r'#\[cfg\(test\)\]\s*mod\s+test\s*\{',
        r'#\[cfg\(test\)\]\s*pub\s+mod\s+test\s*\{',
        r'/// .*?#\[cfg\(test\)\]\s*\n.*?/// .*?mod\s+tests\s*\{',  # Doc comment tests
    ]
    
    found_positions = set()  # Track positions to avoid duplicates
    for cfg_pattern in cfg_patterns:
        for match in re.finditer(cfg_pattern, content):
            start_pos = match.start()
            
            # Skip if we already found a test block at this position
            if start_pos in found_positions:
                continue
                
            brace_pos = match.end() - 1  # Position of opening brace
            
            # Find the matching closing brace
            brace_count = 1
            pos = brace_pos + 1
            
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                # Found complete test block
                test_code = content[start_pos:pos]
                test_blocks.append((start_pos, pos, test_code))
                found_positions.add(start_pos)
    
    # Remove test blocks from content (in reverse order to maintain indices)
    for start_pos, end_pos, test_code in reversed(test_blocks):
        cleaned_content = cleaned_content[:start_pos] + cleaned_content[end_pos:]
    
    # Clean up extra whitespace
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
    cleaned_content = cleaned_content.rstrip() + '\n'
    
    if test_blocks:
        return [block[2] for block in test_blocks], cleaned_content
    else:
        return None, content

def convert_test_to_integration(test_code, module_path):
    """Convert inline test module to integration test."""
    # Remove #[cfg(test)] and mod tests wrapper
    test_code = re.sub(r'#\[cfg\(test\)\]\s*', '', test_code)
    test_code = re.sub(r'mod\s+tests\s*\{\s*', '', test_code)
    test_code = test_code.rstrip().rstrip('}')
    
    # Add proper imports based on module path
    module_parts = module_path.relative_to(src_path).with_suffix('').parts
    
    # Generate use statements
    use_statements = []
    if len(module_parts) > 1:
        crate_path = "::".join(module_parts)
        use_statements.append(f"use fluent_ai_candle::{crate_path}::*;")
    else:
        use_statements.append("use fluent_ai_candle::*;")
    
    # Look for super:: imports and convert them
    super_imports = re.findall(r'use super::(.*?);', test_code)
    for super_import in super_imports:
        if len(module_parts) > 1:
            parent_path = "::".join(module_parts[:-1])
            use_statements.append(f"use fluent_ai_candle::{parent_path}::{super_import};")
        else:
            use_statements.append(f"use fluent_ai_candle::{super_import};")
    
    # Remove super:: references
    test_code = re.sub(r'use super::.*?;', '', test_code)
    test_code = re.sub(r'super::', 'fluent_ai_candle::', test_code)
    
    # Combine imports and test code
    final_code = '\n'.join(use_statements) + '\n\n' + test_code.strip()
    
    return final_code

def process_file(file_path):
    """Process a single file to extract and move tests."""
    print(f"Processing {file_path}...")
    
    test_blocks, cleaned_content = extract_test_module(file_path)
    
    if not test_blocks:
        return False
    
    # Generate test file name
    rel_path = file_path.relative_to(src_path)
    test_name = str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_') + '_tests.rs'
    test_file_path = tests_path / test_name
    
    # Convert test blocks to integration tests
    integration_test_code = ""
    for test_block in test_blocks:
        converted = convert_test_to_integration(test_block, file_path)
        integration_test_code += converted + "\n\n"
    
    # Write integration test file
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(integration_test_code.strip() + '\n')
    
    # Update source file (remove test code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"  → Created {test_file_path}")
    return True

def main():
    """Main function to process all files."""
    # Ensure tests directory exists
    tests_path.mkdir(exist_ok=True)
    
    # Find all .rs files in src/
    rs_files = list(src_path.rglob("*.rs"))
    
    processed_count = 0
    for file_path in rs_files:
        if process_file(file_path):
            processed_count += 1
    
    print(f"\nProcessed {processed_count} files with test code.")
    print("All test code has been moved to tests/ directory.")

if __name__ == "__main__":
    main()