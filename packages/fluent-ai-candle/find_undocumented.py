#!/usr/bin/env python3
"""
Script to find undocumented public functions in Rust code.
Looks for 'pub fn' declarations that are not preceded by documentation comments (///).
"""

import os
import re
import sys

def find_undocumented_functions(file_path):
    """Find undocumented public functions in a Rust file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (UnicodeDecodeError, IOError):
        return []
    
    undocumented = []
    
    for i, line in enumerate(lines):
        # Look for 'pub fn' declarations
        if re.search(r'^\s*pub\s+fn\s+\w+', line):
            function_name = re.search(r'pub\s+fn\s+(\w+)', line)
            if function_name:
                func_name = function_name.group(1)
                
                # Check if there's documentation in the previous 3 lines
                has_docs = False
                for j in range(max(0, i-3), i):
                    if '///' in lines[j]:
                        has_docs = True
                        break
                
                if not has_docs:
                    undocumented.append({
                        'file': file_path,
                        'line': i + 1,
                        'function': func_name,
                        'code': line.strip()
                    })
    
    return undocumented

def main():
    src_dir = "src"
    if not os.path.exists(src_dir):
        print("src directory not found")
        return
    
    all_undocumented = []
    
    # Walk through all .rs files
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.rs'):
                file_path = os.path.join(root, file)
                undocumented = find_undocumented_functions(file_path)
                all_undocumented.extend(undocumented)
    
    # Sort by file path for better organization
    all_undocumented.sort(key=lambda x: x['file'])
    
    print(f"Found {len(all_undocumented)} undocumented public functions:\n")
    
    for func in all_undocumented:
        print(f"File: {func['file']}")
        print(f"Line: {func['line']}")
        print(f"Function: {func['function']}")
        print(f"Code: {func['code']}")
        print("-" * 60)

if __name__ == "__main__":
    main()