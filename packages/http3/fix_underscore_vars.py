#!/usr/bin/env python3

import os
import glob

def fix_underscore_variables(file_path):
    """Fix underscore variable naming issues in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace _expected_count with expected_count in for loop declarations
    original_content = content
    content = content.replace('_expected_count,', 'expected_count,')
    content = content.replace('_expected_count)', 'expected_count)')
    
    # Also fix other common underscore variable patterns that are actually used
    content = content.replace('_results', 'results')
    content = content.replace('_results1', 'results1')  
    content = content.replace('_results_sets', 'results_sets')
    content = content.replace('_result', 'result')
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed: {file_path}")
        return True
    return False

def main():
    test_dir = "/Volumes/samsung_t9/fluent-ai/packages/http3/tests"
    os.chdir(test_dir)
    
    # Find all .rs files with _expected_count
    rust_files = glob.glob("**/*.rs", recursive=True)
    
    fixed_count = 0
    for file_path in rust_files:
        if fix_underscore_variables(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()