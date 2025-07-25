#!/usr/bin/env python3
"""
Mass warning fix script for fluent-ai codebase.
Focuses on quick wins: unused imports, simple dead code, cfg conditions.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

def run_cargo_check() -> Tuple[List[str], int]:
    """Run cargo check and return warnings."""
    result = subprocess.run(
        ["cargo", "check", "--workspace"],
        cwd="/Volumes/samsung_t9/fluent-ai",
        capture_output=True,
        text=True
    )
    
    warnings = []
    for line in result.stderr.split('\n'):
        if 'warning:' in line:
            warnings.append(line.strip())
    
    return warnings, len(warnings)

def fix_unused_imports():
    """Fix unused import warnings using regex patterns."""
    print("🔧 Fixing unused imports...")
    
    # Common unused import patterns
    patterns = [
        (r'use\s+std::collections::HashMap;\s*\n', ''),
        (r'use\s+arrayvec::ArrayVec;\s*\n', ''),
        (r'use\s+ArrayVec;\s*\n', ''),
        # Remove from compound imports
        (r'(use\s+[^{]*\{[^}]*),\s*HashMap\s*([^}]*\})', r'\1\2'),
        (r'(use\s+[^{]*\{[^}]*),\s*ArrayVec\s*([^}]*\})', r'\1\2'),
        (r'(use\s+[^{]*\{)\s*HashMap\s*,\s*([^}]*\})', r'\1\2'),
        (r'(use\s+[^{]*\{)\s*ArrayVec\s*,\s*([^}]*\})', r'\1\2'),
    ]
    
    fixed_files = 0
    
    # Find all Rust files
    for root, dirs, files in os.walk("/Volumes/samsung_t9/fluent-ai/packages"):
        for file in files:
            if file.endswith('.rs'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply all patterns
                    for pattern, replacement in patterns:
                        content = re.sub(pattern, replacement, content)
                    
                    # Clean up empty braces and double commas
                    content = re.sub(r'use\s+[^{]*\{\s*\};\s*\n', '', content)
                    content = re.sub(r',\s*,', ',', content)
                    content = re.sub(r'\{\s*,', '{', content)
                    content = re.sub(r',\s*\}', '}', content)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_files += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
    
    print(f"✅ Fixed unused imports in {fixed_files} files")
    return fixed_files

def fix_cfg_conditions():
    """Fix unexpected cfg condition warnings."""
    print("🔧 Fixing cfg conditions...")
    
    patterns = [
        (r'#\[cfg\(feature\s*=\s*"worker"\)\]', '#[cfg(feature = "worker")]'),
        (r'#\[cfg\(feature\s*=\s*"generation"\)\]', '#[cfg(feature = "generation")]'),
    ]
    
    fixed_files = 0
    
    # Find all Rust files
    for root, dirs, files in os.walk("/Volumes/samsung_t9/fluent-ai/packages"):
        for file in files:
            if file.endswith('.rs'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply patterns
                    for pattern, replacement in patterns:
                        content = re.sub(pattern, replacement, content)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_files += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
    
    print(f"✅ Fixed cfg conditions in {fixed_files} files")
    return fixed_files

def prefix_unused_variables():
    """Add underscore prefix to unused variables."""
    print("🔧 Fixing unused variables...")
    
    # This would be more complex and risky, so we'll skip for now
    # and focus on safer fixes
    print("⚠️  Skipping unused variables (manual review needed)")
    return 0

def main():
    """Main function to mass fix warnings."""
    print("🚀 Starting mass warning fix...")
    
    # Get initial warning count
    initial_warnings, initial_count = run_cargo_check()
    print(f"📊 Initial warning count: {initial_count}")
    
    total_fixed = 0
    
    # Fix unused imports
    total_fixed += fix_unused_imports()
    
    # Fix cfg conditions  
    total_fixed += fix_cfg_conditions()
    
    # Get final warning count
    print("\n📊 Checking final warning count...")
    final_warnings, final_count = run_cargo_check()
    
    reduction = initial_count - final_count
    
    print(f"\n🎉 Summary:")
    print(f"   📈 Initial warnings: {initial_count}")
    print(f"   📉 Final warnings: {final_count}")
    print(f"   ✅ Warnings reduced by: {reduction}")
    print(f"   📁 Files modified: {total_fixed}")
    
    if reduction > 0:
        print(f"   🎯 Success rate: {(reduction/initial_count)*100:.1f}%")
    else:
        print("   ⚠️  No warnings were automatically fixable")

if __name__ == "__main__":
    main()