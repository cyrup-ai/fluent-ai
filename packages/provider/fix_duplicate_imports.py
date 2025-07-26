#!/usr/bin/env python3
"""
Remove duplicate imports that may have been added
"""
import os
import re
from pathlib import Path

def fix_duplicate_imports():
    provider_dir = Path("src/clients")
    
    files_fixed = 0
    
    for file_path in provider_dir.rglob("*.rs"):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Track seen imports to remove duplicates
            seen_imports = set()
            cleaned_lines = []
            imports_removed = 0
            
            for line in lines:
                stripped = line.strip()
                
                # Check if it's a use statement
                if stripped.startswith('use ') and stripped.endswith(';'):
                    if stripped in seen_imports:
                        imports_removed += 1
                        continue  # Skip duplicate
                    else:
                        seen_imports.add(stripped)
                
                cleaned_lines.append(line)
            
            if imports_removed > 0:
                # Write back cleaned content
                with open(file_path, 'w') as f:
                    f.writelines(cleaned_lines)
                    
                print(f"Removed {imports_removed} duplicate imports from: {file_path}")
                files_fixed += 1
                
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    print(f"Total files fixed: {files_fixed}")

if __name__ == "__main__":
    fix_duplicate_imports()