#!/usr/bin/env python3
"""
Batch fix ArrayVec imports across all provider client files
"""
import os
import re
from pathlib import Path

def fix_arrayvec_imports():
    provider_dir = Path("src/clients")
    files_with_arrayvec = []
    
    # Find all files that use ArrayVec but might be missing the import
    for file_path in provider_dir.rglob("*.rs"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check if file uses ArrayVec
            if 'ArrayVec' in content:
                # Check if it already has the import
                if 'use arrayvec::ArrayVec' not in content and 'use arrayvec::{' not in content:
                    files_with_arrayvec.append(file_path)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Fix each file
    for file_path in files_with_arrayvec:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find where to insert the import (after existing use statements)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('use ') and not line.strip().startswith('use super::'):
                    insert_index = i + 1
                elif line.strip() == '' and i > 0 and lines[i-1].strip().startswith('use '):
                    insert_index = i
                    break
            
            # Insert the import
            lines.insert(insert_index, 'use arrayvec::ArrayVec;\n')
            
            # Write back
            with open(file_path, 'w') as f:
                f.writelines(lines)
                
            print(f"Fixed: {file_path}")
            
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

if __name__ == "__main__":
    fix_arrayvec_imports()