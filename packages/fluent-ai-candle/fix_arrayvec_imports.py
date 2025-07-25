#!/usr/bin/env python3
"""Fix missing ArrayVec imports in types.rs files"""

import os
import re

def fix_arrayvec_import(file_path):
    """Add ArrayVec import if missing but ArrayVec is used"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if ArrayVec is used but not imported
    if 'ArrayVec' in content and 'use arrayvec::ArrayVec' not in content:
        print(f"Fixing ArrayVec import in {file_path}")
        
        # Find the serde import line to add ArrayVec import after it
        lines = content.split('\n')
        insert_idx = None
        
        for i, line in enumerate(lines):
            if 'use serde::{Deserialize, Serialize};' in line:
                insert_idx = i + 1
                break
        
        if insert_idx is not None:
            lines.insert(insert_idx, 'use arrayvec::ArrayVec;')
            
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))
            return True
        else:
            print(f"Could not find serde import in {file_path}")
            return False
    else:
        print(f"ArrayVec import not needed in {file_path}")
        return False

# Process all types.rs files in the provider package
provider_clients_dir = "/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients"
fixed_count = 0

for client_dir in os.listdir(provider_clients_dir):
    client_path = os.path.join(provider_clients_dir, client_dir)
    if os.path.isdir(client_path):
        types_file = os.path.join(client_path, "types.rs")
        if os.path.exists(types_file):
            if fix_arrayvec_import(types_file):
                fixed_count += 1

print(f"Fixed ArrayVec imports in {fixed_count} files")