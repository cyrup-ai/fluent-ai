#!/usr/bin/env python3
"""
Batch fix common missing imports: HashMap, HttpClient, HttpResponse
"""
import os
import re
from pathlib import Path

def fix_common_imports():
    provider_dir = Path("src/clients")
    
    # Common imports to add based on usage
    import_patterns = {
        'HashMap': 'use std::collections::HashMap;',
        'HttpClient': 'use fluent_ai_http3::HttpClient;',
        'HttpResponse': 'use fluent_ai_http3::HttpResponse;',
        'HttpError': 'use fluent_ai_http3::HttpError;',
        'HttpRequest': 'use fluent_ai_http3::HttpRequest;'
    }
    
    files_fixed = 0
    
    for file_path in provider_dir.rglob("*.rs"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            lines = content.split('\n')
            imports_to_add = []
            
            # Check what imports are needed
            for pattern, import_stmt in import_patterns.items():
                if pattern in content and import_stmt.split('::')[-1] not in content:
                    imports_to_add.append(import_stmt)
            
            if not imports_to_add:
                continue
                
            # Find where to insert imports
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('use ') and not line.strip().startswith('use super::'):
                    insert_index = i + 1
                elif line.strip() == '' and i > 0 and lines[i-1].strip().startswith('use '):
                    insert_index = i
                    break
            
            # Insert imports
            for import_stmt in sorted(imports_to_add):
                lines.insert(insert_index, import_stmt)
                insert_index += 1
            
            # Write back
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))
                
            print(f"Fixed {len(imports_to_add)} imports in: {file_path}")
            files_fixed += 1
            
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    print(f"Total files fixed: {files_fixed}")

if __name__ == "__main__":
    fix_common_imports()