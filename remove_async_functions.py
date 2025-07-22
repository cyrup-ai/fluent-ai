#!/usr/bin/env python3
"""
Script to remove legacy async function wrappers from search.rs file.
This removes async functions that are just calling .collect() on streaming methods.
"""

import re
import sys

def remove_async_functions(content: str) -> str:
    """Remove async function blocks that are legacy wrappers."""
    
    # Patterns for different async function types to remove
    patterns_to_remove = [
        # Legacy wrapper functions that just call .collect() on streams
        r'    /// .* \(legacy\)\n    (?:pub )?async fn [^{]+{\n(?:        [^\n]+\n)*    }',
        
        # Legacy future-compatible method pattern
        r'    /// .* \(legacy future-compatible method\)\n    pub async fn [^{]+{\n(?:        [^\n]+\n)*    }',
        
        # Simple async wrappers that just convert streams
        r'    pub async fn ([a-zA-Z_]+)\([^)]+\) -> Result<[^,]+, [^>]+> {\n(?:        [^\n]+\n)*    }',
        
        # Private async functions that wrap streams
        r'    async fn ([a-zA-Z_]+)\([^)]+\) -> Result<[^,]+, [^>]+> {\n(?:        [^\n]+\n)*    }',
    ]
    
    modified_content = content
    
    for pattern in patterns_to_remove:
        modified_content = re.sub(pattern, '', modified_content, flags=re.MULTILINE)
    
    # Clean up extra blank lines
    modified_content = re.sub(r'\n\n\n+', '\n\n', modified_content)
    
    return modified_content

def main():
    file_path = "/Volumes/samsung_t9/fluent-ai/packages/domain/src/chat/search.rs"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_async_count = len(re.findall(r'async fn', content))
        print(f"Original async function count: {original_async_count}")
        
        # Remove async functions
        modified_content = remove_async_functions(content)
        
        new_async_count = len(re.findall(r'async fn', modified_content))
        print(f"New async function count: {new_async_count}")
        print(f"Removed {original_async_count - new_async_count} async functions")
        
        # Write back the modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("File updated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())