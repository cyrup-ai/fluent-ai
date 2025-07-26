#!/usr/bin/env python3
"""
Fix missing 'pub mod types;' declarations in mod.rs files
"""
import os
from pathlib import Path

def fix_types_modules():
    clients_dir = Path("src/clients")
    
    # Directories that need types module declarations
    client_dirs = ['ai21', 'openrouter', 'perplexity', 'together']
    
    for client_dir in client_dirs:
        mod_path = clients_dir / client_dir / "mod.rs"
        
        if not mod_path.exists():
            print(f"Warning: {mod_path} does not exist")
            continue
            
        try:
            with open(mod_path, 'r') as f:
                content = f.read()
            
            # Check if types module is already declared
            if 'pub mod types' in content or 'mod types' in content:
                print(f"Skipping {mod_path} - types module already declared")
                continue
            
            # Add the types module declaration at the end
            if not content.endswith('\n'):
                content += '\n'
            content += 'pub mod types;\n'
            
            with open(mod_path, 'w') as f:
                f.write(content)
                
            print(f"Added 'pub mod types;' to {mod_path}")
            
        except Exception as e:
            print(f"Error fixing {mod_path}: {e}")

    # Special case: gemini uses gemini_types.rs instead of types.rs
    gemini_mod_path = clients_dir / "gemini" / "mod.rs"
    if gemini_mod_path.exists():
        try:
            with open(gemini_mod_path, 'r') as f:
                content = f.read()
            
            if 'pub mod gemini_types' not in content and 'mod gemini_types' not in content:
                if not content.endswith('\n'):
                    content += '\n'
                content += 'pub mod gemini_types;\n'
                
                with open(gemini_mod_path, 'w') as f:
                    f.write(content)
                    
                print(f"Added 'pub mod gemini_types;' to {gemini_mod_path}")
            
        except Exception as e:
            print(f"Error fixing {gemini_mod_path}: {e}")

if __name__ == "__main__":
    fix_types_modules()