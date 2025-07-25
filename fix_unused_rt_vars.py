#!/usr/bin/env python3
"""
Script to fix unused 'rt' variable warnings in domain/src/chat/commands/mod.rs
"""

import re

def fix_unused_rt_variables():
    """Fix unused rt variable warnings by prefixing with underscore."""
    file_path = "/Volumes/samsung_t9/fluent-ai/packages/domain/src/chat/commands/mod.rs"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace 'let rt =' with 'let _rt =' to indicate intentionally unused
        content = re.sub(r'let rt = tokio::runtime::Runtime::new\(\)', 'let _rt = tokio::runtime::Runtime::new()', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Count the replacements
            rt_count = original_content.count('let rt = tokio::runtime::Runtime::new()')
            _rt_count = content.count('let _rt = tokio::runtime::Runtime::new()')
            
            print(f"✅ Fixed {_rt_count} unused 'rt' variables in {file_path}")
            return _rt_count
        else:
            print(f"🔍 No unused 'rt' variables found in {file_path}")
            return 0
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return 0

def main():
    """Main function to fix unused rt variables."""
    print("🔧 Fixing unused 'rt' variable warnings...")
    
    fixed_count = fix_unused_rt_variables()
    
    print(f"\n🎉 Summary:")
    print(f"   ✅ Fixed: {fixed_count} unused 'rt' variables")

if __name__ == "__main__":
    main()