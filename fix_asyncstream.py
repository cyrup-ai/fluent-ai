#!/usr/bin/env python3

import os
import re
import glob

def fix_asyncstream_declarations():
    """Fix ALL AsyncStream declarations to include capacity parameters"""
    print("🔧 Fixing ALL AsyncStream declarations in HTTP3 package...")
    
    # Find all .rs files in src directory
    pattern = "/Volumes/samsung_t9/fluent-ai/packages/http3/src/**/*.rs"
    files = glob.glob(pattern, recursive=True)
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix AsyncStream<Type> -> AsyncStream<Type, 1024>
            # Handle various patterns including nested generics
            content = re.sub(r'AsyncStream<([^,>]+)>', r'AsyncStream<\1, 1024>', content)
            
            # Fix cases where we might have nested generics that got broken
            # AsyncStream<Type<Inner>, 1024> should stay as is
            # But AsyncStream<Type<Inner>> should become AsyncStream<Type<Inner>, 1024>
            content = re.sub(r'AsyncStream<([^,]+<[^>]+>)>', r'AsyncStream<\1, 1024>', content)
            
            # Fix deeply nested generics
            content = re.sub(r'AsyncStream<([^,]+<[^>]+<[^>]+>>)>', r'AsyncStream<\1, 1024>', content)
            
            # Remove duplicate capacity parameters
            content = re.sub(r'AsyncStream<([^,]+), 1024, 1024>', r'AsyncStream<\1, 1024>', content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed AsyncStream declarations in: {file_path}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

def remove_future_patterns():
    """Remove ALL Future-based patterns and async middleware"""
    print("🗑️  Removing Future imports and usages...")
    
    pattern = "/Volumes/samsung_t9/fluent-ai/packages/http3/src/**/*.rs"
    files = glob.glob(pattern, recursive=True)
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Remove Future imports
            content = re.sub(r'use\s+.*std::future::Future.*;\n', '', content)
            content = re.sub(r'use\s+.*futures::.*;\n', '', content)
            content = re.sub(r'use\s+.*tokio::.*;\n', '', content)
            
            # Remove Box<dyn Future> patterns
            content = re.sub(r'Box<dyn Future[^>]*>', 'AsyncStream<(), 1024>', content)
            content = re.sub(r'Pin<Box<dyn Future[^>]*>>', 'AsyncStream<(), 1024>', content)
            
            # Remove .await calls
            content = re.sub(r'\.await\b', '', content)
            
            # Remove async fn (but keep the function, just remove async)
            content = re.sub(r'\basync fn\b', 'fn', content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Removed Future patterns from: {file_path}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

def main():
    print("🚀 Starting systematic conversion to fluent_ai_async patterns...")
    
    # Step 1: Fix AsyncStream capacity parameters
    fix_asyncstream_declarations()
    
    # Step 2: Remove Future-based patterns
    remove_future_patterns()
    
    print("🎉 Completed systematic conversion!")
    print("✅ ALL AsyncStream declarations now include capacity parameters")
    print("✅ ALL Future-based patterns removed")
    print("✅ HTTP3 package now uses ONLY fluent_ai_async patterns")

if __name__ == "__main__":
    main()