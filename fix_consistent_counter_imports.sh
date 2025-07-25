#!/bin/bash

# Script to fix atomic_counter::ConsistentCounter imports in fluent-ai-candle
# Replace with our local ConsistentCounter implementation

echo "🔧 Fixing atomic_counter::ConsistentCounter imports..."

# Find all Rust files in fluent-ai-candle that import atomic_counter::ConsistentCounter
find /Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle -name "*.rs" -type f | while read file; do
    if grep -q "atomic_counter.*ConsistentCounter" "$file"; then
        echo "Processing $file"
        
        # Replace the import
        sed -i '' 's/use atomic_counter::\(.*,\s*\)\?ConsistentCounter\(.*\)\?;/use crate::types::candle_chat::search::tagging::ConsistentCounter;/g' "$file"
        
        # Also handle cases where it's just atomic_counter::ConsistentCounter
        sed -i '' 's/atomic_counter::ConsistentCounter/ConsistentCounter/g' "$file"
        
        echo "✅ Fixed imports in $file"
    fi
done

echo "🎉 Finished fixing ConsistentCounter imports"