#!/bin/bash

# Script to fix HashMap import issues in fluent-ai-candle
echo "🔧 Starting HashMap import fixes..."

# Files that need HashMap import (first batch)
FILES1=(
"src/types/candle_context/provider/processor.rs"
"src/types/candle_context/provider/context_impls.rs"
"src/types/candle_model/resolver.rs"
"src/types/candle_chat/chat/search/tagging/types.rs"
"src/types/candle_chat/chat/search/tagging/statistics.rs"
"src/types/candle_chat/chat/integrations/types.rs"
"src/types/candle_chat/chat/integrations/external.rs"
"src/types/candle_chat/chat/integrations/plugin.rs"
"src/types/candle_chat/chat/macros/mod.rs"
"src/types/candle_chat/chat/templates/core.rs"
)

cd /Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle

for file in "${FILES1[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        # Check if HashMap import already exists
        if ! grep -q "use std::collections::HashMap;" "$file"; then
            # Add HashMap import at the beginning of use statements
            if grep -q "^use " "$file"; then
                # Find first use statement and add HashMap import before it
                sed -i '' '1,/^use / { /^use / i\
use std::collections::HashMap;
}; ' "$file"
                echo "  ✅ Added HashMap import to $file"
            fi
        else
            echo "  ✅ HashMap already imported in $file"  
        fi
    else
        echo "  ❌ File not found: $file"
    fi
done

echo "🎉 Completed first batch of HashMap import fixes!"