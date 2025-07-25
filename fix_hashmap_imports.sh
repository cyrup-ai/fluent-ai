#!/bin/bash

# Find all .rs files that use HashMap:: but don't have HashMap import
files=$(find /Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src -name "*.rs" -exec grep -l "HashMap::" {} \; | xargs grep -L "use.*HashMap")

for file in $files; do
    echo "Fixing $file"
    # Check if file has any existing use statements
    if grep -q "^use " "$file"; then
        # Add HashMap import after the first use statement that doesn't import HashMap
        sed -i '' '/^use std::/a\
use std::collections::HashMap;
' "$file"
    else
        # Add import at the beginning if no use statements exist
        sed -i '' '1i\
use std::collections::HashMap;\
' "$file"
    fi
done

echo "Fixed $(echo "$files" | wc -l) files"