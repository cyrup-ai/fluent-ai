#!/bin/bash

# Bulk fix HashMap import issues in fluent-ai-candle
# Finds files that use HashMap but don't import it, then adds the import

cd /Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle

# Get list of files with HashMap errors
cargo check --message-format=short 2>&1 | \
    grep "error.*HashMap.*not found" | \
    awk -F: '{print $1}' | \
    sort | uniq > /tmp/hashmap_files.txt

echo "Files needing HashMap imports:"
cat /tmp/hashmap_files.txt

# Fix each file by adding HashMap import if not already present
while IFS= read -r file; do
    if [ -f "$file" ]; then
        # Check if HashMap import already exists
        if ! grep -q "use std::collections::HashMap" "$file"; then
            echo "Adding HashMap import to: $file"
            
            # Find the first use statement and add HashMap import after it
            # Using a more robust sed approach
            awk '
                /^use / && !added {
                    print "use std::collections::HashMap;"
                    added = 1
                }
                { print }
            ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
        else
            echo "HashMap already imported in: $file"
        fi
    fi
done < /tmp/hashmap_files.txt

echo "HashMap import fixes completed!"