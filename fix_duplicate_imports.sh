#!/bin/bash

# Script to find and fix duplicate HashMap imports
echo "Finding files with duplicate HashMap imports..."

# Get list of files with potential duplicate HashMap imports
find /Volumes/samsung_t9/fluent-ai/packages -name "*.rs" -type f | while read file; do
    # Count HashMap import lines
    hashmap_count=$(grep -c "use.*HashMap" "$file" 2>/dev/null || echo 0)
    
    if [ "$hashmap_count" -gt 1 ]; then
        echo "Processing $file (found $hashmap_count HashMap imports)"
        
        # Create a temporary file to store the result
        temp_file=$(mktemp)
        
        # Process the file to remove duplicate HashMap imports
        # Keep the first occurrence, remove subsequent ones
        awk '
        BEGIN { seen_hashmap = 0 }
        /use.*HashMap/ {
            if (seen_hashmap == 0) {
                print $0
                seen_hashmap = 1
            } else {
                # Skip this duplicate import line
                next
            }
        }
        !/use.*HashMap/ { print $0 }
        ' "$file" > "$temp_file"
        
        # Replace the original file with the processed version
        mv "$temp_file" "$file"
        echo "Fixed duplicate HashMap imports in $file"
    fi
done

echo "Duplicate HashMap import fixing complete!"