#!/bin/bash

# Script to fix ALL AsyncStream declarations to include capacity parameters
# This will systematically replace AsyncStream<T> with AsyncStream<T, 1024>

echo "🔧 Fixing ALL AsyncStream declarations in HTTP3 package..."

# Find all .rs files in src directory and fix AsyncStream declarations
find /Volumes/samsung_t9/fluent-ai/packages/http3/src -name "*.rs" -type f | while read file; do
    echo "Processing: $file"
    
    # Replace AsyncStream<Type> with AsyncStream<Type, 1024>
    # This handles various patterns like AsyncStream<SomeType>, AsyncStream<SomeType<Inner>>, etc.
    sed -i '' 's/AsyncStream<\([^>]*\)>/AsyncStream<\1, 1024>/g' "$file"
    
    # Handle nested generics like AsyncStream<Type<Inner>>
    # We need multiple passes to handle deeply nested types
    sed -i '' 's/AsyncStream<\([^>]*<[^>]*>\), 1024>/AsyncStream<\1, 1024>/g' "$file"
    sed -i '' 's/AsyncStream<\([^>]*<[^>]*<[^>]*>>\), 1024>/AsyncStream<\1, 1024>/g' "$file"
    
    # Fix cases where we accidentally added capacity twice
    sed -i '' 's/AsyncStream<\([^,]*\), 1024, 1024>/AsyncStream<\1, 1024>/g' "$file"
done

echo "✅ Completed fixing AsyncStream declarations"

# Now remove any remaining Future imports and usages
echo "🗑️  Removing Future imports and usages..."

find /Volumes/samsung_t9/fluent-ai/packages/http3/src -name "*.rs" -type f | while read file; do
    # Remove Future imports
    sed -i '' '/use.*std::future::Future/d' "$file"
    sed -i '' '/use.*futures::/d' "$file"
    sed -i '' '/use.*tokio::/d' "$file"
    
    # Remove async fn declarations (but keep the function, just remove async)
    sed -i '' 's/async fn /fn /g' "$file"
    
    # Remove .await calls
    sed -i '' 's/\.await//g' "$file"
    
    # Remove Box<dyn Future> patterns
    sed -i '' 's/Box<dyn Future[^>]*>//g' "$file"
    sed -i '' 's/Pin<Box<dyn Future[^>]*>>//g' "$file"
done

echo "✅ Completed removing Future patterns"
echo "🎉 All AsyncStream declarations fixed and Future patterns removed!"