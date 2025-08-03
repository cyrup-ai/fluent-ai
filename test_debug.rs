use std::collections::HashMap;

fn main() {
    println!("=== Debugging infinite loop ===");
    
    // The patterns that are failing:
    // 1. $..book[2] - Recursive descent to find book arrays, then select index 2
    // 2. $.store.bicycle - Direct property access to store.bicycle
    
    // Expected behavior:
    // For $..book[2]: Should find all "book" properties recursively, 
    // then apply index [2] to get the third element (Herman Melville's book)
    
    // For $.store.bicycle: Should navigate $.store then access .bicycle property
    
    println!("Expected for $..book[2]:");
    println!("  1. Find all descendants with property 'book'");  
    println!("  2. If book is an array, apply index [2]");
    println!("  3. Should return Herman Melville's 'Moby Dick' book");
    
    println!("\nExpected for $.store.bicycle:");
    println!("  1. Navigate to root ($)");
    println!("  2. Access 'store' property");
    println!("  3. Access 'bicycle' property"); 
    println!("  4. Should return red bicycle with price 19.95");
    
    // Test data structure analysis:
    println!("\n=== Data structure analysis ===");
    println!("Root object has:");
    println!("  - store (object)");
    println!("    - book (array with 4 elements)");
    println!("      - [0]: Nigel Rees");
    println!("      - [1]: Evelyn Waugh");
    println!("      - [2]: Herman Melville  <-- This should be selected by $..book[2]");
    println!("      - [3]: J. R. R. Tolkien");
    println!("    - bicycle (object)  <-- This should be selected by $.store.bicycle");
    println!("      - color: 'red'");
    println!("      - price: 19.95");
    
    println!("\n=== Potential infinite loop causes ===");
    println!("1. Recursive descent ($..) might be getting stuck in a loop");
    println!("2. Index selector [2] after recursive descent might be malformed");
    println!("3. Simple property access ($.store.bicycle) timing out suggests basic traversal issue");
    println!("4. CoreJsonPathEvaluator.apply_selector might have infinite recursion");
    
    println!("\n=== Next debugging steps ===");
    println!("1. Check if JsonPathParser correctly parses these expressions");
    println!("2. Check if CoreJsonPathEvaluator.apply_selector has infinite loop");
    println!("3. Check if collect_all_descendants has infinite recursion");
    println!("4. Add logging to see where execution gets stuck");
}