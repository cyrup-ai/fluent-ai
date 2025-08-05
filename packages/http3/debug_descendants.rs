// Quick test to understand what collect_all_descendants_owned returns

fn main() {
    println!("Testing descendant collection...");
    
    // Test JSON structure
    let json_str = r#"{
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    }"#;
    
    println!("JSON: {}", json_str);
    
    // Manual analysis of what descendants should be:
    println!("\nExpected descendants of root:");
    println!("1. store object: {{\"book\": [...]}}");
    println!("2. book array: [{...}, {...}]");
    println!("3. book1 object: {{\"author\": \"Author 1\"}}");
    println!("4. book2 object: {{\"author\": \"Author 2\"}}");
    println!("5. \"Author 1\" string");
    println!("6. \"Author 2\" string");
    println!("Total: 6 descendants");
    
    println!("\nFor $..[?@.author], should apply filter to each descendant:");
    println!("1. store object: no 'author' property → no match");
    println!("2. book array: no 'author' property → no match");
    println!("3. book1 object: has 'author' property → MATCH");
    println!("4. book2 object: has 'author' property → MATCH");
    println!("5. \"Author 1\" string: primitive, no filter support → no match");
    println!("6. \"Author 2\" string: primitive, no filter support → no match");
    println!("Expected final result: 2 matches");
}