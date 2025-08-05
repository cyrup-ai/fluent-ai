// Test the logic of the descendant segment fix
// This tests what should happen with $..[?@.author] on our test JSON

use serde_json::{json, Value};

fn main() {
    let json = json!({
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    });
    
    println!("=== Testing Descendant Segment Logic ===");
    println!("JSON: {}", serde_json::to_string_pretty(&json).unwrap());
    
    println!("\n=== What nodes exist at each depth? ===");
    
    // Root (depth 0)
    println!("Depth 0 (root): {}", json);
    
    // Depth 1: store object
    let store = &json["store"];
    println!("Depth 1 (store): {}", store);
    
    // Depth 2: book array
    let book_array = &json["store"]["book"];
    println!("Depth 2 (book array): {}", book_array);
    
    // Depth 3: individual book objects
    let book1 = &json["store"]["book"][0];
    let book2 = &json["store"]["book"][1];
    println!("Depth 3 (book1): {}", book1);
    println!("Depth 3 (book2): {}", book2);
    
    println!("\n=== Testing filter [?@.author] on each node ===");
    
    // Test filter on each node
    let nodes = vec![
        ("root", &json),
        ("store", store),
        ("book_array", book_array),
        ("book1", book1),
        ("book2", book2),
    ];
    
    let mut expected_matches = Vec::new();
    
    for (name, node) in &nodes {
        let has_author = node.get("author").is_some();
        println!("{}: has 'author' property = {}", name, has_author);
        
        // Filter [?@.author] should match nodes that have an 'author' property
        if has_author {
            expected_matches.push((*name, (*node).clone()));
        }
    }
    
    println!("\n=== Expected Results ===");
    println!("Expected {} matches:", expected_matches.len());
    for (name, value) in &expected_matches {
        println!("  {}: {}", name, value);
    }
    
    println!("\n=== Expected $..[?@.author] result ===");
    println!("Should return exactly {} results:", expected_matches.len());
    for (i, (_, value)) in expected_matches.iter().enumerate() {
        println!("  [{}]: {}", i, value);
    }
}