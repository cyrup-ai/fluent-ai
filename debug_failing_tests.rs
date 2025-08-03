use serde_json::{json, Value};

// Let me create a simple debug program to test the failing patterns
fn main() {
    let bookstore_json = json!({
        "store": {
            "book": [
                {
                    "category": "reference",
                    "author": "Nigel Rees",
                    "title": "Sayings of the Century",
                    "price": 8.95
                },
                {
                    "category": "fiction",
                    "author": "Evelyn Waugh", 
                    "title": "Sword of Honour",
                    "price": 12.99
                },
                {
                    "category": "fiction",
                    "author": "Herman Melville",
                    "title": "Moby Dick",
                    "isbn": "0-553-21311-3",
                    "price": 8.99
                },
                {
                    "category": "fiction",
                    "author": "J. R. R. Tolkien",
                    "title": "The Lord of the Rings",
                    "isbn": "0-395-19395-8", 
                    "price": 22.99
                }
            ],
            "bicycle": {
                "color": "red",
                "price": 19.95
            }
        }
    });

    println!("Bookstore JSON structure:");
    println!("{}", serde_json::to_string_pretty(&bookstore_json).unwrap());
    
    // Test the specific patterns that are failing:
    println!("\n=== Testing failing patterns ===");
    
    // Pattern 1: $..book[2] (third book - Herman Melville)
    println!("Pattern 1: $..book[2]");
    println!("Expected: Herman Melville's book (Moby Dick)");
    println!("Book array length: {}", bookstore_json["store"]["book"].as_array().unwrap().len());
    println!("Book[2]: {}", bookstore_json["store"]["book"][2]);
    
    // Pattern 2: $.store.bicycle
    println!("\nPattern 2: $.store.bicycle");  
    println!("Expected: red bicycle with price 19.95");
    println!("Bicycle: {}", bookstore_json["store"]["bicycle"]);
}