//! Debug test to understand the filter evaluation issue

use serde_json::{Value, json};

fn main() {
    let json_data = json!({
      "store": {
        "book": [
          {
            "category": "reference",
            "author": "Nigel Rees",
            "title": "Sayings of the Century", 
            "price": 8.95,
            "isbn": "0-553-21311-3",
            "metadata": null,
            "tags": ["classic", "quotes"]
          },
          {
            "category": "fiction",
            "author": "Evelyn Waugh",
            "title": "Sword of Honour",
            "price": 12.99,
            "availability": null,
            "tags": ["fiction", "war"]
          },
          {
            "category": "fiction",
            "author": "Herman Melville",
            "title": "Moby Dick",
            "price": 8.99,
            "tags": null
          }
        ]
      }
    });
    
    if let Some(books) = json_data.pointer("/store/book").and_then(|v| v.as_array()) {
        for (i, book) in books.iter().enumerate() {
            println!("Book {}: {}", i, serde_json::to_string_pretty(book).unwrap());
            
            // Check metadata field
            let has_metadata_key = book.as_object().unwrap().contains_key("metadata");
            let metadata_value = book.get("metadata");
            
            println!("  Has metadata key: {}", has_metadata_key);
            println!("  Metadata value: {:?}", metadata_value);
            
            // Check what happens with comparison logic
            if has_metadata_key {
                if let Some(meta) = metadata_value {
                    if meta.is_null() {
                        println!("  metadata is explicitly null");
                        println!("  null != null = {}", false);
                    } else {
                        println!("  metadata is not null: {:?}", meta);
                        println!("  value != null = {}", true);
                    }
                } else {
                    println!("  ERROR: has key but no value");
                }
            } else {
                println!("  metadata is missing (no key)");
                println!("  missing != null = should be false per RFC (current impl) or true (test expects?)");
            }
            println!();
        }
    }

    // Now let's test what the expected behavior should be according to the test
    println!("ANALYSIS:");
    println!("Test expects $.store.book[?@.metadata != null] to return 2 results");
    println!("Looking at the books:");
    println!("- Book 0: metadata = null, so null != null = false (no match)");
    println!("- Book 1: metadata missing, so missing != null = ??? (test expects match)");
    println!("- Book 2: metadata missing, so missing != null = ??? (test expects match)");
    println!();
    println!("Conclusion: Test expects missing properties in != null comparison to be true");
    println!("This contradicts the other test that expects missing != null to be false");
    println!("Maybe there's a different interpretation...");
}