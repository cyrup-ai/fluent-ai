use fluent_ai_http3::json_path::JsonPathParser;
use serde_json::json;

fn main() {
    println!("=== DEBUGGING RECURSIVE DESCENT FILTER ===");
    
    let json = json!({
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    });
    
    println!("JSON structure:");
    println!("{}", serde_json::to_string_pretty(&json).unwrap());
    
    // Test different queries to understand behavior
    let queries = vec![
        "$..*",                    // All descendants
        "$..author",              // All author properties
        "$..[?@.author]",         // Objects that have author property (failing test)
        "$.store.book[?@.author]", // Books that have author (for comparison)
    ];
    
    for query in queries {
        println!("\n--- Testing: '{}' ---", query);
        match JsonPathParser::compile(query) {
            Ok(parser) => {
                // Would need to evaluate against JSON here
                println!("✓ Compiled successfully");
            }
            Err(e) => {
                println!("✗ Compilation failed: {:?}", e);
            }
        }
    }
}