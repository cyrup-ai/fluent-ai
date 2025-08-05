use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    let test_cases = vec![
        "$[?count($..book)]",
        "$[?length(@.items)]",
        "$[?match(@.title, \"test\")]",
        "$[?search(@.content, \"test\")]", 
        "$[?value(@.price)]",
    ];
    
    for query in test_cases {
        println!("Testing: {}", query);
        match JsonPathParser::compile(query) {
            Ok(_) => println!("✅ Successfully compiled: {}\n", query),
            Err(e) => println!("❌ Failed to compile: {} - Error: {:?}\n", query, e),
        }
    }
}