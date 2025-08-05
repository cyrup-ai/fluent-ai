use fluent_ai_http3::json_path::parser::JsonPathParser;

fn main() {
    let test_cases = vec![
        "$",          // Should pass - root only
        "$..",        // Should fail - bare descendant  
        "$..store",   // Should pass - descendant with property
        "$..[0]",     // Should pass - descendant with bracket
    ];
    
    for test_case in test_cases {
        match JsonPathParser::compile(test_case) {
            Ok(_) => println!("✅ PASS: '{}'", test_case),
            Err(e) => println!("❌ FAIL: '{}' - {}", test_case, e),
        }
    }
}