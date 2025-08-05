use fluent_ai_http3::json_path::parser::JsonPathParser;

fn main() {
    let slice_tests = vec![
        ("$[:]", true),           // Full slice
        ("$[1:]", true),          // Start only
        ("$[:5]", true),          // End only
        ("$[1:5]", true),         // Start and end
        ("$[::2]", true),         // Step only
        ("$[1::2]", true),        // Start and step
        ("$[:5:2]", true),        // End and step
        ("$[1:5:2]", true),       // Full slice
        ("$[ 1 : 5 : 2 ]", true), // Whitespace
        ("$[-1:]", true),         // Negative start
        ("$[:-1]", true),         // Negative end
        ("$[::-1]", true),        // Negative step
        ("$[1:5:0]", false),      // Zero step invalid
        ("$[1:5:]", false),       // Missing step after colon
    ];
    
    for (test_case, should_pass) in slice_tests {
        println!("Testing: '{}'", test_case);
        match JsonPathParser::compile(test_case) {
            Ok(_) => {
                if should_pass {
                    println!("✅ PASS: '{}'", test_case);
                } else {
                    println!("❌ UNEXPECTED PASS: '{}' should have failed", test_case);
                }
            }
            Err(e) => {
                if !should_pass {
                    println!("✅ EXPECTED FAIL: '{}' - {}", test_case, e);
                } else {
                    println!("❌ UNEXPECTED FAIL: '{}' - {}", test_case, e);
                }
            }
        }
    }
}