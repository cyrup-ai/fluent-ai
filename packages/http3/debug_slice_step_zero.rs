use crate::json_path::compiler::JsonPathParser;

fn main() {
    println!("Testing slice step zero validation:");
    
    let test_cases = vec![
        "$[1:5:0]",   // Should fail - step size 0
        "$[1:5:1]",   // Should pass - step size 1  
        "$[1:5:2]",   // Should pass - step size 2
        "$[::0]",     // Should fail - step size 0
        "$[1::0]",    // Should fail - step size 0
    ];
    
    for case in test_cases {
        println!("\nTesting: {}", case);
        match JsonPathParser::compile(case) {
            Ok(_) => println!("  ✓ PASSED (compiled successfully)"),
            Err(e) => println!("  ✗ FAILED: {}", e),
        }
    }
}