#[cfg(test)]
mod tests {
    use crate::json_path::JsonPathParser;

#[test]
fn test_recursive_descent_validation() {
    let expressions = vec![
        "$.store..book",     // Should this be valid or invalid?
        "$..book",           // Should be valid (RFC examples show this)
        "$.store.book",      // Should be valid
        "$.store..",         // Should be invalid (ends with ..)
    ];
    
    for expr in expressions {
        println!("\nTesting: {}", expr);
        match JsonPathParser::compile(expr) {
            Ok(_) => println!("  ✓ VALID"),
            Err(e) => println!("  ✗ INVALID: {}", e),
        }
    }
}
}