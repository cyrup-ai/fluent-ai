#[cfg(test)]
mod test_parser_fix {
    use super::*;
    
    #[test]
    fn test_recursive_descent_wildcard() {
        println!("Testing $..*  pattern compilation");
        let result = crate::json_path::parser::JsonPathParser::compile("$..*");
        match result {
            Ok(_) => println!("✅ SUCCESS: $..*  compiled successfully"),
            Err(e) => {
                println!("❌ FAILED: $..*  compilation failed");
                println!("Error: {}", e);
                panic!("$..*  should compile successfully");
            }
        }
    }
}