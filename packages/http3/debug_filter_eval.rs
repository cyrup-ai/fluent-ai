use serde_json::{json, Value};
use fluent_ai_http3::json_path::{
    filter::FilterEvaluator,
    filter_parser::{FilterParser, JsonPathResult},
    ast::FilterExpression,
};

fn main() -> JsonPathResult<()> {
    // Test objects 
    let root_obj = json!({
        "store": {
            "book": [
                {"author": "Author 1"},
                {"author": "Author 2"}
            ]
        }
    });
    
    let store_obj = json!({
        "book": [
            {"author": "Author 1"},
            {"author": "Author 2"}
        ]
    });
    
    let book_obj_with_author = json!({"author": "Author 1"});
    let book_obj_without_author = json!({"title": "Some Book"});
    
    // Parse the filter expression @.author
    let filter_expr = "@.author";
    let mut parser = FilterParser::new(filter_expr);
    let filter_ast = parser.parse_filter_expression()?;
    
    println!("=== Testing Filter Expression: {} ===", filter_expr);
    println!("Filter AST: {:?}", filter_ast);
    println!();
    
    // Test each object
    let test_cases = vec![
        ("root_obj", &root_obj),
        ("store_obj", &store_obj), 
        ("book_with_author", &book_obj_with_author),
        ("book_without_author", &book_obj_without_author),
    ];
    
    for (name, obj) in test_cases {
        let result = FilterEvaluator::evaluate_predicate(obj, &filter_ast)?;
        println!("FilterEvaluator::evaluate_predicate({}, @.author) = {}", name, result);
        println!("  Object: {}", obj);
        println!("  Has 'author' property: {}", obj.get("author").is_some());
        println!();
    }
    
    Ok(())
}