use fluent_ai_provider::clients::anthropic::tools::{
    with_builtins, ToolRegistryBuilder, ToolExecutionContext, ToolOutput, ExpressionEvaluator
};
use serde_json::json;
use std::collections::HashMap;

#[tokio::test]
async fn test_tool_registry_creation() {
    let registry = with_builtins();
    
    // Verify built-in tools are registered
    assert!(registry.has_tool("calculator"));
    assert!(registry.has_tool("file_operations"));
    
    // Verify tool count
    let tools = registry.list_tools();
    assert_eq!(tools.len(), 2);
}

#[tokio::test]
async fn test_calculator_tool_execution() {
    let registry = with_builtins();
    let context = ToolExecutionContext {
        conversation_id: None,
        user_id: None,
        metadata: HashMap::new(),
        timeout_ms: Some(5000),
        max_retries: 3,
        message_history: vec![],
    };

    let input = json!({
        "expression": "2 + 3 * 4"
    });

    let result = registry.execute_tool("calculator", input, &context).await;
    assert!(result.is_ok());
    
    if let Ok(ToolOutput::Json(output)) = result {
        assert_eq!(output["result"], json!(14.0));
        assert_eq!(output["expression"], json!("2 + 3 * 4"));
    } else {
        panic!("Expected JSON output");
    }
}

#[tokio::test]
async fn test_registry_builder() {
    let result = ToolRegistryBuilder::new()
        .with_calculator()
        .expect("Failed to add calculator to tool registry")
        .build();
    
    assert!(result.has_tool("calculator"));
    assert!(!result.has_tool("file_operations"));
}

#[test]
fn test_expression_evaluator() {
    let mut evaluator = ExpressionEvaluator::new();
    
    // Test basic arithmetic
    assert_eq!(evaluator.evaluate("2 + 3").expect("Failed to evaluate 2 + 3"), 5.0);
    assert_eq!(evaluator.evaluate("10 / 2").expect("Failed to evaluate 10 / 2"), 5.0);
    assert_eq!(evaluator.evaluate("2^3").expect("Failed to evaluate 2^3"), 8.0);
    
    // Test functions
    assert!((evaluator.evaluate("sin(0)").expect("Failed to evaluate sin(0)") - 0.0).abs() < f64::EPSILON);
    assert_eq!(evaluator.evaluate("sqrt(16)").expect("Failed to evaluate sqrt(16)"), 4.0);
    
    // Test constants
    assert!((evaluator.evaluate("pi").expect("Failed to evaluate pi") - std::f64::consts::PI).abs() < f64::EPSILON);
    
    // Test variables
    assert_eq!(evaluator.evaluate("x = 5; x * 2").expect("Failed to evaluate variable expression"), 10.0);
    
    // Test error cases
    assert!(evaluator.evaluate("1 / 0").is_err());
    assert!(evaluator.evaluate("sqrt(-1)").is_err());
    assert!(evaluator.evaluate("undefined_var").is_err());
}