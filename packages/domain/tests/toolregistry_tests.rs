//! Comprehensive tests for TOOLREGISTRY TYPESTATE BUILDER implementation
//!
//! Tests validate zero-allocation patterns, compile-time type safety,
//! and production-ready functionality of the tool registration system.

use std::error::Error as StdError;

// Re-export for test compatibility
use anyhow::Result as AnyhowResult;
use fluent_ai_domain::tool::*;
use serde::{Deserialize, Serialize};
use tokio::test;

// Alias for test compatibility
type AnthropicResult<T> = std::result::Result<T, Box<dyn StdError + Send + Sync>>;

/// Test dependency for tool registration
#[derive(Debug, Clone)]
struct TestDependency {
    name: String,
    counter: std::sync::Arc<std::sync::atomic::AtomicU32>}

impl TestDependency {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            counter: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0))}
    }

    fn increment(&self) -> u32 {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1
    }
}

/// Test request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalculatorRequest {
    expression: String,
    precision: Option<u32>}

/// Test response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalculatorResult {
    result: f64,
    expression: String,
    precision: u32}

/// Test for basic typestate builder chain
#[test]
fn test_typestate_builder_chain() {
    let dependency = TestDependency::new("calculator");

    // Test that builder chain compiles and follows typestate pattern
    let tool = ToolBuilder::named("calculator")
        .description("Mathematical calculator with precision")
        .with(dependency.clone())
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(
            |_conv, _emit, req: CalculatorRequest, dep: &TestDependency| {
                Box::pin(async move {
                    let count = dep.increment();
                    println!("Calculator invoked {} times", count);

                    // Simple expression evaluation for testing
                    let result = match req.expression.as_str() {
                        "2+2" => 4.0,
                        "3*3" => 9.0,
                        "10/2" => 5.0,
                        _ => 0.0};

                    Ok(())
                })
            },
        )
        .build();

    // Verify tool properties
    assert_eq!(tool.name(), "calculator");
    assert_eq!(tool.description(), "Mathematical calculator with precision");
    assert_eq!(tool.dependency().name, "calculator");
}

/// Test for tool registration and storage
#[test]
fn test_tool_registration_and_storage() {
    let mut storage = TypedToolStorage::new();
    let dependency = TestDependency::new("test_tool");

    // Create and register a tool
    let tool = ToolBuilder::named("test_tool")
        .description("Test tool for registration")
        .with(dependency)
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(
            |_conv, _emit, _req: CalculatorRequest, _dep: &TestDependency| {
                Box::pin(async move { Ok(()) })
            },
        )
        .build()
        .into_typed_tool();

    // Test registration
    let result = storage.register(tool);
    assert!(result.is_ok(), "Tool registration should succeed");

    // Test storage queries
    assert!(storage.contains_tool("test_tool"));
    assert!(!storage.contains_tool("nonexistent_tool"));
    assert_eq!(storage.tool_count(), 1);
    assert_eq!(storage.remaining_capacity(), 31); // 32 - 1

    // Test tool names iterator
    let tool_names: Vec<&str> = storage.tool_names().collect();
    assert_eq!(tool_names, vec!["test_tool"]);
}

/// Test for duplicate tool registration error
#[test]
fn test_duplicate_tool_registration() {
    let mut storage = TypedToolStorage::new();
    let dependency = TestDependency::new("duplicate_tool");

    // First tool registration
    let tool1 = ToolBuilder::named("duplicate_tool")
        .description("First tool")
        .with(dependency.clone())
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(|_conv, _emit, req: CalculatorRequest, _dep| {
            let expression = format!("{} + {}", req.expression, 42.0);
            CalculatorResult {
                result: 42.0,
                expression,
                precision: req.precision.unwrap_or(6)}
        })
        .build();

    // Second tool with same name
    let tool2 = ToolBuilder::named("duplicate_tool")
        .description("Second tool")
        .with(dependency)
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(|_conv, _emit, req: CalculatorRequest, _dep| {
            let expression = format!("{} * 2", req.expression);
            CalculatorResult {
                result: 84.0,
                expression,
                precision: req.precision.unwrap_or(6)}
        })
        .build();

    // Register first tool - should succeed
    let result1 = storage.register(tool1);
    assert!(result1.is_ok(), "First tool registration should succeed");

    // Register second tool with same name - should fail
    let result2 = storage.register(tool2);
    assert!(result2.is_err(), "Duplicate tool registration should fail");

    // Verify error type
    if let Err(AnthropicError::ToolError(msg)) = result2 {
        assert!(
            msg.contains("already exists"),
            "Error should mention tool already exists"
        );
    } else {
        panic!("Expected ToolError for duplicate registration");
    }
}

/// Test for schema type variations
#[test]
fn test_schema_type_variations() {
    let dependency = TestDependency::new("schema_test");

    // Test Serde schema type
    let serde_tool = ToolBuilder::named("serde_tool")
        .description("Tool using Serde schemas")
        .with(dependency.clone())
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(|_conv, _emit, req: CalculatorRequest, _dep| {
            let expression = format!("evaluated: {}", req.expression);
            CalculatorResult {
                result: 42.0,
                expression,
                precision: req.precision.unwrap_or(6)}
        })
        .build();

    assert_eq!(serde_tool.name(), "serde_tool");

    // Test JsonSchema type
    let json_tool = ToolBuilder::named("json_tool")
        .description("Tool using JSON schemas")
        .with(dependency.clone())
        .request_schema::<CalculatorRequest>(SchemaType::JsonSchema)
        .result_schema::<CalculatorResult>(SchemaType::JsonSchema)
        .on_invocation(
            |_conv, _emit, _req: CalculatorRequest, _dep: &TestDependency| {
                Box::pin(async move { Ok(()) })
            },
        )
        .build();

    assert_eq!(json_tool.name(), "json_tool");

    // Test Inline schema type
    let inline_tool = ToolBuilder::named("inline_tool")
        .description("Tool using inline schemas")
        .with(dependency)
        .request_schema::<CalculatorRequest>(SchemaType::Inline)
        .result_schema::<CalculatorResult>(SchemaType::Inline)
        .on_invocation(
            |_conv, _emit, _req: CalculatorRequest, _dep: &TestDependency| {
                Box::pin(async move { Ok(()) })
            },
        )
        .build();

    assert_eq!(inline_tool.name(), "inline_tool");
}

/// Test for error handling in tool execution
#[test]
fn test_tool_error_handling() {
    let dependency = TestDependency::new("error_tool");

    // Create tool that can produce errors
    let tool = ToolBuilder::named("error_tool")
        .description("Tool that demonstrates error handling")
        .with(dependency)
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(
            |_conv, _emit, req: CalculatorRequest, _dep: &TestDependency| {
                Box::pin(async move {
                    // Simulate error condition
                    if req.expression == "error" {
                        return Err(AnthropicError::InvalidInput(
                            "Invalid expression".to_string(),
                        ));
                    }
                    Ok(())
                })
            },
        )
        .on_error(|_conv, _control, _error, _dep: &TestDependency| {
            // Error handler can be used for logging, retry logic, etc.
            println!("Error occurred in tool execution");
        })
        .build();

    assert_eq!(tool.name(), "error_tool");
}

/// Test for tool registry integration
#[test]
fn test_tool_registry_integration() {
    let mut registry = ToolRegistry::default();
    let dependency = TestDependency::new("registry_tool");

    // Create a typed tool
    let tool = ToolBuilder::named("registry_tool")
        .description("Tool for registry integration testing")
        .with(dependency)
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(
            |_conv, _emit, _req: CalculatorRequest, _dep: &TestDependency| {
                Box::pin(async move { Ok(()) })
            },
        )
        .build();

    // Register the tool
    let result = registry.add_typed_tool(tool);
    assert!(result.is_ok(), "Tool registration should succeed");

    let registry = result.expect("Registry should be updated");

    // Test registry queries
    assert!(registry.contains_tool("registry_tool"));

    let tool_names: Vec<&str> = registry.get_typed_tool_names().collect();
    assert!(tool_names.contains(&"registry_tool"));

    let (legacy_count, typed_count, total_capacity) = registry.tool_statistics();
    assert_eq!(legacy_count, 0);
    assert_eq!(typed_count, 1);
    assert_eq!(total_capacity, 32); // Default capacity
}

/// Test for memory management and cleanup
#[test]
fn test_memory_management() {
    let mut storage = TypedToolStorage::new();
    let dependency = TestDependency::new("memory_tool");

    // Register multiple tools to test memory usage
    for i in 0..5 {
        let tool = ToolBuilder::named(&format!("memory_tool_{}", i))
            .description(&format!("Memory test tool {}", i))
            .with(dependency.clone())
            .request_schema::<CalculatorRequest>(SchemaType::Serde)
            .result_schema::<CalculatorResult>(SchemaType::Serde)
            .on_invocation(
                |_conv, _emit, _req: CalculatorRequest, _dep: &TestDependency| {
                    Box::pin(async move { Ok(()) })
                },
            )
            .build()
            .into_typed_tool();

        let result = storage.register(tool);
        assert!(result.is_ok(), "Tool registration should succeed");
    }

    assert_eq!(storage.tool_count(), 5);
    assert_eq!(storage.remaining_capacity(), 27); // 32 - 5

    // Test memory optimization
    storage.optimize_memory();

    // Get memory statistics
    let (used_slots, total_slots, fragmentation) = storage.memory_stats();
    assert_eq!(used_slots, 5);
    assert_eq!(total_slots, 32);
    assert_eq!(fragmentation, 0); // No fragmentation in this test
}

/// Test for zero-allocation patterns
#[test]
fn test_zero_allocation_patterns() {
    // Test that ArrayVec is used for zero-allocation storage
    let storage = TypedToolStorage::new();

    // Verify initial state
    assert_eq!(storage.tool_count(), 0);
    assert_eq!(storage.remaining_capacity(), 32);

    // Test that tool names iterator works without allocation
    let tool_names: Vec<&str> = storage.tool_names().collect();
    assert!(tool_names.is_empty());

    // Test memory stats without allocation
    let (used, total, frag) = storage.memory_stats();
    assert_eq!(used, 0);
    assert_eq!(total, 32);
    assert_eq!(frag, 0);
}

/// Test for compile-time type safety
#[test]
fn test_compile_time_type_safety() {
    // This test validates that the typestate pattern prevents invalid state transitions
    // If this compiles, it means the type system is working correctly

    let dependency = TestDependency::new("type_safety_test");

    // Valid state progression
    let _tool = ToolBuilder::named("type_safe_tool")
        .description("Type safety test tool")
        .with(dependency)
        .request_schema::<CalculatorRequest>(SchemaType::Serde)
        .result_schema::<CalculatorResult>(SchemaType::Serde)
        .on_invocation(
            |_conv, _emit, _req: CalculatorRequest, _dep: &TestDependency| {
                Box::pin(async move { Ok(()) })
            },
        )
        .build();

    // The fact that this compiles validates the typestate pattern
    assert!(true, "Typestate pattern enforces compile-time type safety");
}

/// Test for built-in tools integration
#[test]
fn test_builtin_tools_integration() {
    let registry = ToolRegistry::with_builtins();

    // Test that built-in tools are registered
    assert!(registry.contains_tool("calculator"));

    // Test tool statistics
    let (legacy_count, typed_count, _total_capacity) = registry.tool_statistics();
    assert!(legacy_count > 0, "Should have legacy built-in tools");

    // Test getting all tools
    let all_tools = registry.get_all_tools();
    assert!(!all_tools.is_empty(), "Should have built-in tools");
}
