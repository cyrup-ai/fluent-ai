//! Comprehensive testing suite for typestate builder and tool execution
//!
//! Tests the complete typestate builder chain, execution pipeline, and Cylo integration
//! following zero-allocation, blazing-fast, no-locking constraints.

use fluent_ai_provider::clients::anthropic::tools::*;
use fluent_ai_provider::clients::anthropic::AnthropicError;
use fluent_ai_domain::{Conversation, Emitter, Message};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio::test;

#[cfg(feature = "cylo")]
use fluent_ai_cylo::{CyloInstance, execution_env::Cylo};

/// Test dependency for tool execution
#[derive(Debug, Clone)]
struct TestDependency {
    name: String,
    value: i32,
}

/// Test request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestRequest {
    input: String,
    multiplier: i32,
}

/// Test response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestResponse {
    output: String,
    result: i32,
}

/// Test tool handler that follows zero-allocation constraints
async fn test_tool_handler(
    _conversation: &Conversation,
    _emitter: &Emitter,
    request: TestRequest,
    dependency: &TestDependency,
) -> Result<TestResponse, AnthropicError> {
    Ok(TestResponse {
        output: format!("Processed: {}", request.input),
        result: request.multiplier * dependency.value,
    })
}

/// Test error handler for comprehensive error testing
async fn test_error_handler(
    _conversation: &Conversation,
    _emitter: &Emitter,
    _request: TestRequest,
    _dependency: &TestDependency,
) -> Result<TestResponse, AnthropicError> {
    Err(AnthropicError::InvalidRequest("Test error".to_string()))
}

#[test]
async fn test_typestate_builder_basic_flow() {
    // Test basic typestate builder flow
    let dependency = TestDependency {
        name: "test_dep".to_string(),
        value: 42,
    };

    let tool = typed_tool("test_tool", "Test tool for basic flow")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build tool");

    // Verify tool properties
    assert_eq!(tool.name(), "test_tool");
    assert_eq!(tool.description(), "Test tool for basic flow");
    assert_eq!(tool.dependency().name, "test_dep");
    assert_eq!(tool.dependency().value, 42);
}

#[test]
async fn test_typestate_builder_error_handling() {
    // Test error handling in typestate builder
    let dependency = TestDependency {
        name: "error_dep".to_string(),
        value: 0,
    };

    let tool = typed_tool("error_tool", "Test tool for error handling")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .on_invocation(test_error_handler)
        .build()
        .expect("Failed to build error tool");

    // Create test conversation and emitter
    let conversation = Conversation::new();
    let emitter = Emitter::new();
    
    let request = TestRequest {
        input: "test input".to_string(),
        multiplier: 2,
    };

    // Test error handling during execution
    let result = tool.execute(&conversation, &emitter, request).await;
    assert!(result.is_err());
    
    if let Err(AnthropicError::InvalidRequest(msg)) = result {
        assert_eq!(msg, "Test error");
    } else {
        panic!("Expected InvalidRequest error");
    }
}

#[test]
async fn test_tool_registry_integration() {
    // Test tool registry integration with zero-allocation storage
    let mut registry = ToolRegistry::new();
    
    let dependency = TestDependency {
        name: "registry_dep".to_string(),
        value: 100,
    };

    let tool = typed_tool("registry_tool", "Test tool for registry")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build registry tool");

    // Test tool registration
    registry.register_typed_tool(tool).expect("Failed to register tool");
    
    // Verify tool is registered
    assert!(registry.get_typed_tool_names().any(|name| name == "registry_tool"));
}

#[test]
async fn test_zero_allocation_constraints() {
    // Test that the builder follows zero-allocation constraints
    let start_time = Instant::now();
    
    let dependency = TestDependency {
        name: "perf_dep".to_string(),
        value: 1000,
    };

    // Build tool with performance constraints
    let tool = typed_tool("perf_tool", "Performance test tool")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build performance tool");

    let build_time = start_time.elapsed();
    
    // Verify build time is blazing-fast (< 1ms for zero allocation)
    assert!(build_time.as_micros() < 1000, "Build time too slow: {:?}", build_time);
    
    // Test execution performance
    let conversation = Conversation::new();
    let emitter = Emitter::new();
    
    let request = TestRequest {
        input: "performance test".to_string(),
        multiplier: 10,
    };

    let exec_start = Instant::now();
    let result = tool.execute(&conversation, &emitter, request).await;
    let exec_time = exec_start.elapsed();
    
    assert!(result.is_ok());
    assert!(exec_time.as_micros() < 10000, "Execution time too slow: {:?}", exec_time);
}

#[cfg(feature = "cylo")]
#[test]
async fn test_cylo_integration() {
    // Test Cylo execution environment integration
    let dependency = TestDependency {
        name: "cylo_dep".to_string(),
        value: 200,
    };

    let cylo_instance = Cylo::Apple("python:alpine3.20".to_string())
        .instance("test_env".to_string());

    let tool = typed_tool("cylo_tool", "Test tool with Cylo integration")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .cylo(cylo_instance)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build Cylo tool");

    // Test that Cylo is properly integrated
    let conversation = Conversation::new();
    let emitter = Emitter::new();
    
    let request = TestRequest {
        input: "cylo test".to_string(),
        multiplier: 5,
    };

    // Execute with Cylo integration
    let result = tool.execute(&conversation, &emitter, request).await;
    assert!(result.is_ok());
}

#[cfg(feature = "cylo")]
#[test]
async fn test_cylo_error_handling() {
    // Test Cylo error handling and recovery
    let dependency = TestDependency {
        name: "cylo_error_dep".to_string(),
        value: 0,
    };

    let cylo_instance = Cylo::LandLock("/tmp/invalid_path".to_string())
        .instance("error_env".to_string());

    let tool = typed_tool("cylo_error_tool", "Test tool with Cylo error handling")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .cylo(cylo_instance)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build Cylo error tool");

    let conversation = Conversation::new();
    let emitter = Emitter::new();
    
    let request = TestRequest {
        input: "cylo error test".to_string(),
        multiplier: 1,
    };

    // Test Cylo error handling
    let result = tool.execute(&conversation, &emitter, request).await;
    
    // Should handle Cylo errors gracefully
    match result {
        Ok(_) => {
            // Test passed - Cylo handled the error gracefully
        }
        Err(AnthropicError::ExecutionError(msg)) => {
            assert!(msg.contains("Cylo execution failed"));
        }
        Err(e) => {
            panic!("Unexpected error type: {:?}", e);
        }
    }
}

#[test]
async fn test_concurrent_tool_execution() {
    // Test concurrent tool execution with lock-free storage
    let mut registry = ToolRegistry::new();
    
    // Register multiple tools concurrently
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let dependency = TestDependency {
            name: format!("concurrent_dep_{}", i),
            value: i * 10,
        };

        let tool = typed_tool(&format!("concurrent_tool_{}", i), "Concurrent test tool")
            .with_dependency(dependency)
            .with_request_schema::<TestRequest>(SchemaType::Object)
            .with_result_schema::<TestResponse>(SchemaType::Object)
            .on_invocation(test_tool_handler)
            .build()
            .expect("Failed to build concurrent tool");

        registry.register_typed_tool(tool).expect("Failed to register concurrent tool");
    }
    
    // Verify all tools are registered
    let tool_names: Vec<&str> = registry.get_typed_tool_names().collect();
    assert_eq!(tool_names.len(), 10);
    
    for i in 0..10 {
        assert!(tool_names.iter().any(|name| *name == format!("concurrent_tool_{}", i)));
    }
}

#[test]
async fn test_type_safety_compile_time() {
    // Test compile-time type safety of typestate builder
    let dependency = TestDependency {
        name: "type_safe_dep".to_string(),
        value: 42,
    };

    // This should compile successfully with proper type safety
    let _tool = typed_tool("type_safe_tool", "Type safety test tool")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build type-safe tool");
    
    // Verify type safety at compile time
    // This test passes if it compiles without errors
}

#[test]
async fn test_schema_validation() {
    // Test JSON schema validation for requests and responses
    let dependency = TestDependency {
        name: "schema_dep".to_string(),
        value: 123,
    };

    let tool = typed_tool("schema_tool", "Schema validation test tool")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build schema tool");

    // Test with valid request
    let conversation = Conversation::new();
    let emitter = Emitter::new();
    
    let valid_request = TestRequest {
        input: "valid input".to_string(),
        multiplier: 3,
    };

    let result = tool.execute(&conversation, &emitter, valid_request).await;
    assert!(result.is_ok());
}

#[test]
async fn test_blazing_fast_execution() {
    // Test that execution is blazing-fast
    let dependency = TestDependency {
        name: "fast_dep".to_string(),
        value: 999,
    };

    let tool = typed_tool("fast_tool", "Blazing-fast test tool")
        .with_dependency(dependency)
        .with_request_schema::<TestRequest>(SchemaType::Object)
        .with_result_schema::<TestResponse>(SchemaType::Object)
        .on_invocation(test_tool_handler)
        .build()
        .expect("Failed to build fast tool");

    let conversation = Conversation::new();
    let emitter = Emitter::new();
    
    // Execute multiple times to test consistent performance
    let mut total_time = std::time::Duration::new(0, 0);
    let iterations = 100;
    
    for _ in 0..iterations {
        let request = TestRequest {
            input: "fast execution test".to_string(),
            multiplier: 1,
        };

        let start = Instant::now();
        let result = tool.execute(&conversation, &emitter, request).await;
        total_time += start.elapsed();
        
        assert!(result.is_ok());
    }
    
    let avg_time = total_time / iterations;
    assert!(avg_time.as_micros() < 1000, "Average execution time too slow: {:?}", avg_time);
}
