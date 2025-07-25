//! Example demonstrating secure tool execution using cylo integration
//!
//! This example shows how to:
//! 1. Create secure MCP tools for code execution
//! 2. Configure security settings
//! 3. Execute code safely in sandboxed environments

use fluent_ai_domain::{
    McpToolTrait as Tool, SecureExecutionConfig, SecureMcpTool, SecureMcpToolBuilder,
    set_secure_executor_config};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("🔒 Fluent AI Secure Code Execution Example");
    println!("==========================================");

    // Example 1: Create a Python executor with default security settings
    println!("\n📝 Example 1: Python Code Execution");
    let python_tool = SecureMcpTool::python_executor();

    let python_code = json!({
        "code": "print('Hello from secure Python execution!')\nprint('Math calculation:', 2 + 2)",
        "language": "python"
    });

    match python_tool.execute(python_code).await {
        Ok(result) => println!("✅ Python execution result: {}", result),
        Err(e) => println!("❌ Python execution error: {}", e)}

    // Example 2: Multi-language executor
    println!("\n🔧 Example 2: Multi-language Code Executor");
    let multi_tool = SecureMcpTool::multi_language_executor();

    // JavaScript example
    let js_code = json!({
        "code": "console.log('Hello from secure JavaScript!'); console.log('Result:', Math.sqrt(16));",
        "language": "javascript"
    });

    match multi_tool.execute(js_code).await {
        Ok(result) => println!("✅ JavaScript execution result: {}", result),
        Err(e) => println!("❌ JavaScript execution error: {}", e)}

    // Example 3: Custom security configuration
    println!("\n🛡️  Example 3: Custom Security Configuration");

    let secure_config = SecureExecutionConfig {
        use_firecracker: false, // Keep false for compatibility
        use_landlock: true,
        timeout_seconds: 10,
        memory_limit_mb: Some(256),
        cpu_limit: Some(1)};

    // Set global configuration (can only be done once)
    if let Err(_) = set_secure_executor_config(secure_config.clone()) {
        println!("⚠️  Global executor already configured");
    }

    let custom_tool = SecureMcpToolBuilder::new()
        .name("secure_bash_tool")
        .description("Secure bash execution with custom limits")
        .parameters(json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Bash script to execute"},
                "language": {"type": "string", "default": "bash"}
            },
            "required": ["code"]
        }))
        .memory_limit(256)
        .timeout(10)
        .build();

    let bash_code = json!({
        "code": "echo 'Hello from secure bash!'; date; echo 'Directory listing:'; ls -la",
        "language": "bash"
    });

    match custom_tool.execute(bash_code).await {
        Ok(result) => println!("✅ Bash execution result: {}", result),
        Err(e) => println!("❌ Bash execution error: {}", e)}

    // Example 4: Rust code execution
    println!("\n🦀 Example 4: Rust Code Execution");
    let rust_tool = SecureMcpTool::rust_executor();

    let rust_code = json!({
        "code": r#"
fn main() {
    println!("Hello from secure Rust execution!");
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter().sum();
    println!("Sum of numbers: {}", sum);
}
"#,
        "language": "rust"
    });

    match rust_tool.execute(rust_code).await {
        Ok(result) => println!("✅ Rust execution result: {}", result),
        Err(e) => println!("❌ Rust execution error: {}", e)}

    // Example 5: Tool inspection
    println!("\n🔍 Example 5: Tool Inspection");
    let tools = vec![
        SecureMcpTool::python_executor(),
        SecureMcpTool::javascript_executor(),
        SecureMcpTool::bash_executor(),
        SecureMcpTool::rust_executor(),
        SecureMcpTool::go_executor(),
    ];

    for tool in tools {
        println!("Tool: {} - {}", tool.name(), tool.description());
        println!(
            "  Parameters: {}",
            serde_json::to_string_pretty(tool.parameters())?
        );
    }

    println!("\n🎉 All examples completed!");
    println!(
        "\nNote: Some executions may fail in environments without proper language runtimes installed."
    );
    println!(
        "This is expected behavior - cylo provides secure sandboxing when runtimes are available."
    );

    Ok(())
}
