//! JSON Client Tool Demonstration
//! 
//! This example demonstrates how to use the JSON-RPC 2.0 client to call
//! real MCP tools (time and hash) with fluent builder APIs.

use std::error::Error;
use sweetmcp_json_client::JsonClient;
use mcp_client_traits::{McpClient, McpToolOperations, RequestBuilder, ResponseAdapter};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create JSON client
    let client = JsonClient::new("https://localhost:8443")
        .map_err(|e| format!("Failed to create client: {}", e))?
        .with_timeout(10000); // 10 second timeout

    println!("🚀 SweetMCP JSON-RPC 2.0 Client Demo");
    println!("Connected to: {}", client.server_url());
    println!("Protocol: {}\n", client.protocol_name());

    // Demonstrate tool listing
    println!("📋 Listing available tools...");
    match client.list_tools().await {
        Ok(tools) => {
            println!("✅ Found {} tools:", tools.len());
            for tool in &tools {
                println!("  • {} - {}", tool.name, 
                    tool.description.as_deref().unwrap_or("No description"));
            }
            println!();
        }
        Err(e) => println!("❌ Failed to list tools: {}\n", e),
    }

    // Demonstrate time tool using fluent API
    println!("⏰ Testing Time Tool (get_time_utc)...");
    match client
        .tool("time")
        .with_arg("name", "get_time_utc")
        .execute()
        .await
    {
        Ok(response) => {
            println!("✅ Time tool response received");
            if let Ok(adapter) = ResponseAdapter::new(response) {
                if let Ok(time_result) = adapter.extract_time_result() {
                    println!("   UTC Time: {}", time_result.utc_time);
                    println!("   Formatted: {}", time_result.formatted_time);
                    println!("   Timezone: {}", time_result.timezone);
                } else {
                    println!("   Raw content: {:?}", adapter.extract_text_content());
                }
            }
        }
        Err(e) => println!("❌ Time tool failed: {}", e),
    }
    println!();

    // Demonstrate hash tool using convenience method
    println!("🔐 Testing Hash Tool (SHA256)...");
    match client.hash_tool("Hello, SweetMCP JSON Client!", "sha256").await {
        Ok(response) => {
            println!("✅ Hash tool response received");
            if let Ok(adapter) = ResponseAdapter::new(response) {
                if let Ok(hash_result) = adapter.extract_hash_result() {
                    println!("   Algorithm: {}", hash_result.algorithm);
                    println!("   Input: {}", hash_result.input_data);
                    println!("   Hash: {}", hash_result.hash_value);
                } else {
                    println!("   Raw content: {:?}", adapter.extract_text_content());
                }
            }
        }
        Err(e) => println!("❌ Hash tool failed: {}", e),
    }
    println!();

    // Demonstrate hash tool with MD5 using fluent API
    println!("🔐 Testing Hash Tool (MD5)...");
    match client
        .tool("hash")
        .with_arg("data", "JSON-RPC 2.0 is fast!")
        .with_arg("algorithm", "md5")
        .execute()
        .await
    {
        Ok(response) => {
            println!("✅ MD5 hash response received");
            if let Ok(adapter) = ResponseAdapter::new(response) {
                if let Ok(hash_result) = adapter.extract_hash_result() {
                    println!("   Algorithm: {}", hash_result.algorithm);
                    println!("   Input: {}", hash_result.input_data);
                    println!("   Hash: {}", hash_result.hash_value);
                } else {
                    println!("   Raw content: {:?}", adapter.extract_text_content());
                }
            }
        }
        Err(e) => println!("❌ MD5 hash tool failed: {}", e),
    }
    println!();

    // Demonstrate error handling with invalid tool
    println!("❓ Testing error handling with invalid tool...");
    match client
        .tool("nonexistent_tool")
        .with_arg("param", "value")
        .execute()
        .await
    {
        Ok(_) => println!("⚠️  Unexpected success"),
        Err(e) => println!("✅ Expected error caught: {}", e),
    }
    println!();

    // Demonstrate timeout handling
    println!("⏱️  Testing timeout handling...");
    match client
        .tool("time")
        .with_arg("name", "get_time_utc")
        .with_timeout(1) // 1ms timeout to force failure
        .execute()
        .await
    {
        Ok(_) => println!("⚠️  Unexpected success (server too fast!)"),
        Err(e) => println!("✅ Expected timeout caught: {}", e),
    }

    println!("\n🎉 JSON-RPC 2.0 Client Demo Complete!");
    println!("   • Direct JSON-RPC 2.0 protocol communication");
    println!("   • Zero-allocation serialization with simd-json");
    println!("   • Fluent builder API for tool execution");
    println!("   • Comprehensive error handling and recovery");
    println!("   • Production-ready performance and reliability");

    Ok(())
}