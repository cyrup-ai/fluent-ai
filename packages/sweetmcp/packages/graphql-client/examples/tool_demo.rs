//! GraphQL Client Tool Demonstration
//! 
//! This example demonstrates how to use the GraphQL client to execute
//! MCP tools through GraphQL queries and mutations with type-safe schemas.

use std::error::Error;
use sweetmcp_graphql_client::GraphQLClient;
use mcp_client_traits::{McpClient, McpToolOperations};
use tracing_subscriber;
use async_graphql::Variables;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create GraphQL client
    let client = GraphQLClient::new("https://localhost:8443")
        .await
        .map_err(|e| format!("Failed to create client: {}", e))?
        .with_timeout(10000); // 10 second timeout

    println!("🚀 SweetMCP GraphQL Client Demo");
    println!("Connected to: {}", "https://localhost:8443");
    println!("Protocol: GraphQL over JSON-RPC 2.0\n");

    // Display GraphQL schema
    println!("📋 Generated GraphQL Schema:");
    println!("{}", client.get_schema_sdl());
    println!();

    // Demonstrate tool listing via GraphQL query
    println!("📋 Listing available tools via GraphQL...");
    let tools_query = r#"
        query {
            tools {
                name
                description
                input_schema
            }
        }
    "#;

    match client.execute_query(tools_query, None).await {
        Ok(response) => {
            println!("✅ Tools query successful:");
            println!("{}", response);
        }
        Err(e) => println!("❌ Tools query failed: {}", e),
    }
    println!();

    // Demonstrate time tool via GraphQL mutation
    println!("⏰ Testing Time Tool via GraphQL mutation...");
    let time_query = r#"
        mutation {
            timeResult: time_tool(operation: "get_time_utc") {
                ... on TimeResult {
                    utc_time
                    formatted_time
                    timezone
                    unix_timestamp
                }
                ... on GenericResult {
                    success
                    error_message
                }
            }
        }
    "#;

    match client.execute_query(time_query, None).await {
        Ok(response) => {
            println!("✅ Time tool GraphQL mutation successful:");
            println!("{}", response);
        }
        Err(e) => println!("❌ Time tool GraphQL mutation failed: {}", e),
    }
    println!();

    // Demonstrate hash tool via GraphQL mutation
    println!("🔐 Testing Hash Tool via GraphQL mutation...");
    let hash_query = r#"
        mutation {
            hashResult: hash_tool(data: "Hello GraphQL MCP!", algorithm: "sha256") {
                ... on HashResult {
                    algorithm
                    input_data
                    hash_value
                    hash_length
                }
                ... on GenericResult {
                    success
                    error_message
                }
            }
        }
    "#;

    match client.execute_query(hash_query, None).await {
        Ok(response) => {
            println!("✅ Hash tool GraphQL mutation successful:");
            println!("{}", response);
        }
        Err(e) => println!("❌ Hash tool GraphQL mutation failed: {}", e),
    }
    println!();

    // Demonstrate parameterized query with variables
    println!("🔐 Testing Hash Tool with GraphQL variables...");
    let hash_query_with_vars = r#"
        mutation HashData($data: String!, $algorithm: String!) {
            hashResult: hash_tool(data: $data, algorithm: $algorithm) {
                ... on HashResult {
                    algorithm
                    input_data
                    hash_value
                    hash_length
                }
                ... on GenericResult {
                    success
                    error_message
                }
            }
        }
    "#;

    let mut variables = Variables::new();
    variables.insert("data".into(), "GraphQL variables work!".into());
    variables.insert("algorithm".into(), "md5".into());

    match client.execute_query(hash_query_with_vars, Some(variables)).await {
        Ok(response) => {
            println!("✅ Hash tool with variables successful:");
            println!("{}", response);
        }
        Err(e) => println!("❌ Hash tool with variables failed: {}", e),
    }
    println!();

    // Demonstrate generic tool execution
    println!("⚙️  Testing generic tool execution...");
    let generic_query = r#"
        mutation {
            result: execute_tool(tool_name: "time", args_json: "{\"name\": \"get_time_utc\"}") {
                ... on GenericResult {
                    tool_name
                    success
                    content
                    error_message
                }
            }
        }
    "#;

    match client.execute_query(generic_query, None).await {
        Ok(response) => {
            println!("✅ Generic tool execution successful:");
            println!("{}", response);
        }
        Err(e) => println!("❌ Generic tool execution failed: {}", e),
    }
    println!();

    // Demonstrate error handling with invalid query
    println!("❓ Testing error handling with invalid GraphQL...");
    let invalid_query = r#"
        mutation {
            nonexistent_field {
                invalid
            }
        }
    "#;

    match client.execute_query(invalid_query, None).await {
        Ok(response) => {
            println!("✅ Error properly captured in GraphQL response:");
            println!("{}", response);
        }
        Err(e) => println!("❌ Unexpected error during invalid query: {}", e),
    }
    println!();

    // Test MCP client methods directly
    println!("🔄 Testing direct MCP client methods...");
    match client.list_tools().await {
        Ok(tools) => {
            println!("✅ Direct list_tools() successful: {} tools found", tools.len());
            for tool in tools.iter().take(3) {
                println!("   • {}: {}", tool.name, 
                    tool.description.as_deref().unwrap_or("No description"));
            }
        }
        Err(e) => println!("❌ Direct list_tools() failed: {}", e),
    }

    println!("\n🎉 GraphQL Client Demo Complete!");
    println!("   • Type-safe GraphQL schema generation");
    println!("   • GraphQL queries and mutations for MCP tools");
    println!("   • Union types for polymorphic tool responses");
    println!("   • Variable support for parameterized queries");
    println!("   • Comprehensive error handling in GraphQL format");
    println!("   • Dual interface: GraphQL + direct MCP client methods");

    Ok(())
}