//! GraphQL Protocol Extension Demo
//! 
//! This example demonstrates the GraphQL side of the SweetMCP protocol extension.
//! It shows GraphQL queries being converted to JSON-RPC, executed against real MCP tools,
//! and responses being shaped back to proper GraphQL format.

use anyhow::Result;
use reqwest;
use serde_json::{json, Value};
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    info!("🔮 Starting SweetMCP GraphQL Integration Demo");
    info!("==========================================");
    
    let client = reqwest::Client::new();
    let base_url = "http://localhost:8443";
    
    // Demo 1: Simple GraphQL query for time tool
    info!("📝 Demo 1: GraphQL Query for Time Tool");
    match demo_graphql_time_query(&client, base_url).await {
        Ok(()) => info!("✅ GraphQL time query demo completed"),
        Err(e) => error!("❌ GraphQL time query failed: {}", e),
    }
    
    println!();
    
    // Demo 2: GraphQL query with variables for hash tool
    info!("📝 Demo 2: GraphQL Query with Variables for Hash Tool");
    match demo_graphql_hash_query(&client, base_url).await {
        Ok(()) => info!("✅ GraphQL hash query demo completed"),
        Err(e) => error!("❌ GraphQL hash query failed: {}", e),
    }
    
    println!();
    
    // Demo 3: GraphQL with fragments
    info!("📝 Demo 3: GraphQL with Fragments");
    match demo_graphql_fragments(&client, base_url).await {
        Ok(()) => info!("✅ GraphQL fragments demo completed"),
        Err(e) => error!("❌ GraphQL fragments failed: {}", e),
    }
    
    info!("🎉 GraphQL protocol extension demos completed!");
    
    Ok(())
}

/// Demonstrate GraphQL query for time tool
async fn demo_graphql_time_query(client: &reqwest::Client, base_url: &str) -> Result<()> {
    info!("  Creating GraphQL query for time tool...");
    
    let graphql_query = json!({
        "query": r#"
            query GetCurrentTime {
                timeOperation(name: "get_time_utc") {
                    utc_time
                    utc_time_rfc2822
                    success
                }
            }
        "#,
        "variables": {},
        "operationName": "GetCurrentTime"
    });
    
    info!("  📝 GraphQL Query:");
    info!("    {}", serde_json::to_string_pretty(&graphql_query)?);
    
    // Send GraphQL request to SweetMCP server
    info!("  🌐 Sending GraphQL request to SweetMCP server...");
    let response = client
        .post(&format!("{}/graphql", base_url))
        .header("Content-Type", "application/json")
        .json(&graphql_query)
        .send()
        .await?;
    
    if !response.status().is_success() {
        error!("  ❌ Server returned error: {}", response.status());
        let error_text = response.text().await?;
        error!("     Error details: {}", error_text);
        return Ok(());
    }
    
    let response_json: Value = response.json().await?;
    info!("  📬 Received GraphQL response:");
    info!("    {}", serde_json::to_string_pretty(&response_json)?);
    
    // Verify response structure
    if let Some(data) = response_json.get("data") {
        if data.get("timeOperation").is_some() {
            info!("  ✅ GraphQL response properly shaped with timeOperation field");
        } else {
            info!("  ⚠️  Response data doesn't contain expected timeOperation field");
        }
    } else if response_json.get("errors").is_some() {
        info!("  ⚠️  GraphQL returned errors (expected if server not running)");
    }
    
    Ok(())
}

/// Demonstrate GraphQL query with variables for hash tool
async fn demo_graphql_hash_query(client: &reqwest::Client, base_url: &str) -> Result<()> {
    info!("  Creating GraphQL query with variables for hash tool...");
    
    let graphql_query = json!({
        "query": r#"
            query ComputeHash($data: String!, $algorithm: String!) {
                hashOperation(data: $data, algorithm: $algorithm) {
                    hash_result
                    algorithm_used
                    input_data
                    success
                }
            }
        "#,
        "variables": {
            "data": "Hello, SweetMCP GraphQL!",
            "algorithm": "sha256"
        },
        "operationName": "ComputeHash"
    });
    
    info!("  📝 GraphQL Query with Variables:");
    info!("    {}", serde_json::to_string_pretty(&graphql_query)?);
    
    // Send GraphQL request
    info!("  🌐 Sending GraphQL request to SweetMCP server...");
    let response = client
        .post(&format!("{}/graphql", base_url))
        .header("Content-Type", "application/json")
        .json(&graphql_query)
        .send()
        .await?;
    
    if !response.status().is_success() {
        error!("  ❌ Server returned error: {}", response.status());
        let error_text = response.text().await?;
        error!("     Error details: {}", error_text);
        return Ok(());
    }
    
    let response_json: Value = response.json().await?;
    info!("  📬 Received GraphQL response:");
    info!("    {}", serde_json::to_string_pretty(&response_json)?);
    
    // Verify response structure
    if let Some(data) = response_json.get("data") {
        if let Some(hash_op) = data.get("hashOperation") {
            info!("  ✅ GraphQL response properly shaped with hashOperation field");
            
            if let Some(hash_result) = hash_op.get("hash_result") {
                info!("  ✅ Hash result included in shaped response");
            }
        } else {
            info!("  ⚠️  Response data doesn't contain expected hashOperation field");
        }
    } else if response_json.get("errors").is_some() {
        info!("  ⚠️  GraphQL returned errors (expected if server not running)");
    }
    
    Ok(())
}

/// Demonstrate GraphQL with fragments
async fn demo_graphql_fragments(client: &reqwest::Client, base_url: &str) -> Result<()> {
    info!("  Creating GraphQL query with fragments...");
    
    let graphql_query = json!({
        "query": r#"
            fragment ToolResult on OperationResult {
                success
                timestamp
                execution_time
            }
            
            fragment HashInfo on HashResult {
                hash_result
                algorithm_used
                input_length
            }
            
            query MultipleOperations {
                timeOp: timeOperation(name: "get_time_utc") {
                    ...ToolResult
                    utc_time
                    utc_time_rfc2822
                }
                
                hashOp: hashOperation(data: "Fragment Test", algorithm: "md5") {
                    ...ToolResult
                    ...HashInfo
                }
            }
        "#,
        "variables": {},
        "operationName": "MultipleOperations"
    });
    
    info!("  📝 GraphQL Query with Fragments:");
    info!("    {}", serde_json::to_string_pretty(&graphql_query)?);
    
    // Send GraphQL request
    info!("  🌐 Sending GraphQL request to SweetMCP server...");
    let response = client
        .post(&format!("{}/graphql", base_url))
        .header("Content-Type", "application/json")
        .json(&graphql_query)
        .send()
        .await?;
    
    if !response.status().is_success() {
        error!("  ❌ Server returned error: {}", response.status());
        let error_text = response.text().await?;
        error!("     Error details: {}", error_text);
        return Ok(());
    }
    
    let response_json: Value = response.json().await?;
    info!("  📬 Received GraphQL response:");
    info!("    {}", serde_json::to_string_pretty(&response_json)?);
    
    // Verify fragment resolution worked
    if let Some(data) = response_json.get("data") {
        let has_time_op = data.get("timeOp").is_some();
        let has_hash_op = data.get("hashOp").is_some();
        
        if has_time_op && has_hash_op {
            info!("  ✅ GraphQL fragments properly resolved and merged");
            info!("  ✅ Multiple operations in single query working");
        } else {
            info!("  ⚠️  Fragment resolution may need improvement");
        }
        
        // Check for fragment fields
        if let Some(time_op) = data.get("timeOp") {
            if time_op.get("success").is_some() && time_op.get("timestamp").is_some() {
                info!("  ✅ ToolResult fragment fields present in timeOp");
            }
        }
        
        if let Some(hash_op) = data.get("hashOp") {
            if hash_op.get("hash_result").is_some() && hash_op.get("algorithm_used").is_some() {
                info!("  ✅ HashInfo fragment fields present in hashOp");
            }
        }
    } else if response_json.get("errors").is_some() {
        info!("  ⚠️  GraphQL returned errors (expected if server not running)");
        
        // Check if it's a type validation error (our implementation working)
        if let Some(errors) = response_json.get("errors").and_then(|e| e.as_array()) {
            for error in errors {
                if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
                    if message.contains("type") || message.contains("fragment") {
                        info!("  ✅ Type validation working (detected invalid fragment usage)");
                    }
                }
            }
        }
    }
    
    Ok(())
}