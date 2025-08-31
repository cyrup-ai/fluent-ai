//! GraphQL Schema for MCP Tool Operations
//!
//! This module defines a GraphQL schema that maps MCP tools to GraphQL
//! queries and mutations, providing a type-safe interface for tool execution.

use async_graphql::{Context, Object, Result, Schema, SimpleObject, Union};
use std::collections::HashMap;

use sweet_mcp_type::{JsonValue, Response as McpResponse};
use mcp_client_traits::McpClient;

/// Root Query type for MCP operations
pub struct Query;

/// Root Mutation type for MCP tool execution
pub struct Mutation;

/// MCP tool information
#[derive(SimpleObject)]
pub struct ToolInfo {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: Option<String>,
    /// Input schema as JSON string
    pub input_schema: String,
}

/// Time tool result
#[derive(SimpleObject)]
pub struct TimeResult {
    /// UTC timestamp
    pub utc_time: String,
    /// Formatted time string
    pub formatted_time: String,
    /// Timezone
    pub timezone: String,
    /// Unix timestamp
    pub unix_timestamp: Option<i64>,
}

/// Hash tool result  
#[derive(SimpleObject)]
pub struct HashResult {
    /// The hash algorithm used
    pub algorithm: String,
    /// Input data that was hashed
    pub input_data: String,
    /// Resulting hash value
    pub hash_value: String,
    /// Hash length in characters
    pub hash_length: i32,
}

/// Generic tool execution result
#[derive(SimpleObject)]
pub struct GenericResult {
    /// Tool name that was executed
    pub tool_name: String,
    /// Success status
    pub success: bool,
    /// Result content as JSON string
    pub content: Option<String>,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Union type for tool execution results
#[derive(Union)]
pub enum ToolResult {
    Time(TimeResult),
    Hash(HashResult),
    Generic(GenericResult),
}

/// MCP client context for GraphQL resolvers
pub struct McpClientContext<C> {
    pub client: C,
}

impl<C> McpClientContext<C> {
    pub fn new(client: C) -> Self {
        Self { client }
    }
}

#[Object]
impl Query {
    /// List all available MCP tools
    async fn tools(&self, ctx: &Context<'_>) -> Result<Vec<ToolInfo>> {
        use crate::GraphQLClient;
        let client_ctx = ctx.data::<McpClientContext<GraphQLClient>>()?;
        let tools = client_ctx.client.list_tools().await
            .map_err(|e| async_graphql::Error::new(format!("Failed to list tools: {}", e)))?;

        Ok(tools.into_iter().map(|tool| ToolInfo {
            name: tool.name,
            description: tool.description,
            input_schema: tool.input_schema.encode(),
        }).collect())
    }

    /// Get server information
    async fn server_info(&self, ctx: &Context<'_>) -> Result<String> {
        use crate::GraphQLClient;
        let _client_ctx = ctx.data::<McpClientContext<GraphQLClient>>()?;
        // Try to ping the server to get basic info
        Ok(format!("GraphQL client connected to MCP server"))
    }
}

#[Object]
impl Mutation {
    /// Execute time tool operations
    async fn time_tool(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Operation name (e.g., \"get_time_utc\", \"parse_time\")")]
        operation: String,
        #[graphql(desc = "Optional time string for parsing operations")]
        time_string: Option<String>,
    ) -> Result<ToolResult> {
        use crate::GraphQLClient;
        let client_ctx = ctx.data::<McpClientContext<GraphQLClient>>()?;
        
        let mut args = HashMap::new();
        args.insert("name".to_string(), JsonValue::from(operation.clone()));
        if let Some(time_str) = time_string {
            args.insert("time_string".to_string(), JsonValue::from(time_str));
        }

        let response = client_ctx.client.call_tool("time", JsonValue::from(args)).await
            .map_err(|e| async_graphql::Error::new(format!("Time tool execution failed: {}", e)))?;

        match parse_time_response(response) {
            Ok(time_result) => Ok(ToolResult::Time(time_result)),
            Err(e) => Ok(ToolResult::Generic(GenericResult {
                tool_name: "time".to_string(),
                success: false,
                content: None,
                error_message: Some(e),
            })),
        }
    }

    /// Execute hash tool operations
    async fn hash_tool(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Data to hash")]
        data: String,
        #[graphql(desc = "Hash algorithm (e.g., \"sha256\", \"md5\", \"sha512\")")]
        algorithm: String,
    ) -> Result<ToolResult> {
        use crate::GraphQLClient;
        let client_ctx = ctx.data::<McpClientContext<GraphQLClient>>()?;
        
        let mut args = HashMap::new();
        args.insert("data".to_string(), JsonValue::from(data.clone()));
        args.insert("algorithm".to_string(), JsonValue::from(algorithm.clone()));

        let response = client_ctx.client.call_tool("hash", JsonValue::from(args)).await
            .map_err(|e| async_graphql::Error::new(format!("Hash tool execution failed: {}", e)))?;

        match parse_hash_response(response) {
            Ok(hash_result) => Ok(ToolResult::Hash(hash_result)),
            Err(e) => Ok(ToolResult::Generic(GenericResult {
                tool_name: "hash".to_string(),
                success: false,
                content: None,
                error_message: Some(e),
            })),
        }
    }

    /// Execute any MCP tool with generic arguments
    async fn execute_tool(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Tool name to execute")]
        tool_name: String,
        #[graphql(desc = "Tool arguments as JSON string")]
        args_json: String,
    ) -> Result<ToolResult> {
        use crate::GraphQLClient;
        let client_ctx = ctx.data::<McpClientContext<GraphQLClient>>()?;
        
        // Parse arguments JSON
        let mut args_bytes = args_json.as_bytes().to_vec();
        let args = simd_json::to_owned_value(&mut args_bytes)
            .map_err(|e| async_graphql::Error::new(format!("Invalid JSON arguments: {}", e)))?;

        let response = client_ctx.client.call_tool(&tool_name, args).await
            .map_err(|e| async_graphql::Error::new(format!("Tool execution failed: {}", e)))?;

        Ok(ToolResult::Generic(GenericResult {
            tool_name: tool_name.clone(),
            success: response.error.is_none(),
            content: response.result.map(|r| r.encode()),
            error_message: response.error.map(|e| e.message),
        }))
    }
}

/// Parse time tool response into structured result
fn parse_time_response(response: McpResponse) -> Result<TimeResult, String> {
    if let Some(error) = response.error {
        return Err(error.message);
    }

    let result = response.result.ok_or("No result in response")?;
    
    // Try to extract content from MCP response format
    if let Some(content_array) = result.get("content").and_then(|c| c.as_array()) {
        if let Some(first_item) = content_array.first() {
            if let Some(text_content) = first_item.get("text").and_then(|t| t.as_str()) {
                // Parse the JSON response from time tool
                let mut time_data_bytes = text_content.as_bytes().to_vec();
                if let Ok(time_data) = simd_json::to_owned_value(&mut time_data_bytes) {
                    return Ok(TimeResult {
                        utc_time: time_data.get("utc_time")
                            .and_then(|t| t.as_str())
                            .unwrap_or("unknown").to_string(),
                        formatted_time: time_data.get("formatted_time")
                            .and_then(|t| t.as_str())
                            .unwrap_or("unknown").to_string(),
                        timezone: time_data.get("timezone")
                            .and_then(|t| t.as_str())
                            .unwrap_or("UTC").to_string(),
                        unix_timestamp: time_data.get("unix_timestamp")
                            .and_then(|t| t.as_i64()),
                    });
                }
            }
        }
    }

    Err("Invalid time tool response format".to_string())
}

/// Parse hash tool response into structured result
fn parse_hash_response(response: McpResponse) -> Result<HashResult, String> {
    if let Some(error) = response.error {
        return Err(error.message);
    }

    let result = response.result.ok_or("No result in response")?;
    
    // Try to extract content from MCP response format
    if let Some(content_array) = result.get("content").and_then(|c| c.as_array()) {
        if let Some(first_item) = content_array.first() {
            if let Some(text_content) = first_item.get("text").and_then(|t| t.as_str()) {
                // Parse the JSON response from hash tool
                let mut hash_data_bytes = text_content.as_bytes().to_vec();
                if let Ok(hash_data) = simd_json::to_owned_value(&mut hash_data_bytes) {
                    let hash_value = hash_data.get("hash")
                        .and_then(|h| h.as_str())
                        .unwrap_or("unknown").to_string();
                    
                    return Ok(HashResult {
                        algorithm: hash_data.get("algorithm")
                            .and_then(|a| a.as_str())
                            .unwrap_or("unknown").to_string(),
                        input_data: hash_data.get("input")
                            .and_then(|i| i.as_str())
                            .unwrap_or("unknown").to_string(),
                        hash_value: hash_value.clone(),
                        hash_length: hash_value.len() as i32,
                    });
                }
            }
        }
    }

    Err("Invalid hash tool response format".to_string())
}

/// Create the GraphQL schema
pub fn create_schema() -> Schema<Query, Mutation, async_graphql::EmptySubscription> {
    Schema::build(Query, Mutation, async_graphql::EmptySubscription).finish()
}

use value_trait::prelude::*;