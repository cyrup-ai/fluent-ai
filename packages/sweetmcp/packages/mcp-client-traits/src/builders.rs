//! Fluent builder system for constructing MCP tool requests
//!
//! This module provides a fluent, immutable builder API that works consistently
//! across all MCP client implementations, using sweet-mcp-type JsonValue exclusively.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use sweet_mcp_type::{JsonValue, Response};
use crate::traits::McpClient;
use crate::errors::ClientError;

/// Fluent builder trait for constructing tool requests
///
/// This trait enables the fluent API pattern:
/// ```rust,no_run
/// # async fn example(client: impl mcp_client_traits::RequestBuilder<impl mcp_client_traits::McpClient>) -> Result<(), mcp_client_traits::ClientError> {
/// let response = client
///     .tool("hash")
///     .with_arg("data", "Hello World")
///     .with_arg("algorithm", "sha256")
///     .execute()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub trait RequestBuilder<T> {
    /// Start building a tool request
    ///
    /// # Arguments
    /// * `name` - The tool name to execute
    ///
    /// # Returns
    /// A `ToolRequestBuilder` for fluent method chaining
    fn tool(self, name: &str) -> ToolRequestBuilder<T>;
}

/// Tool-specific request builder for fluent API construction
///
/// This struct provides the implementation of the fluent builder pattern,
/// accumulating arguments until `execute()` is called.
pub struct ToolRequestBuilder<T> {
    /// The tool name to execute
    tool_name: String,
    /// Accumulated arguments using sweet-mcp-type JsonValue
    arguments: HashMap<String, JsonValue>,
    /// The client instance to execute with
    client: T,
    /// Optional request metadata
    metadata: Option<HashMap<String, JsonValue>>,
}

impl<T> ToolRequestBuilder<T> {
    /// Create a new tool request builder
    ///
    /// # Arguments
    /// * `client` - The client instance to use for execution
    /// * `tool_name` - The name of the tool to execute
    pub fn new(client: T, tool_name: String) -> Self {
        Self {
            tool_name,
            arguments: HashMap::new(),
            client,
            metadata: None,
        }
    }

    /// Add an argument to the tool request
    ///
    /// # Arguments
    /// * `key` - The argument name
    /// * `value` - The argument value (must convert to JsonValue)
    ///
    /// # Returns
    /// Updated builder for method chaining
    ///
    /// # Example
    /// ```rust,no_run
    /// # fn example(builder: mcp_client_traits::ToolRequestBuilder<impl mcp_client_traits::McpClient>) {
    /// let builder = builder
    ///     .with_arg("data", "Hello World")
    ///     .with_arg("algorithm", "sha256")
    ///     .with_arg("iterations", 1000_i64);
    /// # }
    /// ```
    pub fn with_arg<V>(mut self, key: &str, value: V) -> Self
    where
        V: Into<JsonValue>,
    {
        self.arguments.insert(key.to_string(), value.into());
        self
    }

    /// Add multiple arguments from an iterator
    ///
    /// # Arguments
    /// * `args` - Iterator of (key, value) pairs
    ///
    /// # Returns
    /// Updated builder for method chaining
    pub fn with_args<I, K, V>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<JsonValue>,
    {
        for (key, value) in args {
            self.arguments.insert(key.into(), value.into());
        }
        self
    }

    /// Add metadata to the request
    ///
    /// # Arguments
    /// * `key` - The metadata key
    /// * `value` - The metadata value
    ///
    /// # Returns
    /// Updated builder for method chaining
    pub fn with_metadata<V>(mut self, key: &str, value: V) -> Self
    where
        V: Into<JsonValue>,
    {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        if let Some(ref mut metadata) = self.metadata {
            metadata.insert(key.to_string(), value.into());
        }
        self
    }

    /// Validate arguments before execution
    ///
    /// # Returns
    /// Result indicating if arguments are valid
    pub fn validate(&self) -> Result<(), ClientError> {
        if self.tool_name.is_empty() {
            return Err(ClientError::invalid_argument(
                "tool_name",
                "Tool name cannot be empty",
                None,
            ));
        }

        // Validate common tool arguments
        match self.tool_name.as_str() {
            "time" => {
                if !self.arguments.contains_key("name") {
                    return Err(ClientError::invalid_argument(
                        "name",
                        "Time tool requires 'name' argument",
                        Some(vec!["get_time_utc".to_string(), "parse_time".to_string()]),
                    ));
                }
            }
            "hash" => {
                if !self.arguments.contains_key("data") {
                    return Err(ClientError::invalid_argument(
                        "data",
                        "Hash tool requires 'data' argument",
                        None,
                    ));
                }
                if !self.arguments.contains_key("algorithm") {
                    return Err(ClientError::invalid_argument(
                        "algorithm",
                        "Hash tool requires 'algorithm' argument",
                        Some(vec![
                            "sha256".to_string(),
                            "sha512".to_string(),
                            "md5".to_string(),
                            "base64".to_string(),
                        ]),
                    ));
                }
            }
            _ => {
                // For unknown tools, just ensure we have some arguments
                // The server will validate tool-specific requirements
            }
        }

        Ok(())
    }

    /// Get the number of arguments currently set
    ///
    /// # Returns
    /// The count of arguments in the builder
    pub fn arg_count(&self) -> usize {
        self.arguments.len()
    }

    /// Check if a specific argument is set
    ///
    /// # Arguments
    /// * `key` - The argument name to check
    ///
    /// # Returns
    /// True if the argument is present
    pub fn has_arg(&self, key: &str) -> bool {
        self.arguments.contains_key(key)
    }

    /// Get the tool name being built
    ///
    /// # Returns
    /// Reference to the tool name
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

impl<T: McpClient> ToolRequestBuilder<T> {
    /// Build and execute the tool request
    ///
    /// This method validates the arguments, constructs the final request,
    /// and executes it using the client's `call_tool` method.
    ///
    /// # Returns
    /// Future resolving to the tool execution response
    ///
    /// # Errors
    /// Returns `ClientError` if:
    /// - Argument validation fails
    /// - Tool execution fails
    /// - Network/transport errors occur
    ///
    /// # Example
    /// ```rust,no_run
    /// # async fn example(client: impl mcp_client_traits::McpClient + mcp_client_traits::RequestBuilder<impl mcp_client_traits::McpClient>) -> Result<(), mcp_client_traits::ClientError> {
    /// let response = client
    ///     .tool("hash")
    ///     .with_arg("data", "Hello World")
    ///     .with_arg("algorithm", "sha256")
    ///     .execute()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn execute(self) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send>>
    where
        T: Send + 'static,
    {
        Box::pin(async move {
            // Validate arguments before execution
            self.validate()?;

            // Convert arguments to JsonValue::Object for call_tool
            let args_json = JsonValue::from(self.arguments);

            // Execute the tool call
            self.client.call_tool(&self.tool_name, args_json).await
        })
    }

    /// Execute with timeout
    ///
    /// # Arguments
    /// * `timeout_ms` - Timeout in milliseconds
    ///
    /// # Returns
    /// Future with timeout applied
    pub fn execute_with_timeout(
        self,
        timeout_ms: u64,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ClientError>> + Send>>
    where
        T: Send + 'static,
    {
        let tool_name = self.tool_name.clone();
        
        Box::pin(async move {
            let timeout_duration = std::time::Duration::from_millis(timeout_ms);
            
            match tokio::time::timeout(timeout_duration, self.execute()).await {
                Ok(result) => result,
                Err(_) => Err(ClientError::timeout(
                    format!("tool execution for '{}'", tool_name),
                    timeout_ms,
                )),
            }
        })
    }
}

// Implement RequestBuilder for any type that implements McpClient
impl<T: McpClient> RequestBuilder<T> for T {
    fn tool(self, name: &str) -> ToolRequestBuilder<T> {
        ToolRequestBuilder::new(self, name.to_string())
    }
}

/// Convenience functions for common tool operations
impl<T: McpClient> ToolRequestBuilder<T> {
    /// Create a time tool request for getting current UTC time
    ///
    /// # Returns
    /// Configured builder for get_time_utc operation
    pub fn get_time_utc(client: T) -> Self {
        ToolRequestBuilder::new(client, "time".to_string())
            .with_arg("name", "get_time_utc")
    }

    /// Create a time tool request for parsing a time string
    ///
    /// # Arguments
    /// * `time_string` - The time string to parse
    ///
    /// # Returns
    /// Configured builder for parse_time operation
    pub fn parse_time(client: T, time_string: &str) -> Self {
        ToolRequestBuilder::new(client, "time".to_string())
            .with_arg("name", "parse_time")
            .with_arg("time_string", time_string)
    }

    /// Create a hash tool request for the specified algorithm
    ///
    /// # Arguments
    /// * `data` - The data to hash
    /// * `algorithm` - The hash algorithm
    ///
    /// # Returns
    /// Configured builder for hash operation
    pub fn hash(client: T, data: &str, algorithm: &str) -> Self {
        ToolRequestBuilder::new(client, "hash".to_string())
            .with_arg("data", data)
            .with_arg("algorithm", algorithm)
    }
}