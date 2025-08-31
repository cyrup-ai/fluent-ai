//! MCP Client Traits Library
//!
//! This library provides common traits and types for MCP (Model Context Protocol) 
//! client implementations. All traits are designed to work with `sweet-mcp-type`
//! structures for high-performance, zero-allocation JSON processing.
//!
//! # Overview
//!
//! The library defines several key traits:
//! - [`McpClient`] - Core client interface for MCP operations
//! - [`McpToolOperations`] - Convenience methods for common tools (time, hash)
//! - [`ProtocolClient`] - Protocol-specific client implementation interface
//! - [`RequestBuilder`] - Fluent API for building tool requests
//!
//! # Example
//!
//! ```rust,no_run
//! use mcp_client_traits::{McpClient, RequestBuilder};
//! 
//! async fn example<C: McpClient + RequestBuilder<C>>(client: C) -> Result<(), Box<dyn std::error::Error>> {
//!     // Use fluent API to call hash tool
//!     let response = client
//!         .tool("hash")
//!         .with_arg("data", "Hello World")
//!         .with_arg("algorithm", "sha256")
//!         .execute()
//!         .await?;
//!     
//!     println!("Hash result: {:?}", response);
//!     Ok(())
//! }
//! ```

pub mod traits;
pub mod builders;
pub mod errors;
pub mod response;

// Re-export main types for convenience
pub use traits::{McpClient, McpToolOperations, ProtocolClient, ClientCapabilities};
pub use builders::{RequestBuilder, ToolRequestBuilder};
pub use errors::ClientError;
pub use response::{ResponseAdapter, ContentExtractor};

// Re-export sweet-mcp-type for client implementations
pub use sweet_mcp_type::{
    Request, Response, Notification, Message,
    RequestId, JsonValue, JsonRpcError, McpError,
    ToolInfo, ToolContent, Implementation
};