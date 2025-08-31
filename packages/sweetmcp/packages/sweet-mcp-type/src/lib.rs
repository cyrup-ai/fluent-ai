//! sweet_mcp_type: Zero-Serde, high-perf JSON/TOML for MCP 2025-03-26

pub mod mcp;

// Re-export commonly used types for easier access
pub use mcp::{
    Request, Response, Notification, Message,
    RequestId, ProgressToken, LogLevel, Role,
    JsonRpcError, McpError,
    Implementation, ServerCapabilities,
    Resource, Prompt, PromptArgument, PromptMessage, PromptContent,
    TextContent, ImageContent, EmbeddedResource, ToolContent, ToolInfo,
    PromptReference, ResourceReference, ResourceContent, SamplingMessage,
    LoggingCapability, PromptsCapability, ResourcesCapability, 
    ToolsCapability, CompletionsCapability,
};

// Re-export JsonValue from simd-json for client usage
pub use simd_json::value::owned::Value as JsonValue;
