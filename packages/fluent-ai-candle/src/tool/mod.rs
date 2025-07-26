//! Unified Tool Module
//!
//! This module consolidates all tool-related functionality including:
//! - Core Tool trait and implementations (from tool.rs)
//! - MCP (Model Context Protocol) client and transport (from mcp.rs)
//! - MCP tool traits and data structures (from mcp_tool_traits.rs)
//! - Tool execution and management utilities
//!
//! The module provides a clean, unified interface for all tool operations
//! while maintaining backward compatibility and eliminating version confusion.

pub mod core;
pub mod mcp;
pub mod traits;
pub mod types;

// Re-export core tool functionality
pub use core::{
    ExecToText, NamedTool, Perplexity, Tool, ToolDefinition, ToolEmbeddingDyn, ToolSet};

// Re-export MCP functionality
pub use mcp::{Client as McpClient, McpError, StdioTransport, Transport};
// Re-export MCP tool traits and data
pub use traits::{McpTool, Tool as ToolTrait};
pub use types::{McpToolData, Tool as McpToolType};
