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

// Re-export core Candle tool functionality
pub use core::{
    CandleExecToText, CandleNamedTool, CandlePerplexity, CandleTool, CandleToolDefinition, CandleToolEmbeddingDyn, CandleToolSet};

// Re-export Candle MCP functionality
pub use mcp::{CandleClient as CandleMcpClient, CandleMcpError, CandleStdioTransport, CandleTransport};
// Re-export Candle MCP tool traits and data
pub use traits::{CandleMcpTool, CandleTool as CandleToolTrait};
pub use types::{CandleMcpToolData, CandleTool as CandleMcpToolType};
