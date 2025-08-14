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

// Re-export core Candle tool functionality (avoid conflicts with trait)
pub use core::{CandleExecToText, CandleNamedTool, CandlePerplexity};

// Re-export Candle MCP functionality
pub use mcp::{
    CandleClient as CandleMcpClient, CandleMcpError, CandleStdioTransport, CandleTransport,
};
// Re-export trait-backed architecture types (NEW PREFERRED APPROACH)
pub use traits::{CandleMcpTool, CandleTool};
// Re-export legacy MCP types for backward compatibility
pub use types::CandleMcpToolData;
