//! Production-ready Anthropic Claude API provider implementation
//!
//! This module provides a complete, battle-tested implementation for Anthropic's Claude API
//! with zero-allocation patterns, robust error handling, and comprehensive feature support.
//!
//! ## Features
//! - Full Claude 4/3.5/3 model support with proper max_tokens handling
//! - Complete message/content type system with tool use
//! - Document attachment support (PDFs, images)
//! - Tool calling with ToolDefinition/ToolChoice
//! - Proper error handling with ApiResponse envelope
//! - JSON payload building with zero-alloc merge_inplace
//! - Streaming support with SSE decoding
//! - Usage token tracking (input/output tokens)

pub mod client;
pub mod completion;
pub mod messages;
pub mod streaming;
pub mod tools;
pub mod error;

pub use client::*;
pub use completion::*;
pub use messages::*;
pub use streaming::*;
pub use tools::*;
pub use error::*;