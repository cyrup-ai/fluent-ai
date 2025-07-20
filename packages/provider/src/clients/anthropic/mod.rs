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
pub mod config;
pub mod discovery;
pub mod error;
pub mod expression_evaluator;
pub mod messages;
pub mod requests;
pub mod responses;
pub mod streaming;
pub mod tools;

pub use client::*;
pub use completion::*;
// Re-export the polymorphic extension traits
pub use completion::{AnthropicExtensions, SearchResultData};
pub use config::*;
pub use discovery::AnthropicDiscovery;
pub use error::*;
pub use messages::*;
pub use requests::*;
pub use responses::*;
pub use streaming::*;
pub use tools::*;
