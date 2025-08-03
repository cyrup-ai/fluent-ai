//! HTTP response types and utilities
//!
//! Decomposed response module providing:
//! - Core HttpResponse struct and status handling
//! - Header management and caching utilities  
//! - Body processing including JSON streaming and SSE parsing

pub mod body;
pub mod core;
pub mod headers;

// Re-export all main types for backward compatibility
pub use core::HttpResponse;

pub use body::{JsonStream, SseEvent};
