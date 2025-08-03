//! Blazing-fast ergonomic Http3 builder API with zero allocation and elegant fluent interface
//! Supports streaming HttpChunk/BadHttpChunk responses, Serde integration, and shorthand methods
//!
//! This module has been decomposed into focused submodules for better maintainability:
//! - `core`: Core Http3Builder struct and types
//! - `methods`: HTTP method implementations (GET, POST, PUT, etc.)
//! - `headers`: Header management and manipulation
//! - `auth`: Authentication methods (Bearer, Basic, API key)
//! - `body`: Request body handling (JSON, form, raw)
//! - `execution`: Request execution and response handling
//! - `streaming`: JSONPath streaming functionality
//! - `fluent`: Fluent API extensions and download functionality

// Core builder structures and types
pub mod core;

// HTTP method implementations
pub mod methods;

// Header management functionality
pub mod headers;

// Authentication methods
pub mod auth;

// Request body handling
pub mod body;

// Request execution and response handling
pub mod execution;

// JSONPath streaming functionality
pub mod streaming;

// Fluent API extensions and download functionality
pub mod fluent;

// Re-export all main types for backward compatibility
pub use core::{BodyNotSet, BodySet, ContentType, Http3Builder, JsonPathStreaming};

// Legacy alias for backward compatibility
pub use Http3Builder as Builder;
pub use execution::HttpStreamExt;
pub use fluent::{DownloadBuilder, DownloadProgress};
pub use headers::header;
pub use streaming::JsonPathStream;
