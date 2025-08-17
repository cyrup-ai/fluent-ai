//! Async Implementation Modules
//!
//! This module provides async implementation infrastructure for HTTP/3
//! streaming with fluent_ai_async architecture.

pub mod client;
pub mod connection;
pub mod request;
pub mod response;
pub mod resolver;
// pub mod utilities; // Temporarily commented out until utilities module is created

// Re-export core streaming types
pub use client::connection::{
    ConnectionConfig, ConnectionManager, ConnectionPoolStats, ConnectionState,
    DefaultConnectionManager, HttpConnection, HttpConnectionPool,
};
pub use response::core::streaming::{HttpStreamingResponse, HttpStreamingResponseBuilder};
