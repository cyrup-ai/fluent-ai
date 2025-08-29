//! HTTP/1.1 protocol implementation
//!
//! This module provides basic HTTP/1.1 functionality for fallback scenarios
//! when HTTP/2 and HTTP/3 are not available.

pub mod adapter;

pub use adapter::execute_http1_request;