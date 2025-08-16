//! HTTP Request types module
//!
//! This module provides the core Request and RequestBuilder types for HTTP requests,
//! along with their associated traits and conversions.

pub mod builder;
pub mod conversions;
pub mod debug;
pub mod request;

// Re-export the main types for easy access
pub use builder::RequestBuilder;
pub use request::Request;
