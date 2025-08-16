//! H3 client module
//!
//! Module organization file for HTTP/3 client submodules and re-exports.

pub mod connection_management;
pub mod public_api;
pub mod request_execution;
pub mod request_execution_cookies;
pub mod request_handler;
pub mod stream_execution;
pub mod stream_execution_internal;
pub mod types;

// Re-export main types and traits
pub use public_api::*;
pub use request_handler::*;
pub use types::H3Client;
