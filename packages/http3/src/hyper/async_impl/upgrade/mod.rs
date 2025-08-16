//! HTTP upgrade support for bidirectional protocols
//!
//! This module provides comprehensive support for HTTP protocol upgrades
//! including WebSocket, HTTP/2 server push, and custom protocols using
//! zero-allocation streaming patterns with fluent_ai_async.

pub mod connection;
pub mod hyper_integration;
pub mod io_processing;
pub mod management;
pub mod response_extensions;
pub mod types;
pub mod write_operations;

// Re-export main types for convenience
pub use connection::Upgraded;
pub use hyper_integration::HyperCompatibleUpgraded;
pub use io_processing::create_read_stream;
// Re-export key functions
pub use management::{close_connection, get_connection_stats, is_connection_active};
pub use response_extensions::{detect_upgrade_protocol, extract_upgrade_info, is_upgrade_response};
pub use types::{ConnectionState, ConnectionStats, UpgradeProtocol};
pub use write_operations::write_data;
