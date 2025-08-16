//! H3 client connection module
//!
//! Modular HTTP/3 connection establishment with QUIC transport,
//! TLS configuration, and streaming connection patterns.

pub mod configuration;
pub mod connection_logic;
pub mod connector_core;
pub mod h3_establishment;
pub mod types;
pub mod utilities;

// Re-export main types and traits
pub use configuration::H3ClientConfig;
pub use connector_core::H3Connector;
pub use types::{BoxError, DynResolver, H3Connection, PoolClient};
pub use utilities::{
    calculate_retry_delay, is_retryable_error, is_valid_address, validate_server_name,
};
