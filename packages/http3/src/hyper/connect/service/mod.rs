//! HTTP/3 connector service with zero-allocation connection establishment
//!
//! Decomposed connector service providing TCP, TLS, and proxy connections
//! with elite polling and streaming architecture.

pub mod core;
pub mod direct;
pub mod interface;
pub mod proxy;
pub mod tls;

// Re-export main service
pub use core::ConnectorService;

// Re-export TLS connection types
#[cfg(feature = "default-tls")]
pub use tls::NativeTlsConnection;
#[cfg(feature = "__rustls")]
pub use tls::RustlsConnection;
