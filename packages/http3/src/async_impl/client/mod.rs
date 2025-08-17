//! Client Module
//!
//! This module provides client infrastructure for HTTP/3
//! streaming with fluent_ai_async architecture.

pub mod connection;
pub mod quiche_client;
pub mod quiche_connection;

// Re-export connection management types
pub use connection::{
    ConnectionConfig, ConnectionManager, ConnectionPoolStats, ConnectionState,
    DefaultConnectionManager, HttpConnection, HttpConnectionPool,
};
// Re-export Quiche client types
pub use quiche_client::{
    QuicheHttp3Client, process_readable_streams, read_stream_data, receive_packets,
    write_stream_data,
};
// Re-export Quiche connection types
pub use quiche_connection::QuicheConnection;
