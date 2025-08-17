//! Transport layer integration for HTTP/2 and HTTP/3 using ONLY AsyncStream patterns
//!
//! Zero-allocation transport handling with Quiche (H3) and hyper (H2) integration.
//! Uses canonical Connection from protocols/connection.rs - no duplicate Connection types.

use std::collections::HashMap;
use std::net::SocketAddr;

use fluent_ai_async::prelude::*;
use quiche::{Config, Connection as QuicheConnection};

use super::connection::{Connection, ConnectionManager};
use super::h2::{H2Connection, H2Stream};
use super::h3::{H3Connection, H3Stream};
use crate::streaming::chunks::{FrameChunk, H2Frame, H3Frame};

/// Transport type for connection negotiation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransportType {
    H2,
    H3,
    Auto, // Try H3 first, fallback to H2
}

/// Transport layer manager that uses canonical Connection from protocols/connection.rs
///
/// This struct wraps the canonical Connection and provides transport-specific functionality
/// without duplicating the Connection type itself.
#[derive(Debug)]
pub struct TransportManager {
    connection_manager: ConnectionManager,
    transport_configs: HashMap<String, TransportConfig>,
    default_transport: TransportType,
}

/// Configuration for transport connections
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub transport_type: TransportType,
    pub timeout_ms: u64,
    pub max_streams: u32,
    pub enable_push: bool,
    pub quiche_config: Option<Config>,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            transport_type: TransportType::Auto,
            timeout_ms: 30000,
            max_streams: 100,
            enable_push: true,
            quiche_config: None,
        }
    }
}

impl TransportManager {
    /// Create new transport manager
    #[inline]
    pub fn new(default_transport: TransportType) -> Self {
        Self {
            connection_manager: ConnectionManager::new(),
            transport_configs: HashMap::new(),
            default_transport,
        }
    }

    /// Set transport configuration
    #[inline]
    pub fn set_config(&mut self, connection_id: String, config: TransportConfig) {
        self.transport_configs.insert(connection_id, config);
    }

    /// Create connection with transport negotiation
    pub fn create_connection(
        &mut self,
        remote_addr: SocketAddr,
        config: Option<TransportConfig>,
    ) -> AsyncStream<Connection, 1024> {
        let config = config.unwrap_or_default();
        let transport_type = config.transport_type;

        match transport_type {
            TransportType::H2 => self.connection_manager.create_h2_connection(true),
            TransportType::H3 => self.connection_manager.create_h3_connection(true),
            TransportType::Auto => {
                // Try H3 first, fallback to H2
                self.negotiate_connection(remote_addr, config)
            }
        }
    }

    /// Negotiate connection type (H3 first, H2 fallback)
    fn negotiate_connection(
        &mut self,
        remote_addr: SocketAddr,
        config: TransportConfig,
    ) -> AsyncStream<Connection, 1024> {
        // First attempt H3
        let h3_stream = self.connection_manager.create_h3_connection(true);

        AsyncStream::with_channel(move |sender| {
            // In a real implementation, we would:
            // 1. Try to establish H3 connection
            // 2. If it fails, fallback to H2
            // For now, we'll emit the H3 connection directly
            let connection = Connection::new_h3(true);
            emit!(sender, connection);
        })
    }

    /// Get connection by ID (delegates to canonical ConnectionManager)
    #[inline]
    pub fn get_connection(&self, id: &str) -> Option<&Connection> {
        self.connection_manager.get_connection(id)
    }

    /// Get mutable connection by ID
    #[inline]
    pub fn get_connection_mut(&mut self, id: &str) -> Option<&mut Connection> {
        self.connection_manager.get_connection_mut(id)
    }

    /// Remove connection
    #[inline]
    pub fn remove_connection(&mut self, id: &str) -> Option<Connection> {
        self.transport_configs.remove(id);
        self.connection_manager.remove_connection(id)
    }

    /// Get transport statistics
    pub fn transport_stats(&self) -> TransportStats {
        let conn_stats = self.connection_manager.stats();
        TransportStats {
            total_connections: conn_stats.total_connections,
            h2_connections: conn_stats.h2_connections,
            h3_connections: conn_stats.h3_connections,
            error_connections: conn_stats.error_connections,
            active_transports: self.transport_configs.len(),
        }
    }
}

/// Transport statistics
#[derive(Debug, Clone)]
pub struct TransportStats {
    pub total_connections: usize,
    pub h2_connections: usize,
    pub h3_connections: usize,
    pub error_connections: usize,
    pub active_transports: usize,
}

/// Transport connection wrapper for protocol-specific operations
///
/// This wraps the canonical Connection with transport-specific metadata
/// without duplicating the Connection type itself.
#[derive(Debug)]
pub struct TransportConnection {
    pub connection_id: String,
    pub transport_type: TransportType,
    pub connection: Connection,
    pub remote_addr: SocketAddr,
    pub config: TransportConfig,
}

impl MessageChunk for TransportConnection {
    fn bad_chunk(error: String) -> Self {
        TransportConnection {
            connection_id: "error".to_string(),
            transport_type: TransportType::Auto,
            connection: Connection::bad_chunk(error),
            remote_addr: "0.0.0.0:0".parse().unwrap(),
            config: TransportConfig::default(),
        }
    }

    fn is_error(&self) -> bool {
        self.connection.is_error()
    }

    fn error(&self) -> Option<&str> {
        self.connection.error()
    }
}

impl TransportConnection {
    /// Create new transport connection
    #[inline]
    pub fn new(
        connection_id: String,
        transport_type: TransportType,
        connection: Connection,
        remote_addr: SocketAddr,
        config: TransportConfig,
    ) -> Self {
        Self {
            connection_id,
            transport_type,
            connection,
            remote_addr,
            config,
        }
    }

    /// Send data through transport connection
    pub fn send_data(self, data: Vec<u8>) -> AsyncStream<FrameChunk, 1024> {
        self.connection.send_data(data)
    }

    /// Receive data from transport connection
    pub fn receive_data(self) -> AsyncStream<FrameChunk, 1024> {
        self.connection.receive_data()
    }

    /// Close transport connection
    pub fn close(self) -> AsyncStream<FrameChunk, 1024> {
        self.connection.close()
    }

    /// Check if connection supports server push
    #[inline]
    pub fn supports_push(&self) -> bool {
        match self.transport_type {
            TransportType::H2 | TransportType::H3 => self.config.enable_push,
            TransportType::Auto => true, // Will be determined during negotiation
        }
    }

    /// Get maximum concurrent streams
    #[inline]
    pub fn max_streams(&self) -> u32 {
        self.config.max_streams
    }

    /// Check if connection is H3
    #[inline]
    pub fn is_h3(&self) -> bool {
        matches!(self.transport_type, TransportType::H3) || self.connection.is_h3()
    }

    /// Check if connection is H2
    #[inline]
    pub fn is_h2(&self) -> bool {
        matches!(self.transport_type, TransportType::H2) || self.connection.is_h2()
    }
}

/// Transport layer utilities
pub mod utils {
    use super::*;

    /// Detect optimal transport type based on server capabilities
    pub fn detect_transport_type(server_capabilities: &[&str]) -> TransportType {
        if server_capabilities.contains(&"h3") {
            TransportType::H3
        } else if server_capabilities.contains(&"h2") {
            TransportType::H2
        } else {
            TransportType::Auto
        }
    }

    /// Create default Quiche config for H3 connections
    pub fn default_quiche_config() -> Config {
        let mut config = Config::new(quiche::PROTOCOL_VERSION).unwrap();
        config.set_application_protos(&[b"h3"]).unwrap();
        config.set_max_idle_timeout(30000);
        config.set_max_recv_udp_payload_size(1350);
        config.set_max_send_udp_payload_size(1350);
        config.set_initial_max_data(10_000_000);
        config.set_initial_max_stream_data_bidi_local(1_000_000);
        config.set_initial_max_stream_data_bidi_remote(1_000_000);
        config.set_initial_max_streams_bidi(100);
        config.set_initial_max_streams_uni(100);
        config.set_disable_active_migration(true);
        config
    }
}

/// Transport connection factory
pub struct TransportFactory;

impl TransportFactory {
    /// Create transport connection with automatic type detection
    pub fn create_auto_connection(
        connection_id: String,
        remote_addr: SocketAddr,
        server_capabilities: &[&str],
    ) -> AsyncStream<TransportConnection, 1024> {
        let transport_type = utils::detect_transport_type(server_capabilities);
        let config = TransportConfig {
            transport_type,
            ..Default::default()
        };

        AsyncStream::with_channel(move |sender| {
            let connection = match transport_type {
                TransportType::H2 => Connection::new_h2(true),
                TransportType::H3 => Connection::new_h3(true),
                TransportType::Auto => Connection::new_h3(true), // Default to H3
            };

            let transport_conn = TransportConnection::new(
                connection_id,
                transport_type,
                connection,
                remote_addr,
                config,
            );

            emit!(sender, transport_conn);
        })
    }

    /// Create H2-specific transport connection
    pub fn create_h2_connection(
        connection_id: String,
        remote_addr: SocketAddr,
    ) -> AsyncStream<TransportConnection, 1024> {
        AsyncStream::with_channel(move |sender| {
            let connection = Connection::new_h2(true);
            let config = TransportConfig {
                transport_type: TransportType::H2,
                ..Default::default()
            };

            let transport_conn = TransportConnection::new(
                connection_id,
                TransportType::H2,
                connection,
                remote_addr,
                config,
            );

            emit!(sender, transport_conn);
        })
    }

    /// Create H3-specific transport connection
    pub fn create_h3_connection(
        connection_id: String,
        remote_addr: SocketAddr,
    ) -> AsyncStream<TransportConnection, 1024> {
        AsyncStream::with_channel(move |sender| {
            let connection = Connection::new_h3(true);
            let mut config = TransportConfig {
                transport_type: TransportType::H3,
                ..Default::default()
            };
            config.quiche_config = Some(utils::default_quiche_config());

            let transport_conn = TransportConnection::new(
                connection_id,
                TransportType::H3,
                connection,
                remote_addr,
                config,
            );

            emit!(sender, transport_conn);
        })
    }
}
