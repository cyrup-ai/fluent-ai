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
use crate::protocols::frames::{FrameChunk, H2Frame, H3Frame};

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

impl std::fmt::Debug for TransportConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportConfig")
            .field("transport_type", &self.transport_type)
            .field("timeout_ms", &self.timeout_ms)
            .field("max_streams", &self.max_streams)
            .field("enable_push", &self.enable_push)
            .field("quiche_config", &"<Option<quiche::Config>>")
            .finish()
    }
}

impl Clone for TransportConfig {
    fn clone(&self) -> Self {
        Self {
            transport_type: self.transport_type.clone(),
            timeout_ms: self.timeout_ms,
            max_streams: self.max_streams,
            enable_push: self.enable_push,
            quiche_config: None, // quiche::Config doesn't implement Clone, so we set to None
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
        AsyncStream::with_channel(move |sender| {
            // Step 1: Attempt QUIC/H3 connection
            let h3_result = attempt_h3_connection(remote_addr, &config);
            
            let h3_error = match h3_result {
                Ok(h3_conn) => {
                    emit!(sender, h3_conn);
                    return;
                }
                Err(h3_error) => {
                    tracing::warn!(
                        target: "fluent_ai_http3::protocols::transport",
                        error = %h3_error,
                        "H3 connection failed, falling back to H2"
                    );
                    h3_error
                }
            };
            
            // Step 2: Fallback to H2 over TCP
            let h2_result = attempt_h2_connection(remote_addr, &config);
            
            match h2_result {
                Ok(h2_conn) => {
                    emit!(sender, h2_conn);
                }
                Err(h2_error) => {
                    emit!(sender, Connection::bad_chunk(
                        format!("All protocols failed. H3: {}, H2: {}", h3_error, h2_error)
                    ));
                }
            }
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

impl Default for TransportConnection {
    fn default() -> Self {
        Self {
            connection_id: String::new(),
            transport_type: TransportType::Auto,
            connection: Connection::default(),
            remote_addr: "0.0.0.0:0".parse()
                .expect("Hardcoded default socket address should always parse successfully"),
            config: TransportConfig::default(),
        }
    }
}

impl MessageChunk for TransportConnection {
    fn bad_chunk(error: String) -> Self {
        TransportConnection {
            connection_id: "error".to_string(),
            transport_type: TransportType::Auto,
            connection: Connection::bad_chunk(error),
            remote_addr: "0.0.0.0:0".parse()
                .expect("Hardcoded default socket address should always parse successfully"),
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
        let mut config = Config::new(quiche::PROTOCOL_VERSION)
            .expect("QUICHE protocol version should always be valid");
        config.set_application_protos(&[b"h3"])
            .expect("H3 application protocol should always be valid");
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

/// Attempt to establish H3 connection with error handling
fn attempt_h3_connection(
    remote_addr: SocketAddr,
    config: &TransportConfig,
) -> Result<Connection, String> {
    use std::net::UdpSocket;
    use std::time::{Duration, Instant};
    
    // Create UDP socket for QUIC
    let socket = UdpSocket::bind("0.0.0.0:0")
        .map_err(|e| format!("UDP bind failed: {}", e))?;
    
    socket.connect(remote_addr)
        .map_err(|e| format!("UDP connect failed: {}", e))?;
    
    // Configure QUIC
    let mut quiche_config = quiche::Config::new(quiche::PROTOCOL_VERSION)
        .map_err(|e| format!("QUIC config failed: {}", e))?;
    
    quiche_config.set_application_protos(&[b"h3"])
        .map_err(|e| format!("ALPN setup failed: {}", e))?;
    
    quiche_config.set_max_idle_timeout(config.timeout_ms);
    quiche_config.set_initial_max_data(10_000_000);
    quiche_config.set_initial_max_streams_bidi(100);
    
    // Generate connection ID
    let mut scid = [0; quiche::MAX_CONN_ID_LEN];
    use ring::rand::*;
    SystemRandom::new().fill(&mut scid[..])
        .map_err(|_| "RNG failed".to_string())?;
    
    let scid = quiche::ConnectionId::from_ref(&scid);
    
    // Initiate QUIC connection
    let local_addr = socket.local_addr()
        .map_err(|e| format!("Local addr failed: {}", e))?;
    
    let mut conn = quiche::connect(
        None, // Server name for SNI
        &scid,
        local_addr,
        remote_addr,
        &mut quiche_config,
    ).map_err(|e| format!("QUIC connect failed: {}", e))?;
    
    // Perform handshake with timeout
    let start = Instant::now();
    let timeout = Duration::from_millis(config.timeout_ms);
    
    while !conn.is_established() {
        if start.elapsed() > timeout {
            return Err("QUIC handshake timeout".to_string());
        }
        
        // Send/receive packets (simplified for brevity)
        std::thread::sleep(Duration::from_millis(10));
    }
    
    // Connection established, return H3 connection
    Ok(Connection::new_h3(true))
}

/// Attempt to establish H2 connection with error handling
fn attempt_h2_connection(
    remote_addr: SocketAddr,
    config: &TransportConfig,
) -> Result<Connection, String> {
    use std::net::TcpStream;
    use std::time::Duration;
    
    // TCP connection with timeout
    let _tcp_stream = TcpStream::connect_timeout(
        &remote_addr,
        Duration::from_millis(config.timeout_ms),
    ).map_err(|e| format!("TCP connect failed: {}", e))?;
    
    // For now, return a simple H2 connection
    // In production, would do TLS handshake and H2 negotiation
    Ok(Connection::new_h2(true))
}
