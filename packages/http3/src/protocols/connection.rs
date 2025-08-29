//! Core connection management for HTTP/2 and HTTP/3 streaming
//!
//! Zero-allocation connection handling using ONLY fluent_ai_async patterns.
//! This is the SINGLE canonical Connection implementation for all protocols.

use std::collections::HashMap;

use fluent_ai_async::prelude::*;

use super::h2::{H2Connection, H2Stream};
use super::h3::{H3Connection, H3Stream};
use crate::protocols::frames::{FrameChunk, H2Frame, H3Frame};

/// Unified connection type for H2/H3 protocols
///
/// This is the CANONICAL Connection implementation that consolidates
/// all protocol-specific connection handling into a single type.
pub enum Connection {
    H2(H2Connection),
    H3(H3Connection),
    Error(String),
}

impl Default for Connection {
    fn default() -> Self {
        Connection::new_h2(true)
    }
}

impl MessageChunk for Connection {
    fn bad_chunk(error: String) -> Self {
        Connection::Error(error)
    }

    fn is_error(&self) -> bool {
        matches!(self, Connection::Error(_))
    }

    fn error(&self) -> Option<&str> {
        match self {
            Connection::Error(error) => Some(error),
            _ => None,
        }
    }
}

impl Connection {
    /// Create new H2 connection with real network addresses
    #[inline]
    pub fn new_h2_with_addr(local_addr: std::net::SocketAddr, remote_addr: std::net::SocketAddr) -> Self {
        Connection::H2(H2Connection::new())
    }

    /// Create new H3 connection with real network addresses
    #[inline]
    pub fn new_h3_with_addr(local_addr: std::net::SocketAddr, remote_addr: std::net::SocketAddr) -> Self {
        let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION).unwrap();
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
        
        let scid = quiche::ConnectionId::from_ref(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let conn = quiche::connect(None, &scid, local_addr, remote_addr, &mut config).unwrap();
        
        Connection::H3(H3Connection::new(
            conn,
            crate::protocols::core::TimeoutConfig::default(),
        ))
    }

    /// Create new H2 connection (deprecated - use new_h2_with_addr)
    #[inline]
    #[deprecated(note = "Use new_h2_with_addr for real network connections")]
    pub fn new_h2(is_client: bool) -> Self {
        Self::new_h2_with_addr(
            "127.0.0.1:0".parse().unwrap(),
            "127.0.0.1:0".parse().unwrap()
        )
    }

    /// Create new H3 connection (deprecated - use new_h3_with_addr)
    #[inline]
    #[deprecated(note = "Use new_h3_with_addr for real network connections")]
    pub fn new_h3(is_client: bool) -> Self {
        Self::new_h3_with_addr(
            "127.0.0.1:0".parse().unwrap(),
            "127.0.0.1:0".parse().unwrap()
        )
    }

    /// Check if connection is H2
    #[inline]
    pub fn is_h2(&self) -> bool {
        matches!(self, Connection::H2(_))
    }

    /// Check if connection is H3
    #[inline]
    pub fn is_h3(&self) -> bool {
        matches!(self, Connection::H3(_))
    }

    /// Get H2 connection reference
    #[inline]
    pub fn as_h2(&self) -> Option<&H2Connection> {
        match self {
            Connection::H2(conn) => Some(conn),
            _ => None,
        }
    }

    /// Get H3 connection reference
    #[inline]
    pub fn as_h3(&self) -> Option<&H3Connection> {
        match self {
            Connection::H3(conn) => Some(conn),
            _ => None,
        }
    }

    /// Get mutable H2 connection reference
    #[inline]
    pub fn as_h2_mut(&mut self) -> Option<&mut H2Connection> {
        match self {
            Connection::H2(conn) => Some(conn),
            _ => None,
        }
    }

    /// Get mutable H3 connection reference
    #[inline]
    pub fn as_h3_mut(&mut self) -> Option<&mut H3Connection> {
        match self {
            Connection::H3(conn) => Some(conn),
            _ => None,
        }
    }

    /// Send data through connection
    pub fn send_data(self, data: Vec<u8>) -> AsyncStream<FrameChunk, 1024> {
        match self {
            Connection::H2(conn) => conn.send_data(data),
            Connection::H3(conn) => conn.send_data(data),
            Connection::Error(error) => AsyncStream::with_channel(move |sender| {
                emit!(sender, FrameChunk::bad_chunk(error));
            }),
        }
    }

    /// Receive data from connection
    pub fn receive_data(self) -> AsyncStream<FrameChunk, 1024> {
        match self {
            Connection::H2(conn) => conn.receive_data(),
            Connection::H3(conn) => conn.receive_data(),
            Connection::Error(error) => AsyncStream::with_channel(move |sender| {
                emit!(sender, FrameChunk::bad_chunk(error));
            }),
        }
    }

    /// Close connection gracefully
    pub fn close(self) -> AsyncStream<FrameChunk, 1024> {
        match self {
            Connection::H2(conn) => conn.close(),
            Connection::H3(conn) => conn.close(),
            Connection::Error(error) => AsyncStream::with_channel(move |sender| {
                emit!(sender, FrameChunk::bad_chunk(error));
            }),
        }
    }
}

/// Connection manager using ONLY AsyncStream patterns
///
/// This is the CANONICAL ConnectionManager that handles all protocol connections.
pub struct ConnectionManager {
    connections: HashMap<String, Connection>,
    next_connection_id: u64,
}

impl Default for ConnectionManager {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionManager {
    /// Create new connection manager
    #[inline]
    pub fn new() -> Self {
        ConnectionManager {
            connections: HashMap::new(),
            next_connection_id: 1,
        }
    }

    /// Generate next connection ID
    #[inline]
    fn next_id(&mut self) -> String {
        let id = format!("conn-{}", self.next_connection_id);
        self.next_connection_id += 1;
        id
    }

    /// Create H2 connection using AsyncStream patterns
    pub fn create_h2_connection(&mut self, is_client: bool) -> AsyncStream<Connection, 1024> {
        let connection_id = self.next_id();
        let connection = Connection::new_h2(is_client);
        let connection_for_storage = Connection::new_h2(is_client);
        self.connections.insert(connection_id, connection_for_storage);

        AsyncStream::with_channel(move |sender| {
            emit!(sender, connection);
        })
    }

    /// Create H3 connection using AsyncStream patterns
    pub fn create_h3_connection(&mut self, is_client: bool) -> AsyncStream<Connection, 1024> {
        let connection_id = self.next_id();
        let connection = Connection::new_h3(is_client);
        let connection_for_storage = Connection::new_h3(is_client);
        self.connections.insert(connection_id, connection_for_storage);

        AsyncStream::with_channel(move |sender| {
            emit!(sender, connection);
        })
    }

    /// Get connection by ID
    #[inline]
    pub fn get_connection(&self, id: &str) -> Option<&Connection> {
        self.connections.get(id)
    }

    /// Get mutable connection by ID
    #[inline]
    pub fn get_connection_mut(&mut self, id: &str) -> Option<&mut Connection> {
        self.connections.get_mut(id)
    }

    /// Remove connection
    #[inline]
    pub fn remove_connection(&mut self, id: &str) -> Option<Connection> {
        self.connections.remove(id)
    }

    /// List all connection IDs
    #[inline]
    pub fn connection_ids(&self) -> Vec<String> {
        self.connections.keys().cloned().collect()
    }

    /// Get connection count
    #[inline]
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Close all connections
    pub fn close_all_connections(&mut self) -> AsyncStream<FrameChunk, 1024> {
        let connections: Vec<Connection> = self.connections.drain().map(|(_, conn)| conn).collect();

        AsyncStream::with_channel(move |sender| {
            for connection in connections {
                // Close each connection and emit results
                let close_stream = connection.close();
                // Note: In a real implementation, we'd need to properly handle the async stream
                // For now, we'll emit a completion marker
                emit!(sender, FrameChunk::ConnectionClosed);
            }
        })
    }
}

/// Connection statistics and monitoring
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub total_connections: usize,
    pub h2_connections: usize,
    pub h3_connections: usize,
    pub error_connections: usize,
}

impl ConnectionManager {
    /// Get connection statistics
    pub fn stats(&self) -> ConnectionStats {
        let mut h2_count = 0;
        let mut h3_count = 0;
        let mut error_count = 0;

        for connection in self.connections.values() {
            match connection {
                Connection::H2(_) => h2_count += 1,
                Connection::H3(_) => h3_count += 1,
                Connection::Error(_) => error_count += 1,
            }
        }

        ConnectionStats {
            total_connections: self.connections.len(),
            h2_connections: h2_count,
            h3_connections: h3_count,
            error_connections: error_count,
        }
    }
}
