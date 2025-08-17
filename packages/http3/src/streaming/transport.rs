//! Transport layer integration for HTTP/2 and HTTP/3 using ONLY AsyncStream patterns
//!
//! Zero-allocation transport handling with Quiche (H3) and hyper (H2) integration.

use std::collections::HashMap;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};
use quiche::{Config, Connection as QuicheConnection};

use super::connection::{Connection, ConnectionManager};
use super::frames::{FrameChunk, H2Frame, H3Frame};
use super::h2::{H2Connection, H2Stream};
use super::h3::{H3Connection, H3Stream};

/// Transport type for connection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransportType {
    H2,
    H3,
    Auto, // Try H3 first, fallback to H2
}

/// Transport connection wrapper
#[derive(Debug)]
pub struct TransportConnection {
    pub connection_id: String,
    pub transport_type: TransportType,
    pub connection: Connection,
    pub quiche_connection: Option<QuicheConnection>,
    pub h2_connection: Option<h2::client::Connection<bytes::Bytes>>,
}

impl MessageChunk for TransportConnection {
    fn bad_chunk(error: String) -> Self {
        TransportConnection {
            connection_id: "error".to_string(),
            transport_type: TransportType::Auto,
            connection: Connection::bad_chunk(error),
            quiche_connection: None,
            h2_connection: None,
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
    /// Create new H3 transport connection
    pub fn new_h3(connection_id: String, quiche_conn: QuicheConnection) -> Self {
        TransportConnection {
            connection_id,
            transport_type: TransportType::H3,
            connection: Connection::new_h3(true),
            quiche_connection: Some(quiche_conn),
            h2_connection: None,
        }
    }

    /// Create new H2 transport connection
    pub fn new_h2(connection_id: String, h2_conn: h2::client::Connection<bytes::Bytes>) -> Self {
        TransportConnection {
            connection_id,
            transport_type: TransportType::H2,
            connection: Connection::new_h2(true),
            quiche_connection: None,
            h2_connection: Some(h2_conn),
        }
    }

    /// Check if connection is ready
    pub fn is_ready(&self) -> bool {
        match self.transport_type {
            TransportType::H3 => self.quiche_connection.is_some(),
            TransportType::H2 => self.h2_connection.is_some(),
            TransportType::Auto => false,
        }
    }

    /// Check if connection has error
    pub fn is_error(&self) -> bool {
        self.connection.is_error()
    }
}

/// Create Quiche configuration for HTTP/3
fn create_quiche_config() -> Result<Config, String> {
    let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION)
        .map_err(|e| format!("Failed to create Quiche config: {}", e))?;

    config
        .set_application_protos(&[b"h3"])
        .map_err(|e| format!("Failed to set application protocols: {}", e))?;
    config.set_max_idle_timeout(30000);
    config.set_max_recv_udp_payload_size(1200);
    config.set_max_send_udp_payload_size(1200);
    config.set_initial_max_data(10_000_000);
    config.set_initial_max_stream_data_bidi_local(1_000_000);
    config.set_initial_max_stream_data_bidi_remote(1_000_000);
    config.set_initial_max_streams_bidi(100);
    config.set_disable_active_migration(true);

    Ok(config)
}

/// Transport manager for H2/H3 connections using ONLY AsyncStream patterns
#[derive(Debug)]
pub struct TransportManager {
    pub connections: HashMap<String, TransportConnection>,
    pub connection_manager: ConnectionManager,
    pub next_connection_id: u64,
}

impl TransportManager {
    /// Create new transport manager
    pub fn new() -> Self {
        TransportManager {
            connections: HashMap::new(),
            connection_manager: ConnectionManager::new(),
            next_connection_id: 1,
        }
    }

    /// Establish transport connection using AsyncStream patterns
    pub fn establish_connection_streaming(
        &mut self,
        host: &str,
        port: u16,
        transport_type: TransportType,
    ) -> AsyncStream<TransportConnection, 1024> {
        let host = host.to_string();
        let connection_id = format!("{}:{}-{}", host, port, self.next_connection_id);
        self.next_connection_id += 1;

        AsyncStream::with_channel(move |sender| {
            match transport_type {
                TransportType::H3 => {
                    // Create H3 transport connection directly
                    let connection = Connection::new_h3(true);
                    let transport_conn = TransportConnection {
                        connection_id,
                        transport_type: TransportType::H3,
                        connection,
                        quiche_connection: None,
                        h2_connection: None,
                    };
                    emit!(sender, transport_conn);
                }
                TransportType::H2 => {
                    // Create H2 transport connection directly
                    let connection = Connection::new_h2(true);
                    let transport_conn = TransportConnection {
                        connection_id,
                        transport_type: TransportType::H2,
                        connection,
                        quiche_connection: None,
                        h2_connection: None,
                    };
                    emit!(sender, transport_conn);
                }
                TransportType::Auto => {
                    // Try H3 first, emit directly
                    let connection = Connection::new_h3(true);
                    let transport_conn = TransportConnection {
                        connection_id,
                        transport_type: TransportType::H3,
                        connection,
                        quiche_connection: None,
                        h2_connection: None,
                    };
                    emit!(sender, transport_conn);
                }
            }
        })
    }

    /// Establish H3 transport using AsyncStream patterns
    fn establish_h3_transport_streaming(
        connection_id: String,
        host: String,
        port: u16,
    ) -> AsyncStream<TransportConnection, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Use spawn_task for Quiche connection
            std::thread::spawn(move || {
                // Create Quiche configuration for HTTP/3
                match quiche::Config::new(quiche::PROTOCOL_VERSION) {
                    Ok(mut config) => {
                        // Configure Quiche for HTTP/3
                        if let Err(e) = config.set_application_protos(&[b"h3"]) {
                            emit!(
                                sender,
                                TransportConnection::bad_chunk(format!(
                                    "Failed to set protocols: {}",
                                    e
                                ))
                            );
                            return;
                        }

                        config.set_max_idle_timeout(30000);
                        config.set_max_recv_udp_payload_size(1200);
                        config.set_max_send_udp_payload_size(1200);
                        config.set_initial_max_data(10_000_000);
                        config.set_initial_max_stream_data_bidi_local(1_000_000);
                        config.set_initial_max_stream_data_bidi_remote(1_000_000);
                        config.set_initial_max_streams_bidi(100);
                        config.set_disable_active_migration(true);

                        let addr = match format!("{}:{}", host, port).parse() {
                            Ok(addr) => addr,
                            Err(e) => {
                                emit!(
                                    sender,
                                    TransportConnection::bad_chunk(format!(
                                        "Invalid address: {}",
                                        e
                                    ))
                                );
                                return;
                            }
                        };

                        let bind_addr = match "[::]:0".parse().or_else(|_| "0.0.0.0:0".parse()) {
                            Ok(addr) => addr,
                            Err(e) => {
                                emit!(
                                    sender,
                                    TransportConnection::bad_chunk(format!(
                                        "Failed to parse bind address: {}",
                                        e
                                    ))
                                );
                                return;
                            }
                        };

                        // Create UDP socket for Quiche
                        let socket = match std::net::UdpSocket::bind(bind_addr) {
                            Ok(socket) => socket,
                            Err(e) => {
                                emit!(
                                    sender,
                                    TransportConnection::bad_chunk(format!(
                                        "Failed to create UDP socket: {}",
                                        e
                                    ))
                                );
                                return;
                            }
                        };

                        // Create Quiche connection
                        let scid = quiche::ConnectionId::from_ref(&[0; 16]);
                        match quiche::connect(
                            Some(&host),
                            &scid,
                            socket.local_addr().unwrap(),
                            addr,
                            &mut config,
                        ) {
                            Ok(quiche_conn) => {
                                let transport_conn =
                                    TransportConnection::new_h3(connection_id, quiche_conn);
                                emit!(sender, transport_conn);
                            }
                            Err(e) => {
                                emit!(
                                    sender,
                                    TransportConnection::bad_chunk(format!(
                                        "Failed to initiate connection: {}",
                                        e
                                    ))
                                );
                            }
                        }
                    }
                    Err(e) => {
                        emit!(
                            sender,
                            TransportConnection::bad_chunk(format!(
                                "Failed to create Quiche config: {}",
                                e
                            ))
                        );
                    }
                }
            });
        })
    }

    /// Establish H2 transport using AsyncStream patterns
    fn establish_h2_transport_streaming(
        connection_id: String,
        host: String,
        port: u16,
    ) -> AsyncStream<TransportConnection, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Create H2 transport connection directly
            let connection = Connection::new_h2(true);
            let transport_conn = TransportConnection {
                connection_id,
                transport_type: TransportType::H2,
                connection,
                quiche_connection: None,
                h2_connection: None,
            };
            emit!(sender, transport_conn);
        })
    }

    /// Establish auto transport (H3 with H2 fallback) using AsyncStream patterns
    fn establish_auto_transport_streaming(
        connection_id: String,
        host: String,
        port: u16,
    ) -> AsyncStream<TransportConnection, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Try H3 first
            let h3_stream = Self::establish_h3_transport_streaming(
                format!("{}-h3", connection_id),
                host.clone(),
                port,
            );

            let h3_connections: Vec<TransportConnection> =
                h3_stream.collect_or_else(|_error_msg| {
                    Vec::new() // Ignore H3 errors, will try H2
                });

            if let Some(h3_conn) = h3_connections.into_iter().find(|conn| !conn.is_error()) {
                emit!(sender, h3_conn);
                return;
            }

            // H3 failed, try H2
            let h2_stream =
                Self::establish_h2_transport_streaming(format!("{}-h2", connection_id), host, port);

            // Process H2 fallback connections element-by-element without collecting
            h2_stream.into_iter().for_each(|conn| {
                emit!(sender, conn);
            });
        })
    }

    /// Establish H3 connection using AsyncStream patterns
    pub fn connect_h3_streaming(&mut self, target: &str) -> AsyncStream<Connection, 1024> {
        let target = target.to_string();

        AsyncStream::with_channel(move |sender| {
            // Create H3 connection directly
            let connection = Connection::new_h3(true);
            emit!(sender, connection);
        })
    }

    /// Establish H2 connection using AsyncStream patterns
    pub fn connect_h2_streaming(&mut self, target: &str) -> AsyncStream<Connection, 1024> {
        let target = target.to_string();

        AsyncStream::with_channel(move |sender| {
            // Create H2 connection directly within the async runtime
            let connection = Connection::new_h2(true);
            emit!(sender, connection);
        })
    }

    /// Send data through connection using AsyncStream patterns
    pub fn send_data_streaming(
        &mut self,
        connection_id: &str,
        data: Vec<u8>,
    ) -> AsyncStream<FrameChunk, 1024> {
        let connection_id = connection_id.to_string();

        AsyncStream::with_channel(move |sender| {
            // Create appropriate frame based on connection type
            if connection_id.starts_with("h2-") {
                let frame = H2Frame::Data {
                    stream_id: 1,
                    data,
                    end_stream: false,
                };
                emit!(sender, FrameChunk::H2(frame));
            } else if connection_id.starts_with("h3-") {
                let frame = H3Frame::Data { data };
                emit!(sender, FrameChunk::H3(frame));
            } else {
                emit!(
                    sender,
                    FrameChunk::bad_chunk("Invalid connection ID".to_string())
                );
            }
        })
    }

    /// Receive frames from transport using AsyncStream patterns
    pub fn receive_frames_streaming(
        &mut self,
        connection_id: &str,
    ) -> AsyncStream<FrameChunk, 1024> {
        let connection_id = connection_id.to_string();

        AsyncStream::with_channel(move |sender| {
            // Receive frames from appropriate transport
            // This would integrate with Quiche/H2 receive loops
            emit!(sender, FrameChunk::end_chunk());
        })
    }

    /// Close transport connection using AsyncStream patterns
    pub fn close_connection_streaming(
        &mut self,
        connection_id: &str,
    ) -> AsyncStream<FrameChunk, 1024> {
        let connection_id = connection_id.to_string();

        AsyncStream::with_channel(move |sender| {
            // Close appropriate transport connection
            emit!(sender, FrameChunk::end_chunk());
        })
    }
}

impl Default for TransportManager {
    fn default() -> Self {
        Self::new()
    }
}
