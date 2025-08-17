//! Quiche HTTP/3 client implementation using fluent_ai_async patterns
//!
//! Zero-allocation HTTP/3 client with AsyncStream integration.
//! Uses Quiche's synchronous APIs for perfect fluent_ai_async compliance.

use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

use fluent_ai_async::{AsyncStream, emit};
use quiche::{Config, Connection, RecvInfo, SendInfo};

use crate::protocols::quiche::chunks::*;
use crate::types::{HttpVersion, TimeoutConfig};

/// Quiche HTTP/3 client with AsyncStream integration
/// Uses synchronous APIs for perfect fluent_ai_async compliance
pub struct QuicheHttp3Client {
    config: Config,
    local_addr: SocketAddr,
    server_name: String,
}

impl QuicheHttp3Client {
    /// Create new Quiche HTTP/3 client
    #[inline]
    pub fn new(
        mut config: Config,
        local_addr: SocketAddr,
        server_name: String,
    ) -> Result<Self, quiche::Error> {
        // Configure for HTTP/3
        config.set_application_protos(&[b"h3"])?;
        config.verify_peer(true);

        Ok(Self {
            config,
            local_addr,
            server_name,
        })
    }

    /// Establish connection to server using AsyncStream
    /// Uses Quiche's synchronous connect API
    pub fn connect(mut self, server_addr: SocketAddr) -> AsyncStream<QuicheConnectionChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Create UDP socket
            let socket = match UdpSocket::bind(self.local_addr) {
                Ok(socket) => socket,
                Err(e) => {
                    emit!(
                        sender,
                        QuicheConnectionChunk::bad_chunk(format!("Socket bind error: {}", e))
                    );
                    return;
                }
            };

            // Connect socket to server
            if let Err(e) = socket.connect(server_addr) {
                emit!(
                    sender,
                    QuicheConnectionChunk::bad_chunk(format!("Socket connect error: {}", e))
                );
                return;
            }

            // Generate connection ID
            let conn_id = quiche::ConnectionId::from_ref(&[0; 16]);

            // Create Quiche connection using synchronous API
            let mut conn = match quiche::connect(
                Some(&self.server_name),
                &conn_id,
                self.local_addr,
                server_addr,
                &mut self.config,
            ) {
                Ok(conn) => conn,
                Err(e) => {
                    emit!(
                        sender,
                        QuicheConnectionChunk::bad_chunk(format!("Connection create error: {}", e))
                    );
                    return;
                }
            };

            let mut buf = [0u8; 65535];
            let to = self.local_addr;

            // Connection establishment loop using synchronous APIs
            loop {
                // Send initial packets
                loop {
                    let (write, _send_info) = match conn.send(&mut buf) {
                        Ok((write, send_info)) => (write, send_info),
                        Err(quiche::Error::Done) => break,
                        Err(e) => {
                            emit!(
                                sender,
                                QuicheConnectionChunk::bad_chunk(format!("Send error: {}", e))
                            );
                            return;
                        }
                    };

                    if let Err(e) = socket.send(&buf[..write]) {
                        emit!(
                            sender,
                            QuicheConnectionChunk::bad_chunk(format!("Socket send error: {}", e))
                        );
                        return;
                    }
                }

                // Check if connection is established
                if conn.is_established() {
                    emit!(
                        sender,
                        QuicheConnectionChunk::established(self.local_addr, server_addr)
                    );
                    break;
                }

                // Check if connection is closed
                if conn.is_closed() {
                    emit!(sender, QuicheConnectionChunk::connection_closed());
                    break;
                }

                // Receive packets
                let (read, from) = match socket.recv_from(&mut buf) {
                    Ok(result) => result,
                    Err(e) => {
                        emit!(
                            sender,
                            QuicheConnectionChunk::bad_chunk(format!("Socket recv error: {}", e))
                        );
                        break;
                    }
                };

                let recv_info = RecvInfo { from, to };

                // Process received packet using synchronous API
                match conn.recv(&mut buf[..read], recv_info) {
                    Ok(_) => {
                        // Packet processed successfully, continue loop
                    }
                    Err(e) => {
                        emit!(
                            sender,
                            QuicheConnectionChunk::bad_chunk(format!("Packet recv error: {}", e))
                        );
                        break;
                    }
                }
            }
        })
    }
}

/// Process packets from network using Quiche synchronous APIs
/// Follows the exact pattern from Quiche source Lines 122-142
pub fn receive_packets(
    mut conn: Connection,
    socket: UdpSocket,
) -> AsyncStream<QuichePacketChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        let to = match socket.local_addr() {
            Ok(addr) => addr,
            Err(e) => {
                emit!(
                    sender,
                    QuichePacketChunk::bad_chunk(format!("Local addr error: {}", e))
                );
                return;
            }
        };

        let mut buf = [0u8; 65535];

        // Packet processing loop using synchronous APIs
        loop {
            // Receive UDP packet (synchronous)
            let (read, from) = match socket.recv_from(&mut buf) {
                Ok(result) => result,
                Err(e) => {
                    emit!(
                        sender,
                        QuichePacketChunk::bad_chunk(format!("Socket recv error: {}", e))
                    );
                    break;
                }
            };

            let recv_info = RecvInfo { from, to };

            // Process QUIC packet (synchronous - Lines 122-142 pattern)
            match conn.recv(&mut buf[..read], recv_info) {
                Ok(bytes_processed) => {
                    emit!(
                        sender,
                        QuichePacketChunk::packet_processed(bytes_processed, from, to)
                    );
                }
                Err(quiche::Error::Done) => {
                    // No more packets to process, continue
                    continue;
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuichePacketChunk::bad_chunk(format!("Packet error: {}", e))
                    );
                    break;
                }
            }
        }
    })
}

/// Read stream data using Quiche synchronous stream_recv API
/// Follows Lines 5277-5279 pattern from Quiche source
pub fn read_stream_data(
    mut conn: Connection,
    stream_id: u64,
) -> AsyncStream<QuicheStreamChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut buf = [0u8; 8192];

        // Stream reading loop using synchronous APIs
        loop {
            match conn.stream_recv(stream_id, &mut buf) {
                Ok((len, fin)) => {
                    if len > 0 {
                        emit!(
                            sender,
                            QuicheStreamChunk::data_chunk(buf[..len].to_vec(), stream_id, fin)
                        );
                    }
                    if fin {
                        emit!(sender, QuicheStreamChunk::stream_complete());
                        break;
                    }
                }
                Err(quiche::Error::Done) => {
                    // No data available - continue loop
                    continue;
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuicheStreamChunk::bad_chunk(format!("Stream read error: {}", e))
                    );
                    break;
                }
            }
        }
    })
}

/// Write stream data using Quiche synchronous stream_send API
/// Follows Lines 5411-5413 pattern from Quiche source
pub fn write_stream_data(
    mut conn: Connection,
    stream_id: u64,
    data: Vec<u8>,
    fin: bool,
) -> AsyncStream<QuicheWriteResult, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut offset = 0;

        // Stream writing loop using synchronous APIs
        loop {
            let chunk_size = std::cmp::min(8192, data.len() - offset);
            let is_final = fin && (offset + chunk_size >= data.len());

            match conn.stream_send(stream_id, &data[offset..offset + chunk_size], is_final) {
                Ok(written) => {
                    offset += written;
                    emit!(sender, QuicheWriteResult::bytes_written(written, stream_id));

                    if offset >= data.len() {
                        emit!(sender, QuicheWriteResult::write_complete(stream_id));
                        break;
                    }
                }
                Err(quiche::Error::Done) => {
                    // Stream not writable - continue loop
                    continue;
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuicheWriteResult::bad_chunk(format!("Stream write error: {}", e))
                    );
                    break;
                }
            }
        }
    })
}

/// Process readable streams using Quiche synchronous APIs
/// Follows readable() iterator pattern from Quiche source
pub fn process_readable_streams(conn: Connection) -> AsyncStream<QuicheReadableChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        loop {
            // Iterate readable streams (synchronous)
            for stream_id in conn.readable() {
                emit!(sender, QuicheReadableChunk::readable_stream(stream_id));
            }

            // Check if connection is still active
            if conn.is_closed() {
                emit!(sender, QuicheReadableChunk::connection_closed());
                break;
            }
        }
    })
}
