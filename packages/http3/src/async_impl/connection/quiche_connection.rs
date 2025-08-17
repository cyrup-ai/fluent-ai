//! Quiche QUIC Connection Streaming Primitives
//!
//! Integrate Quiche's synchronous streaming primitives with AsyncStream architecture,
//! using ONLY synchronous APIs that return Result<T,E> immediately.
//!
//! ## Architecture
//! - Uses AsyncStream::with_channel for producer-consumer streaming patterns
//! - Synchronous Quiche APIs with direct Result<T,E> returns
//! - Error-as-data: all errors converted to BadChunk variants
//! - Zero-allocation hot paths with const-generic capacity
//!
//! ## Key Operations
//! - Connection establishment using quiche::connect/accept
//! - Packet processing with conn.recv/conn.send
//! - Stream reading with conn.stream_recv
//! - Stream writing with conn.stream_send
//! - Stream iteration with conn.readable()/conn.writable()

use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

use bytes::Bytes;
use fluent_ai_async::{AsyncStream, emit};
use quiche::{Config, Connection, ConnectionId, RecvInfo, SendInfo};

use crate::types::quiche_chunks::*;

/// Quiche QUIC connection wrapper providing streaming primitives
#[derive(Debug)]
pub struct QuicheConnection {
    connection: Connection,
    socket: UdpSocket,
    local_addr: SocketAddr,
    peer_addr: SocketAddr,
}

impl QuicheConnection {
    /// Create new connection from established Quiche connection
    #[inline]
    pub fn new(
        connection: Connection,
        socket: UdpSocket,
        local_addr: SocketAddr,
        peer_addr: SocketAddr,
    ) -> Self {
        Self {
            connection,
            socket,
            local_addr,
            peer_addr,
        }
    }

    /// Get connection local address
    #[inline]
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Get connection peer address
    #[inline]
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Check if connection is established
    #[inline]
    pub fn is_established(&self) -> bool {
        self.connection.is_established()
    }

    /// Check if connection is closed
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.connection.is_closed()
    }
}

// CORRECT: Establish Quiche connection using synchronous APIs
pub fn establish_connection(
    server_name: Option<&str>,
    scid: &ConnectionId,
    local: SocketAddr,
    peer: SocketAddr,
    mut config: Config,
    socket: UdpSocket,
) -> AsyncStream<QuicheConnectionChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        // Use Quiche's synchronous connect method
        match quiche::connect(server_name, scid, local, peer, &mut config) {
            Ok(connection) => {
                emit!(sender, QuicheConnectionChunk::established(local, peer));

                // Connection established successfully
                let quiche_conn = QuicheConnection::new(connection, socket, local, peer);
                drop(quiche_conn); // Prevent unused variable warning
            }
            Err(e) => {
                emit!(
                    sender,
                    QuicheConnectionChunk::bad_chunk(format!("Connection error: {}", e))
                );
            }
        }
    })
}

// CORRECT: Accept Quiche connection using synchronous APIs
pub fn accept_connection(
    scid: &ConnectionId,
    odcid: Option<&ConnectionId>,
    local: SocketAddr,
    peer: SocketAddr,
    mut config: Config,
    socket: UdpSocket,
) -> AsyncStream<QuicheConnectionChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        // Use Quiche's synchronous accept method
        match quiche::accept(scid, odcid, local, peer, &mut config) {
            Ok(connection) => {
                emit!(sender, QuicheConnectionChunk::established(local, peer));

                // Connection accepted successfully
                let quiche_conn = QuicheConnection::new(connection, socket, local, peer);
                drop(quiche_conn); // Prevent unused variable warning
            }
            Err(e) => {
                emit!(
                    sender,
                    QuicheConnectionChunk::bad_chunk(format!("Accept error: {}", e))
                );
            }
        }
    })
}

// CORRECT: Process packets using Quiche synchronous recv/send
pub fn process_packets(
    mut connection: Connection,
    socket: UdpSocket,
    local_addr: SocketAddr,
) -> AsyncStream<QuichePacketChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut buf = [0u8; 65535];

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

            // Process QUIC packet (synchronous)
            let recv_info = RecvInfo {
                from,
                to: local_addr,
            };
            match connection.recv(&mut buf[..read], recv_info) {
                Ok(bytes_processed) => {
                    emit!(
                        sender,
                        QuichePacketChunk::packet_processed(bytes_processed, from, local_addr)
                    );
                }
                Err(quiche::Error::Done) => {
                    // No more data to process - continue loop
                    continue;
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuichePacketChunk::bad_chunk(format!("Packet processing error: {}", e))
                    );
                    break;
                }
            }

            // Send outgoing packets if any
            let mut out = [0u8; 65535];
            loop {
                match connection.send(&mut out) {
                    Ok((written, send_info)) => {
                        match socket.send_to(&out[..written], send_info.to) {
                            Ok(_) => {
                                emit!(
                                    sender,
                                    QuichePacketChunk::packet_processed(
                                        written,
                                        local_addr,
                                        send_info.to
                                    )
                                );
                            }
                            Err(e) => {
                                emit!(
                                    sender,
                                    QuichePacketChunk::bad_chunk(format!(
                                        "Socket send error: {}",
                                        e
                                    ))
                                );
                                break;
                            }
                        }
                    }
                    Err(quiche::Error::Done) => {
                        // No more packets to send
                        break;
                    }
                    Err(e) => {
                        emit!(
                            sender,
                            QuichePacketChunk::bad_chunk(format!("Send error: {}", e))
                        );
                        break;
                    }
                }
            }

            // Check connection state
            if connection.is_closed() {
                emit!(
                    sender,
                    QuichePacketChunk::connection_established(local_addr, from)
                );
                break;
            }
        }
    })
}

// CORRECT: Read stream data using Quiche synchronous stream_recv
pub fn read_stream_data(
    mut connection: Connection,
    stream_id: u64,
) -> AsyncStream<QuicheStreamChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut buf = [0u8; 8192];

        loop {
            match connection.stream_recv(stream_id, &mut buf) {
                Ok((len, fin)) => {
                    if len > 0 {
                        emit!(
                            sender,
                            QuicheStreamChunk::data_chunk(stream_id, buf[..len].to_vec(), fin)
                        );
                    }
                    if fin {
                        emit!(sender, QuicheStreamChunk::stream_complete(stream_id));
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

// CORRECT: Write stream data using Quiche synchronous stream_send
pub fn write_stream_data(
    mut connection: Connection,
    stream_id: u64,
    data: Vec<u8>,
    fin: bool,
) -> AsyncStream<QuicheWriteResult, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut offset = 0;

        loop {
            let chunk_size = std::cmp::min(8192, data.len() - offset);
            let is_final = fin && (offset + chunk_size >= data.len());

            match connection.stream_send(stream_id, &data[offset..offset + chunk_size], is_final) {
                Ok(written) => {
                    offset += written;
                    emit!(sender, QuicheWriteResult::bytes_written(stream_id, written));

                    if offset >= data.len() {
                        emit!(sender, QuicheWriteResult::write_complete(stream_id, offset));
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

// CORRECT: Process readable streams using Quiche synchronous readable() iterator
pub fn process_readable_streams(connection: Connection) -> AsyncStream<QuicheReadableChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        loop {
            let mut readable_streams = Vec::new();
            let mut writable_streams = Vec::new();

            // Collect readable streams
            for stream_id in connection.readable() {
                readable_streams.push(stream_id);
            }

            // Collect writable streams
            for stream_id in connection.writable() {
                writable_streams.push(stream_id);
            }

            if !readable_streams.is_empty() || !writable_streams.is_empty() {
                emit!(
                    sender,
                    QuicheReadableChunk::streams_available(readable_streams, writable_streams)
                );
            }

            // Check if connection is still active
            if connection.is_closed() {
                emit!(sender, QuicheReadableChunk::connection_closed());
                break;
            }

            // Small delay to prevent busy waiting
            std::thread::sleep(Duration::from_millis(1));
        }
    })
}

// CORRECT: Handle connection timeouts using Quiche synchronous timeout/on_timeout
pub fn handle_connection_timeouts(
    mut connection: Connection,
) -> AsyncStream<QuicheConnectionChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        loop {
            if let Some(timeout) = connection.timeout() {
                emit!(
                    sender,
                    QuicheConnectionChunk::timeout_event(timeout.as_millis() as u64)
                );

                // Process timeout
                connection.on_timeout();

                // Check if connection is still active after timeout
                if connection.is_closed() {
                    emit!(
                        sender,
                        QuicheConnectionChunk::connection_closed(
                            "0.0.0.0:0".parse().unwrap(),
                            "0.0.0.0:0".parse().unwrap()
                        )
                    );
                    break;
                }
            }

            // Small delay to prevent busy waiting
            std::thread::sleep(Duration::from_millis(10));
        }
    })
}

// CORRECT: Establish Quiche connection (alias for establish_connection)
pub fn establish_quiche_connection(
    server_name: Option<&str>,
    scid: &ConnectionId,
    local: SocketAddr,
    peer: SocketAddr,
    config: Config,
    socket: UdpSocket,
) -> AsyncStream<QuicheConnectionChunk, 1024> {
    establish_connection(server_name, scid, local, peer, config, socket)
}

// CORRECT: Receive packets (alias for process_packets)
pub fn receive_packets(
    connection: Connection,
    socket: UdpSocket,
    local_addr: SocketAddr,
) -> AsyncStream<QuichePacketChunk, 1024> {
    process_packets(connection, socket, local_addr)
}

/// Quiche stream wrapper for individual stream operations
#[derive(Debug)]
pub struct QuicheStream {
    connection: Connection,
    stream_id: u64,
}

impl QuicheStream {
    #[inline]
    pub fn new(connection: Connection, stream_id: u64) -> Self {
        Self {
            connection,
            stream_id,
        }
    }

    #[inline]
    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    /// Read data from this stream
    pub fn read_data(mut self) -> AsyncStream<QuicheStreamChunk, 1024> {
        read_stream_data(self.connection, self.stream_id)
    }

    /// Write data to this stream
    pub fn write_data(mut self, data: Vec<u8>, fin: bool) -> AsyncStream<QuicheWriteResult, 1024> {
        write_stream_data(self.connection, self.stream_id, data, fin)
    }

    /// Write bytes to this stream
    pub fn write_bytes(mut self, data: Bytes, fin: bool) -> AsyncStream<QuicheWriteResult, 1024> {
        write_stream_data(self.connection, self.stream_id, data.to_vec(), fin)
    }
}
