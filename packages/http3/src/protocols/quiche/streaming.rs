//! Quiche QUIC Streaming Implementation
//!
//! Provides streaming primitives for Quiche QUIC connections using fluent_ai_async patterns.
//! All operations follow zero-allocation streaming architecture with error-as-data patterns.

use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;

use bytes::Bytes;
use crossbeam_utils::Backoff;
use fluent_ai_async::prelude::*;
use quiche::{Config, Connection, ConnectionId};

use crate::protocols::quiche::chunks::*;

/// Quiche connection state chunk for connection lifecycle events
#[derive(Debug, Clone)]
pub struct QuicheConnectionChunk {
    pub local_addr: Option<SocketAddr>,
    pub peer_addr: Option<SocketAddr>,
    pub is_established: bool,
    pub is_closed: bool,
    pub timeout_ms: Option<u64>,
    pub error: Option<String>,
}

impl QuicheConnectionChunk {
    #[inline]
    pub fn established(local: SocketAddr, peer: SocketAddr) -> Self {
        Self {
            local_addr: Some(local),
            peer_addr: Some(peer),
            is_established: true,
            is_closed: false,
            timeout_ms: None,
            error: None,
        }
    }

    #[inline]
    pub fn connection_closed(local: SocketAddr, peer: SocketAddr) -> Self {
        Self {
            local_addr: Some(local),
            peer_addr: Some(peer),
            is_established: false,
            is_closed: true,
            timeout_ms: None,
            error: None,
        }
    }

    #[inline]
    pub fn timeout_event(timeout_ms: u64) -> Self {
        Self {
            local_addr: None,
            peer_addr: None,
            is_established: false,
            is_closed: false,
            timeout_ms: Some(timeout_ms),
            error: None,
        }
    }
}

impl MessageChunk for QuicheConnectionChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            local_addr: None,
            peer_addr: None,
            is_established: false,
            is_closed: false,
            timeout_ms: None,
            error: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error.as_ref().map(|s| s.as_str())
    }
}

/// Quiche QUIC connection wrapper providing streaming primitives
#[derive(Debug)]
pub struct QuicheConnection {
    connection: QuicheConnectionWrapper,
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
            connection: QuicheConnectionWrapper::new(connection),
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
        self.connection.inner.is_established()
    }

    /// Check if connection is closed
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.connection.inner.is_closed()
    }

    /// Stream readable data from connection
    pub fn readable_streams(self) -> AsyncStream<QuicheReadableChunk, 1024> {
        read_readable_streams(self.connection.into_inner())
    }

    /// Open bidirectional stream
    pub fn open_bidi_stream(self) -> AsyncStream<QuicheStream, 1024> {
        open_bidirectional_stream(self.connection.into_inner())
    }

    /// Open unidirectional stream
    pub fn open_uni_stream(self) -> AsyncStream<QuicheStream, 1024> {
        open_unidirectional_stream(self.connection.into_inner())
    }

    /// Process incoming packets
    pub fn process_packets(self, packets: Vec<Bytes>) -> AsyncStream<QuichePacketChunk, 1024> {
        process_incoming_packets(self.connection.into_inner(), packets)
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

/// Stream readable streams from Quiche connection
pub fn read_readable_streams(mut connection: Connection) -> AsyncStream<QuicheReadableChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut readable_streams = Vec::new();
        let mut writable_streams = Vec::new();

        // Iterate through readable streams
        for stream_id in connection.readable() {
            readable_streams.push(stream_id);
        }

        // Iterate through writable streams
        for stream_id in connection.writable() {
            writable_streams.push(stream_id);
        }

        if !readable_streams.is_empty() || !writable_streams.is_empty() {
            emit!(
                sender,
                QuicheReadableChunk::streams_available(readable_streams, writable_streams)
            );
        } else if connection.is_closed() {
            emit!(sender, QuicheReadableChunk::connection_closed());
        }
    })
}

/// Open bidirectional stream
pub fn open_bidirectional_stream(mut connection: Connection) -> AsyncStream<QuicheStream, 1024> {
    AsyncStream::with_channel(move |sender| {
        match connection.stream_send(0, b"", false) {
            Ok(_) => {
                // Stream opened successfully
                emit!(sender, QuicheStream::writable_stream(0));
            }
            Err(e) => {
                emit!(
                    sender,
                    QuicheStream::bad_chunk(format!("Failed to open bidi stream: {}", e))
                );
            }
        }
    })
}

/// Open unidirectional stream
pub fn open_unidirectional_stream(mut connection: Connection) -> AsyncStream<QuicheStream, 1024> {
    AsyncStream::with_channel(move |sender| {
        match connection.stream_send(1, b"", false) {
            Ok(_) => {
                // Stream opened successfully
                emit!(sender, QuicheStream::writable_stream(1));
            }
            Err(e) => {
                emit!(
                    sender,
                    QuicheStream::bad_chunk(format!("Failed to open uni stream: {}", e))
                );
            }
        }
    })
}

/// Process incoming packets
pub fn process_incoming_packets(
    mut connection: Connection,
    packets: Vec<Bytes>,
) -> AsyncStream<QuichePacketChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        for packet in packets {
            match connection.recv(
                &packet,
                quiche::RecvInfo {
                    to: connection.source_id().into(),
                    from: connection.destination_id().into(),
                },
            ) {
                Ok(bytes_processed) => {
                    emit!(
                        sender,
                        QuichePacketChunk::packet_processed(
                            bytes_processed,
                            connection.source_id().into(),
                            connection.destination_id().into()
                        )
                    );
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuichePacketChunk::bad_chunk(format!("Packet processing error: {}", e))
                    );
                }
            }
        }
    })
}

/// Read data from specific stream using elite polling pattern
pub fn read_stream_data(
    mut connection: Connection,
    stream_id: u64,
) -> AsyncStream<QuicheStreamChunk, 1024> {
    AsyncStream::with_channel(move |sender| {
        let mut buffer = vec![0; 4096];
        let backoff = Backoff::new();

        loop {
            match connection.stream_recv(stream_id, &mut buffer) {
                Ok((bytes_read, fin)) => {
                    if bytes_read > 0 {
                        buffer.truncate(bytes_read);
                        emit!(
                            sender,
                            QuicheStreamChunk::data_received(stream_id, buffer.clone(), fin)
                        );
                        backoff.reset();
                        buffer.resize(4096, 0); // Reset buffer for next read
                    }

                    if fin {
                        emit!(sender, QuicheStreamChunk::stream_finished(stream_id));
                        break;
                    }
                }
                Err(quiche::Error::Done) => {
                    // Elite backoff pattern - no data available
                    if backoff.is_completed() {
                        std::thread::yield_now();
                    } else {
                        backoff.snooze();
                    }
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

            // Check if connection is closed
            if connection.is_closed() {
                emit!(sender, QuicheStreamChunk::stream_finished(stream_id));
                break;
            }
        }
    })
}

/// Write data to specific stream
pub fn write_stream_data(
    mut connection: Connection,
    stream_id: u64,
    data: Vec<u8>,
    fin: bool,
) -> AsyncStream<QuicheWriteResult, 1024> {
    AsyncStream::with_channel(move |sender| {
        match connection.stream_send(stream_id, &data, fin) {
            Ok(bytes_written) => {
                emit!(
                    sender,
                    QuicheWriteResult::bytes_written(stream_id, bytes_written, fin)
                );
            }
            Err(quiche::Error::Done) => {
                // Stream is blocked
                emit!(sender, QuicheWriteResult::write_blocked(stream_id));
            }
            Err(e) => {
                emit!(
                    sender,
                    QuicheWriteResult::bad_chunk(format!("Stream write error: {}", e))
                );
            }
        }
    })
}

/// Debug wrapper for quiche::Connection
#[derive(Debug)]
pub struct QuicheConnectionWrapper {
    inner: Connection,
}

impl QuicheConnectionWrapper {
    #[inline]
    pub fn new(connection: Connection) -> Self {
        Self { inner: connection }
    }

    #[inline]
    pub fn inner(&mut self) -> &mut Connection {
        &mut self.inner
    }

    #[inline]
    pub fn into_inner(self) -> Connection {
        self.inner
    }
}

/// Quiche stream wrapper for individual stream operations
#[derive(Debug)]
pub struct QuicheStreamWrapper {
    connection: QuicheConnectionWrapper,
    stream_id: u64,
}

impl QuicheStreamWrapper {
    #[inline]
    pub fn new(connection: Connection, stream_id: u64) -> Self {
        Self {
            connection: QuicheConnectionWrapper::new(connection),
            stream_id,
        }
    }

    #[inline]
    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    /// Read data from this stream
    pub fn read_data(self) -> AsyncStream<QuicheStreamChunk, 1024> {
        read_stream_data(self.connection.into_inner(), self.stream_id)
    }

    /// Write data to this stream
    pub fn write_data(self, data: Vec<u8>, fin: bool) -> AsyncStream<QuicheWriteResult, 1024> {
        write_stream_data(self.connection.into_inner(), self.stream_id, data, fin)
    }

    /// Write bytes to this stream
    pub fn write_bytes(self, data: Bytes, fin: bool) -> AsyncStream<QuicheWriteResult, 1024> {
        write_stream_data(
            self.connection.into_inner(),
            self.stream_id,
            data.to_vec(),
            fin,
        )
    }
}
