//! Quiche connection management with packet processing loops
//!
//! Manages QUIC connection lifecycle using Quiche's synchronous APIs
//! and fluent_ai_async streaming patterns.

use std::collections::VecDeque;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use fluent_ai_async::{AsyncStream, emit};
use quiche::{Config, Connection, RecvInfo, SendInfo};

use crate::protocols::quiche::chunks::*;
use crate::types::{HttpVersion, TimeoutConfig};

/// Quiche connection manager with packet processing loops
pub struct QuicheConnection {
    connection: Connection,
    socket: UdpSocket,
    local_addr: SocketAddr,
    peer_addr: SocketAddr,
    send_queue: VecDeque<Vec<u8>>,
}

impl QuicheConnection {
    /// Create new connection from established Quiche connection
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
            send_queue: VecDeque::new(),
        }
    }

    /// Main packet processing loop using synchronous Quiche APIs
    /// Handles both incoming and outgoing packets
    pub fn process_packets(mut self) -> AsyncStream<QuichePacketChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut buf = [0u8; 65535];
            let mut last_send = Instant::now();

            // Main packet processing loop
            loop {
                // Process outgoing packets first
                loop {
                    let (write, send_info) = match self.connection.send(&mut buf) {
                        Ok((write, send_info)) => (write, send_info),
                        Err(quiche::Error::Done) => break,
                        Err(e) => {
                            emit!(
                                sender,
                                QuichePacketChunk::bad_chunk(format!("Send error: {}", e))
                            );
                            return;
                        }
                    };

                    // Send packet via UDP socket
                    if let Err(e) = self.socket.send(&buf[..write]) {
                        emit!(
                            sender,
                            QuichePacketChunk::bad_chunk(format!("Socket send error: {}", e))
                        );
                        return;
                    }

                    emit!(
                        sender,
                        QuichePacketChunk::packet_sent(write, self.local_addr, self.peer_addr)
                    );
                    last_send = Instant::now();
                }

                // Check connection state
                if self.connection.is_closed() {
                    emit!(sender, QuichePacketChunk::connection_closed());
                    break;
                }

                // Set socket timeout for non-blocking receive
                if let Err(e) = self
                    .socket
                    .set_read_timeout(Some(Duration::from_millis(10)))
                {
                    emit!(
                        sender,
                        QuichePacketChunk::bad_chunk(format!("Socket timeout error: {}", e))
                    );
                    return;
                }

                // Receive incoming packets
                match self.socket.recv_from(&mut buf) {
                    Ok((read, from)) => {
                        let recv_info = RecvInfo {
                            from,
                            to: self.local_addr,
                        };

                        // Process received packet using synchronous API
                        match self.connection.recv(&mut buf[..read], recv_info) {
                            Ok(bytes_processed) => {
                                emit!(
                                    sender,
                                    QuichePacketChunk::packet_processed(
                                        bytes_processed,
                                        from,
                                        self.local_addr
                                    )
                                );
                            }
                            Err(quiche::Error::Done) => {
                                // Packet processed but no action needed
                                continue;
                            }
                            Err(e) => {
                                emit!(
                                    sender,
                                    QuichePacketChunk::bad_chunk(format!(
                                        "Packet recv error: {}",
                                        e
                                    ))
                                );
                                continue;
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No packets available - check for timeout
                        if self.connection.timeout().is_some()
                            && last_send.elapsed() > Duration::from_millis(100)
                        {
                            // Handle timeout by calling on_timeout
                            self.connection.on_timeout();
                            emit!(sender, QuichePacketChunk::timeout_handled());
                        }
                        continue;
                    }
                    Err(e) => {
                        emit!(
                            sender,
                            QuichePacketChunk::bad_chunk(format!("Socket recv error: {}", e))
                        );
                        break;
                    }
                }
            }
        })
    }

    /// Open bidirectional stream using synchronous Quiche API
    pub fn open_bidirectional_stream(mut self) -> AsyncStream<QuicheStreamChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Open bidirectional stream using synchronous API
            match self.connection.stream_open_bidi() {
                Ok(stream_id) => {
                    emit!(sender, QuicheStreamChunk::stream_opened(stream_id, true));
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuicheStreamChunk::bad_chunk(format!("Stream open error: {}", e))
                    );
                }
            }
        })
    }

    /// Open unidirectional stream using synchronous Quiche API
    pub fn open_unidirectional_stream(mut self) -> AsyncStream<QuicheStreamChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Open unidirectional stream using synchronous API
            match self.connection.stream_open_uni() {
                Ok(stream_id) => {
                    emit!(sender, QuicheStreamChunk::stream_opened(stream_id, false));
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuicheStreamChunk::bad_chunk(format!("Stream open error: {}", e))
                    );
                }
            }
        })
    }

    /// Process all readable streams using synchronous APIs
    pub fn process_readable_streams(self) -> AsyncStream<QuicheReadableChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut buf = [0u8; 8192];

            loop {
                let mut streams_processed = false;

                // Iterate through all readable streams
                for stream_id in self.connection.readable() {
                    streams_processed = true;
                    emit!(sender, QuicheReadableChunk::readable_stream(stream_id));

                    // Read data from this stream
                    loop {
                        match self.connection.stream_recv(stream_id, &mut buf) {
                            Ok((len, fin)) => {
                                if len > 0 {
                                    emit!(
                                        sender,
                                        QuicheReadableChunk::stream_data(
                                            stream_id,
                                            buf[..len].to_vec(),
                                            fin
                                        )
                                    );
                                }
                                if fin {
                                    emit!(sender, QuicheReadableChunk::stream_finished(stream_id));
                                    break;
                                }
                            }
                            Err(quiche::Error::Done) => {
                                // No more data available on this stream
                                break;
                            }
                            Err(e) => {
                                emit!(
                                    sender,
                                    QuicheReadableChunk::bad_chunk(format!(
                                        "Stream {} read error: {}",
                                        stream_id, e
                                    ))
                                );
                                break;
                            }
                        }
                    }
                }

                // If no streams were processed and connection is closed, exit
                if !streams_processed && self.connection.is_closed() {
                    emit!(sender, QuicheReadableChunk::connection_closed());
                    break;
                }

                // Brief pause to prevent busy loop
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    }

    /// Write data to stream using synchronous Quiche API
    pub fn write_to_stream(
        mut self,
        stream_id: u64,
        data: Vec<u8>,
        fin: bool,
    ) -> AsyncStream<QuicheWriteResult, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut offset = 0;
            let total_len = data.len();

            // Stream writing loop
            loop {
                let remaining = total_len - offset;
                if remaining == 0 {
                    emit!(sender, QuicheWriteResult::write_complete(stream_id));
                    break;
                }

                let chunk_size = std::cmp::min(8192, remaining);
                let is_final = fin && (offset + chunk_size >= total_len);

                match self.connection.stream_send(
                    stream_id,
                    &data[offset..offset + chunk_size],
                    is_final,
                ) {
                    Ok(written) => {
                        offset += written;
                        emit!(sender, QuicheWriteResult::bytes_written(written, stream_id));

                        if written < chunk_size {
                            // Stream is not ready for more data, continue loop
                            continue;
                        }
                    }
                    Err(quiche::Error::Done) => {
                        // Stream not writable right now, continue loop
                        continue;
                    }
                    Err(e) => {
                        emit!(
                            sender,
                            QuicheWriteResult::bad_chunk(format!(
                                "Stream {} write error: {}",
                                stream_id, e
                            ))
                        );
                        break;
                    }
                }
            }
        })
    }

    /// Get connection statistics using synchronous Quiche API
    pub fn get_stats(self) -> AsyncStream<QuicheConnectionChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Get connection statistics
            let stats = self.connection.stats();

            emit!(
                sender,
                QuicheConnectionChunk::connection_stats(
                    stats.recv, stats.sent, stats.lost, stats.rtt, stats.cwnd,
                )
            );

            // Check if connection is established
            if self.connection.is_established() {
                emit!(
                    sender,
                    QuicheConnectionChunk::established(self.local_addr, self.peer_addr)
                );
            } else if self.connection.is_closed() {
                emit!(sender, QuicheConnectionChunk::connection_closed());
            }
        })
    }

    /// Close connection gracefully using synchronous Quiche API
    pub fn close_connection(
        mut self,
        err: u64,
        reason: &[u8],
    ) -> AsyncStream<QuicheConnectionChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Close connection with error code and reason
            match self.connection.close(false, err, reason) {
                Ok(_) => {
                    emit!(sender, QuicheConnectionChunk::connection_closing(err));

                    // Send final packets
                    let mut buf = [0u8; 65535];
                    loop {
                        let (write, _send_info) = match self.connection.send(&mut buf) {
                            Ok((write, send_info)) => (write, send_info),
                            Err(quiche::Error::Done) => break,
                            Err(e) => {
                                emit!(
                                    sender,
                                    QuicheConnectionChunk::bad_chunk(format!(
                                        "Close send error: {}",
                                        e
                                    ))
                                );
                                return;
                            }
                        };

                        if let Err(e) = self.socket.send(&buf[..write]) {
                            emit!(
                                sender,
                                QuicheConnectionChunk::bad_chunk(format!(
                                    "Close socket send error: {}",
                                    e
                                ))
                            );
                            return;
                        }
                    }

                    emit!(sender, QuicheConnectionChunk::connection_closed());
                }
                Err(e) => {
                    emit!(
                        sender,
                        QuicheConnectionChunk::bad_chunk(format!("Connection close error: {}", e))
                    );
                }
            }
        })
    }
}
