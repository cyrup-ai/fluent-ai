//! Direct TCP connection establishment with DNS resolution
//!
//! Handles direct connection establishment with zero-allocation streaming,
//! DNS resolution, and connection timeout management.

use std::net::{TcpStream, SocketAddr};

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http::Uri;

use super::super::chunks::TcpConnectionChunk;
use super::core::ConnectorService;

/// Extract local and remote addresses from stream and emit connection event
/// Preserves connection attempt ordering for IPv4/IPv6 happy eyeballs logic
pub(super) fn emit_stream_connection(
    stream: TcpStream,
    sender: &fluent_ai_async::AsyncStreamSender<TcpConnectionChunk>,
) -> Result<(), String> {
    let local_addr = stream.local_addr()
        .map_err(|e| format!("Failed to get local address: {}", e))?;
    let remote_addr = stream.peer_addr()
        .map_err(|e| format!("Failed to get remote address: {}", e))?;
    
    // Wrap TcpStream in ConnectionTrait implementation
    let conn_trait: Box<dyn crate::connect::types::ConnectionTrait + Send> = 
        Box::new(crate::connect::types::TcpConnection::from(stream));
    
    emit!(sender, TcpConnectionChunk::connected(local_addr, remote_addr, Some(conn_trait)));
    Ok(())
}

impl ConnectorService {
    /// Connect with optional proxy handling
    pub fn connect_with_maybe_proxy(
        &self,
        dst: Uri,
        via_proxy: bool,
    ) -> AsyncStream<TcpConnectionChunk, 1024> {
        let connector_service = self.clone();
        let destination = dst.clone();

        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                let host = match destination.host() {
                    Some(h) => h,
                    None => {
                        emit!(
                            sender,
                            TcpConnectionChunk::bad_chunk("URI missing host".to_string())
                        );
                        return;
                    }
                };

                let port =
                    destination
                        .port_u16()
                        .unwrap_or_else(|| match destination.scheme_str() {
                            Some("https") => 443,
                            Some("http") => 80,
                            _ => 80,
                        });

                // Resolve addresses with zero allocation
                let addresses = match super::super::tcp::resolve_host_sync(host, port) {
                    Ok(addrs) => addrs,
                    Err(e) => {
                        emit!(
                            sender,
                            TcpConnectionChunk::bad_chunk(format!("DNS resolution failed: {}", e))
                        );
                        return;
                    }
                };

                if addresses.is_empty() {
                    emit!(
                        sender,
                        TcpConnectionChunk::bad_chunk("No addresses resolved".to_string())
                    );
                    return;
                }

                // Try connecting to each address with elite polling
                for addr in addresses {
                    match connector_service.connect_timeout {
                        Some(timeout) => {
                            match TcpStream::connect_timeout(&addr, timeout) {
                                Ok(mut stream) => {
                                    // Configure socket for optimal performance
                                    if connector_service.nodelay {
                                        let _ = stream.set_nodelay(true);
                                    }

                                    // Extract addresses and emit connection event
                                    if let Err(error) = emit_stream_connection(stream, &sender) {
                                        emit!(sender, TcpConnectionChunk::bad_chunk(error));
                                        return;
                                    }
                                    return;
                                }
                                Err(_) => continue,
                            }
                        }
                        None => match TcpStream::connect(&addr) {
                            Ok(mut stream) => {
                                if connector_service.nodelay {
                                    let _ = stream.set_nodelay(true);
                                }

                                // Extract addresses and emit connection event
                                if let Err(error) = emit_stream_connection(stream, &sender) {
                                    emit!(sender, TcpConnectionChunk::bad_chunk(error));
                                    return;
                                }
                                return;
                            }
                            Err(_) => continue,
                        },
                    }
                }

                emit!(
                    sender,
                    TcpConnectionChunk::bad_chunk("Failed to connect to any address".to_string())
                );
            });
        })
    }
}
