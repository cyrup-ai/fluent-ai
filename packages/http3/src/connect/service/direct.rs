//! Direct TCP connection establishment with DNS resolution
//!
//! Handles direct connection establishment with zero-allocation streaming,
//! DNS resolution, and connection timeout management.

use std::net::TcpStream;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http::Uri;

use super::super::types::TcpStreamWrapper;
use super::core::ConnectorService;

impl ConnectorService {
    /// Connect with optional proxy handling
    pub fn connect_with_maybe_proxy(
        &self,
        dst: Uri,
        via_proxy: bool,
    ) -> AsyncStream<TcpStreamWrapper> {
        let connector_service = self.clone();
        let destination = dst.clone();

        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                let host = match destination.host() {
                    Some(h) => h,
                    None => {
                        emit!(
                            sender,
                            TcpStreamWrapper::bad_chunk("URI missing host".to_string())
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
                            TcpStreamWrapper::bad_chunk(format!("DNS resolution failed: {}", e))
                        );
                        return;
                    }
                };

                if addresses.is_empty() {
                    emit!(
                        sender,
                        TcpStreamWrapper::bad_chunk("No addresses resolved".to_string())
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

                                    emit!(sender, TcpStreamWrapper(stream));
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

                                emit!(sender, TcpStreamWrapper(stream));
                                return;
                            }
                            Err(_) => continue,
                        },
                    }
                }

                emit!(
                    sender,
                    TcpStreamWrapper::bad_chunk("Failed to connect to any address".to_string())
                );
            });
        })
    }
}
