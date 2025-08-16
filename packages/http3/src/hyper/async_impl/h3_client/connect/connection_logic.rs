//! Connection logic for H3 client
//!
//! URI-based connection establishment, address resolution,
//! and streaming connection patterns using fluent_ai_async.

use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;

use fluent_ai_async::{AsyncStream, emit, spawn_task};
use http::Uri;

use super::connector_core::H3Connector;
use super::types::H3Connection;
use crate::response::HttpResponseChunk;

impl H3Connector {
    /// Connect to destination URI and return response stream
    pub fn connect(&mut self, dest: Uri) -> AsyncStream<HttpResponseChunk> {
        AsyncStream::with_channel(move |sender| {
            let _task = spawn_task(|| async move {
                let host = match dest.host() {
                    Some(h) => h.trim_start_matches('[').trim_end_matches(']'),
                    None => {
                        emit!(
                            sender,
                            HttpResponseChunk::bad_chunk(
                                "destination must have a host".to_string()
                            )
                        );
                        return;
                    }
                };
                let port = dest.port_u16().unwrap_or(443);

                let addrs = if let Ok(addr) = IpAddr::from_str(host) {
                    vec![SocketAddr::new(addr, port)]
                } else {
                    vec![SocketAddr::new(
                        IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                        port,
                    )]
                };

                // Use streams-first pattern - collect the connection from the stream
                let connection_stream = Self::establish_complete_h3_connection(dest.clone());
                for conn in connection_stream {
                    emit!(sender, HttpResponseChunk::connection(conn));
                }
            });

            // Task will emit values directly via emit! macro
        })
    }

    /// Remote connection with address resolution
    pub fn remote_connect(
        self,
        addrs: Vec<SocketAddr>,
        server_name: String,
    ) -> AsyncStream<H3Connection> {
        AsyncStream::with_channel(move |sender| {
            let _task = spawn_task(move || {
                let connection_stream =
                    Self::establish_complete_h3_connection_with_retry(addrs, server_name);
                for connection in connection_stream {
                    emit!(sender, connection);
                }
            });

            // Task will emit values directly via emit! macro
        })
    }

    /// Resolve host to socket addresses
    pub fn resolve_host(host: &str, port: u16) -> Vec<SocketAddr> {
        if let Ok(addr) = IpAddr::from_str(host) {
            vec![SocketAddr::new(addr, port)]
        } else {
            // In production, this would use DNS resolution
            // For now, fallback to localhost
            vec![SocketAddr::new(
                IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                port,
            )]
        }
    }

    /// Extract host and port from URI
    pub fn extract_host_port(uri: &Uri) -> Result<(&str, u16), String> {
        let host = uri
            .host()
            .ok_or("URI must have a host")?
            .trim_start_matches('[')
            .trim_end_matches(']');
        let port = uri.port_u16().unwrap_or(443);
        Ok((host, port))
    }
}
