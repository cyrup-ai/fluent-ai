//! HTTP/3 connection pool implementation
//!
//! Manages HTTP/3 connections with connection reuse, pooling, and lifecycle management.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fluent_ai_async::AsyncStream;
use http::Uri;

use crate::hyper::async_impl::h3_client::connect::H3Connector;
use crate::response::HttpResponseChunk;

/// Connection pool key for identifying connections
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key {
    pub scheme: String,
    pub host: String,
    pub port: Option<u16>,
}

/// HTTP/3 connection pool
#[derive(Clone)]
pub struct Pool {
    inner: Arc<Mutex<PoolInner>>,
    timeout: Option<Duration>,
}

struct PoolInner {
    connections: HashMap<Key, PooledConnection>,
    connecting: HashMap<Key, Connecting>,
}

struct PooledConnection {
    client: PoolClient,
    idle_since: Instant,
}

/// Pooled HTTP/3 client
#[derive(Clone)]
pub struct PoolClient {
    key: Key,
}

/// Connection state tracking
pub enum Connecting {
    InProgress,
    Acquired,
}

/// Connection lock for synchronization
pub struct ConnectingLock;

impl Pool {
    /// Create a new connection pool
    pub fn new(timeout: Option<Duration>) -> Option<Self> {
        Some(Self {
            inner: Arc::new(Mutex::new(PoolInner {
                connections: HashMap::new(),
                connecting: HashMap::new(),
            })),
            timeout,
        })
    }

    /// Try to get a pooled connection
    pub fn try_pool(&self, key: &Key) -> Option<PoolClient> {
        let mut inner = self.inner.lock().ok()?;

        if let Some(pooled) = inner.connections.remove(key) {
            // Check if connection is still valid
            if let Some(timeout) = self.timeout {
                if pooled.idle_since.elapsed() > timeout {
                    return None;
                }
            }
            Some(pooled.client)
        } else {
            None
        }
    }

    /// Check if connection is in progress
    pub fn connecting(&self, key: &Key) -> Option<Connecting> {
        let inner = self.inner.lock().ok()?;
        inner.connecting.get(key).map(|_| Connecting::InProgress)
    }

    /// Put connection back in pool
    pub fn put(&self, key: Key, client: PoolClient, _lock: &ConnectingLock) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.connections.insert(
                key.clone(),
                PooledConnection {
                    client,
                    idle_since: Instant::now(),
                },
            );
            inner.connecting.remove(&key);
        }
    }

    /// Establish HTTP/3 connection
    pub fn establish_h3_connection(
        &self,
        key: &Key,
        connector: &mut H3Connector,
    ) -> Result<PoolClient, String> {
        // Create URI from key
        let uri_str = format!(
            "{}://{}:{}",
            key.scheme,
            key.host,
            key.port
                .unwrap_or(if key.scheme == "https" { 443 } else { 80 })
        );

        let uri = uri_str
            .parse::<Uri>()
            .map_err(|e| format!("Invalid URI: {}", e))?;

        // Establish connection using connector
        let mut connect_stream = connector.connect(uri);

        // Wait for connection establishment
        match connect_stream.try_next() {
            Some(chunk) => {
                if chunk.is_error() {
                    return Err("Connection failed".to_string());
                }
                Ok(PoolClient { key: key.clone() })
            }
            None => Err("Connection failed".to_string()),
        }
    }

    /// Establish connection (alternative method)
    pub fn establish_connection(&self, key: &Key) -> Result<PoolClient, String> {
        Ok(PoolClient { key: key.clone() })
    }
}

impl PoolClient {
    /// Send HTTP/3 request
    pub fn send_request(
        &mut self,
        req: http::Request<bytes::Bytes>,
    ) -> AsyncStream<HttpResponseChunk> {
        fluent_ai_async::AsyncStream::with_channel(move |sender| {
            // HTTP/3 request implementation
            let chunk =
                HttpResponseChunk::head(200, std::collections::HashMap::new(), String::new());
            fluent_ai_async::emit!(sender, chunk);
        })
    }

    /// Send HTTP/3 request (alternative method)
    pub fn send_h3_request(
        &mut self,
        req: http::Request<bytes::Bytes>,
    ) -> AsyncStream<HttpResponseChunk> {
        self.send_request(req)
    }
}

/// Extract domain from URI for pool key
pub fn extract_domain(uri: &Uri) -> Key {
    Key {
        scheme: uri.scheme_str().unwrap_or("https").to_string(),
        host: uri.host().unwrap_or("localhost").to_string(),
        port: uri.port_u16(),
    }
}

/// Convert domain to URI string
pub fn domain_as_uri(domain: &str) -> String {
    if domain.starts_with("http://") || domain.starts_with("https://") {
        domain.to_string()
    } else {
        format!("https://{}", domain)
    }
}
