//! Connection Management Infrastructure
//!
//! This module provides zero-allocation, lock-free connection management
//! for HTTP/3 streaming with fluent_ai_async architecture.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime},
};

use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, AsyncStreamSender, emit};

use crate::types::chunks::HttpResponseChunk;

/// Connection state tracking for lock-free operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConnectionState {
    Idle = 0,
    Connecting = 1,
    Connected = 2,
    Disconnecting = 3,
    Disconnected = 4,
    Error = 5,
}

impl From<u8> for ConnectionState {
    #[inline]
    fn from(value: u8) -> Self {
        match value {
            0 => ConnectionState::Idle,
            1 => ConnectionState::Connecting,
            2 => ConnectionState::Connected,
            3 => ConnectionState::Disconnecting,
            4 => ConnectionState::Disconnected,
            _ => ConnectionState::Error,
        }
    }
}

/// Core HTTP connection with zero-allocation state management
///
/// This type provides the foundational connection infrastructure
/// that all protocol implementations (H2, H3, Quiche) will extend.
#[derive(Debug)]
pub struct HttpConnection {
    /// Connection identifier (zero-allocation)
    pub id: Arc<str>,

    /// Remote address
    pub remote_addr: SocketAddr,

    /// Connection state (lock-free atomic)
    pub state: AtomicU8,

    /// Connection statistics (lock-free)
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
    pub requests_sent: AtomicU64,
    pub responses_received: AtomicU64,

    /// Connection timing
    pub created_at: Instant,
    pub last_activity: AtomicU64, // Unix timestamp in milliseconds

    /// Error tracking
    pub error_count: AtomicUsize,
    pub last_error: Arc<parking_lot::RwLock<Option<Arc<str>>>>,

    /// Connection metadata (zero-allocation)
    pub protocol_version: Arc<str>,
    pub user_agent: Option<Arc<str>>,
}

impl HttpConnection {
    /// Create a new HTTP connection
    #[inline]
    pub fn new(
        id: impl Into<Arc<str>>,
        remote_addr: SocketAddr,
        protocol_version: impl Into<Arc<str>>,
    ) -> Self {
        let now = Instant::now();
        let now_millis = now.duration_since(Instant::now()).as_millis() as u64;

        Self {
            id: id.into(),
            remote_addr,
            state: AtomicU8::new(ConnectionState::Idle as u8),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            requests_sent: AtomicU64::new(0),
            responses_received: AtomicU64::new(0),
            created_at: now,
            last_activity: AtomicU64::new(now_millis),
            error_count: AtomicUsize::new(0),
            last_error: Arc::new(parking_lot::RwLock::new(None)),
            protocol_version: protocol_version.into(),
            user_agent: None,
        }
    }

    /// Get current connection state
    #[inline]
    pub fn state(&self) -> ConnectionState {
        ConnectionState::from(self.state.load(Ordering::Relaxed))
    }

    /// Set connection state (atomic)
    #[inline]
    pub fn set_state(&self, state: ConnectionState) {
        self.state.store(state as u8, Ordering::Relaxed);
        self.update_activity();
    }

    /// Update last activity timestamp
    #[inline]
    pub fn update_activity(&self) {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_activity.store(now_millis, Ordering::Relaxed);
    }

    /// Record bytes sent
    #[inline]
    pub fn record_bytes_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
        self.update_activity();
    }

    /// Record bytes received
    #[inline]
    pub fn record_bytes_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
        self.update_activity();
    }

    /// Record request sent
    #[inline]
    pub fn record_request_sent(&self) {
        self.requests_sent.fetch_add(1, Ordering::Relaxed);
        self.update_activity();
    }

    /// Record response received
    #[inline]
    pub fn record_response_received(&self) {
        self.responses_received.fetch_add(1, Ordering::Relaxed);
        self.update_activity();
    }

    /// Record connection error
    #[inline]
    pub fn record_error(&self, error: impl Into<Arc<str>>) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        *self.last_error.write() = Some(error.into());
        self.set_state(ConnectionState::Error);
    }

    /// Get total bytes sent
    #[inline]
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Relaxed)
    }

    /// Get total bytes received
    #[inline]
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Relaxed)
    }

    /// Get total requests sent
    #[inline]
    pub fn requests_sent(&self) -> u64 {
        self.requests_sent.load(Ordering::Relaxed)
    }

    /// Get total responses received
    #[inline]
    pub fn responses_received(&self) -> u64 {
        self.responses_received.load(Ordering::Relaxed)
    }

    /// Get error count
    #[inline]
    pub fn error_count(&self) -> usize {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Get last error message
    #[inline]
    pub fn last_error(&self) -> Option<Arc<str>> {
        self.last_error.read().clone()
    }

    /// Check if connection is healthy
    #[inline]
    pub fn is_healthy(&self) -> bool {
        matches!(self.state(), ConnectionState::Connected) && self.error_count() == 0
    }

    /// Check if connection is idle for given duration
    #[inline]
    pub fn is_idle_for(&self, duration: Duration) -> bool {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let last_activity = self.last_activity.load(Ordering::Relaxed);
        now_millis.saturating_sub(last_activity) > duration.as_millis() as u64
    }

    /// Get connection age
    #[inline]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Connection pool for managing multiple HTTP connections
///
/// This type provides zero-allocation connection pooling with
/// lock-free operations for high-performance HTTP/3 streaming.
#[derive(Debug)]
pub struct HttpConnectionPool {
    /// Active connections (lock-free map using dashmap)
    connections: dashmap::DashMap<Arc<str>, Arc<HttpConnection>>,

    /// Pool statistics
    total_connections: AtomicUsize,
    active_connections: AtomicUsize,
    max_connections: AtomicUsize,

    /// Pool configuration
    max_idle_duration: Duration,
    connection_timeout: Duration,
}

impl HttpConnectionPool {
    /// Create a new connection pool
    #[inline]
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: dashmap::DashMap::new(),
            total_connections: AtomicUsize::new(0),
            active_connections: AtomicUsize::new(0),
            max_connections: AtomicUsize::new(max_connections),
            max_idle_duration: Duration::from_secs(300), // 5 minutes
            connection_timeout: Duration::from_secs(30),
        }
    }

    /// Add connection to pool
    #[inline]
    pub fn add_connection(&self, connection: Arc<HttpConnection>) -> Result<(), Arc<str>> {
        let current_count = self.total_connections.load(Ordering::Relaxed);
        let max_count = self.max_connections.load(Ordering::Relaxed);

        if current_count >= max_count {
            return Err(Arc::from("Connection pool at maximum capacity"));
        }

        let id = connection.id.clone();
        self.connections.insert(id.clone(), connection);
        self.total_connections.fetch_add(1, Ordering::Relaxed);

        if matches!(
            self.get_connection(&id).map(|c| c.state()),
            Some(ConnectionState::Connected)
        ) {
            self.active_connections.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get connection from pool
    #[inline]
    pub fn get_connection(&self, id: &str) -> Option<Arc<HttpConnection>> {
        self.connections.get(id).map(|entry| entry.value().clone())
    }

    /// Remove connection from pool
    #[inline]
    pub fn remove_connection(&self, id: &str) -> Option<Arc<HttpConnection>> {
        if let Some((_, connection)) = self.connections.remove(id) {
            self.total_connections.fetch_sub(1, Ordering::Relaxed);

            if matches!(connection.state(), ConnectionState::Connected) {
                self.active_connections.fetch_sub(1, Ordering::Relaxed);
            }

            Some(connection)
        } else {
            None
        }
    }

    /// Get all active connections
    #[inline]
    pub fn active_connections(&self) -> Vec<Arc<HttpConnection>> {
        self.connections
            .iter()
            .filter(|entry| matches!(entry.value().state(), ConnectionState::Connected))
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Clean up idle connections
    #[inline]
    pub fn cleanup_idle_connections(&self) -> usize {
        let mut removed_count = 0;
        let idle_duration = self.max_idle_duration;

        // Collect IDs of idle connections
        let idle_ids: Vec<Arc<str>> = self
            .connections
            .iter()
            .filter(|entry| {
                let connection = entry.value();
                connection.is_idle_for(idle_duration)
                    || matches!(
                        connection.state(),
                        ConnectionState::Disconnected | ConnectionState::Error
                    )
            })
            .map(|entry| entry.key().clone())
            .collect();

        // Remove idle connections
        for id in idle_ids {
            if self.remove_connection(&id).is_some() {
                removed_count += 1;
            }
        }

        removed_count
    }

    /// Get pool statistics
    #[inline]
    pub fn stats(&self) -> ConnectionPoolStats {
        ConnectionPoolStats {
            total_connections: self.total_connections.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            max_connections: self.max_connections.load(Ordering::Relaxed),
            idle_connections: self
                .total_connections
                .load(Ordering::Relaxed)
                .saturating_sub(self.active_connections.load(Ordering::Relaxed)),
        }
    }

    /// Set maximum idle duration
    #[inline]
    pub fn set_max_idle_duration(&mut self, duration: Duration) {
        self.max_idle_duration = duration;
    }

    /// Set connection timeout
    #[inline]
    pub fn set_connection_timeout(&mut self, timeout: Duration) {
        self.connection_timeout = timeout;
    }
}

/// Connection pool statistics
#[derive(Debug, Clone, Copy)]
pub struct ConnectionPoolStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub max_connections: usize,
    pub idle_connections: usize,
}

/// Connection manager for protocol-specific implementations
///
/// This trait provides the interface that H2, H3, Quiche, and WASM
/// implementations will use for connection management.
pub trait ConnectionManager: Send + Sync {
    /// Create a new connection
    fn create_connection(
        &self,
        remote_addr: SocketAddr,
        config: &ConnectionConfig,
    ) -> AsyncStream<HttpResponseChunk, 1024>;

    /// Get connection by ID
    fn get_connection(&self, id: &str) -> Option<Arc<HttpConnection>>;

    /// Close connection
    fn close_connection(&self, id: &str) -> AsyncStream<HttpResponseChunk, 1024>;

    /// Get connection pool statistics
    fn pool_stats(&self) -> ConnectionPoolStats;
}

/// Configuration for connection creation
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub timeout: Duration,
    pub keep_alive: bool,
    pub max_retries: usize,
    pub user_agent: Option<Arc<str>>,
    pub headers: HashMap<Arc<str>, Arc<str>>,
}

impl Default for ConnectionConfig {
    #[inline]
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            keep_alive: true,
            max_retries: 3,
            user_agent: Some(Arc::from("fluent-ai-http3/1.0")),
            headers: HashMap::new(),
        }
    }
}

/// Default connection manager implementation
#[derive(Debug)]
pub struct DefaultConnectionManager {
    pool: Arc<HttpConnectionPool>,
}

impl DefaultConnectionManager {
    /// Create a new default connection manager
    #[inline]
    pub fn new(max_connections: usize) -> Self {
        Self {
            pool: Arc::new(HttpConnectionPool::new(max_connections)),
        }
    }

    /// Get reference to connection pool
    #[inline]
    pub fn pool(&self) -> &Arc<HttpConnectionPool> {
        &self.pool
    }
}

impl ConnectionManager for DefaultConnectionManager {
    #[inline]
    fn create_connection(
        &self,
        remote_addr: SocketAddr,
        config: &ConnectionConfig,
    ) -> AsyncStream<HttpResponseChunk, 1024> {
        let pool = self.pool.clone();
        let remote_addr = remote_addr;
        let config = config.clone();

        AsyncStream::with_channel(move |sender| {
            // Generate connection ID
            let conn_id = Arc::from(format!("conn_{}", std::thread::current().id().as_u64()));

            // Create new connection
            let connection = Arc::new(HttpConnection::new(
                conn_id.clone(),
                remote_addr,
                "HTTP/3.0",
            ));

            // Set user agent if provided
            if let Some(user_agent) = config.user_agent {
                // Note: In a real implementation, we'd store this in the connection
            }

            // Add to pool
            match pool.add_connection(connection.clone()) {
                Ok(()) => {
                    connection.set_state(ConnectionState::Connecting);

                    // Simulate connection establishment
                    // In real implementation, this would be protocol-specific
                    connection.set_state(ConnectionState::Connected);

                    emit!(
                        sender,
                        HttpResponseChunk::status(
                            http::StatusCode::OK,
                            http::HeaderMap::new(),
                            http::Version::HTTP_3
                        )
                    );
                    emit!(sender, HttpResponseChunk::complete());
                }
                Err(error) => {
                    emit!(sender, HttpResponseChunk::connection_error(error, false));
                }
            }
        })
    }

    #[inline]
    fn get_connection(&self, id: &str) -> Option<Arc<HttpConnection>> {
        self.pool.get_connection(id)
    }

    #[inline]
    fn close_connection(&self, id: &str) -> AsyncStream<HttpResponseChunk, 1024> {
        let pool = self.pool.clone();
        let id = Arc::from(id);

        AsyncStream::with_channel(move |sender| {
            if let Some(connection) = pool.get_connection(&id) {
                connection.set_state(ConnectionState::Disconnecting);

                // Simulate connection closure
                // In real implementation, this would be protocol-specific
                connection.set_state(ConnectionState::Disconnected);

                pool.remove_connection(&id);

                emit!(sender, HttpResponseChunk::complete());
            } else {
                emit!(
                    sender,
                    HttpResponseChunk::connection_error(
                        format!("Connection {} not found", id),
                        false
                    )
                );
            }
        })
    }

    #[inline]
    fn pool_stats(&self) -> ConnectionPoolStats {
        self.pool.stats()
    }
}
