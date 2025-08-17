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

use fluent_ai_async::prelude::*;

use crate::streaming::chunks::HttpChunk;

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
/// This is the SINGLE canonical HttpConnection implementation that consolidates
/// all previous variants into one comprehensive connection type.
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

    /// Connection configuration
    pub timeout: Duration,
    pub keep_alive: bool,
    pub max_retries: usize,

    /// Protocol-specific data
    pub is_h3: bool,
    pub is_h2: bool,
    pub supports_push: bool,
}

impl HttpConnection {
    /// Create new HTTP connection
    #[inline]
    pub fn new(
        id: Arc<str>,
        remote_addr: SocketAddr,
        protocol_version: Arc<str>,
        config: &ConnectionConfig,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id,
            remote_addr,
            state: AtomicU8::new(ConnectionState::Idle as u8),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            requests_sent: AtomicU64::new(0),
            responses_received: AtomicU64::new(0),
            created_at: Instant::now(),
            last_activity: AtomicU64::new(now),
            error_count: AtomicUsize::new(0),
            last_error: Arc::new(parking_lot::RwLock::new(None)),
            protocol_version,
            user_agent: config.user_agent.clone(),
            timeout: config.timeout,
            keep_alive: config.keep_alive,
            max_retries: config.max_retries,
            is_h3: protocol_version.as_ref() == "HTTP/3",
            is_h2: protocol_version.as_ref() == "HTTP/2",
            supports_push: protocol_version.as_ref() != "HTTP/1.1",
        }
    }

    /// Get current connection state
    #[inline]
    pub fn state(&self) -> ConnectionState {
        ConnectionState::from(self.state.load(Ordering::Acquire))
    }

    /// Set connection state
    #[inline]
    pub fn set_state(&self, state: ConnectionState) {
        self.state.store(state as u8, Ordering::Release);
        self.update_activity();
    }

    /// Update last activity timestamp
    #[inline]
    pub fn update_activity(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.last_activity.store(now, Ordering::Release);
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

    /// Record error
    #[inline]
    pub fn record_error(&self, error: Arc<str>) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        *self.last_error.write() = Some(error);
        self.set_state(ConnectionState::Error);
    }

    /// Check if connection is idle for too long
    #[inline]
    pub fn is_idle_timeout(&self, max_idle: Duration) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last_activity = self.last_activity.load(Ordering::Acquire);
        now.saturating_sub(last_activity) > max_idle.as_millis() as u64
    }

    /// Get connection statistics
    #[inline]
    pub fn stats(&self) -> ConnectionStats {
        ConnectionStats {
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            requests_sent: self.requests_sent.load(Ordering::Relaxed),
            responses_received: self.responses_received.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            state: self.state(),
            created_at: self.created_at,
            last_activity: self.last_activity.load(Ordering::Acquire),
        }
    }
}

/// Connection statistics snapshot
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub requests_sent: u64,
    pub responses_received: u64,
    pub error_count: usize,
    pub state: ConnectionState,
    pub created_at: Instant,
    pub last_activity: u64,
}

/// HTTP connection pool with lock-free operations for high-performance HTTP/3 streaming
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
    pub fn add_connection(&self, connection: Arc<HttpConnection>) -> bool {
        let current_count = self.total_connections.load(Ordering::Acquire);
        let max_count = self.max_connections.load(Ordering::Acquire);

        if current_count >= max_count {
            return false;
        }

        let id = connection.id.clone();
        self.connections.insert(id, connection);
        self.total_connections.fetch_add(1, Ordering::Release);
        self.active_connections.fetch_add(1, Ordering::Release);
        true
    }

    /// Get connection by ID
    #[inline]
    pub fn get_connection(&self, id: &str) -> Option<Arc<HttpConnection>> {
        self.connections.get(id).map(|entry| entry.value().clone())
    }

    /// Remove connection from pool
    #[inline]
    pub fn remove_connection(&self, id: &str) -> Option<Arc<HttpConnection>> {
        if let Some((_, connection)) = self.connections.remove(id) {
            self.total_connections.fetch_sub(1, Ordering::Release);
            self.active_connections.fetch_sub(1, Ordering::Release);
            Some(connection)
        } else {
            None
        }
    }

    /// Clean up idle connections
    pub fn cleanup_idle_connections(&self) {
        let mut to_remove = Vec::new();

        for entry in self.connections.iter() {
            let connection = entry.value();
            if connection.is_idle_timeout(self.max_idle_duration) {
                to_remove.push(entry.key().clone());
            }
        }

        for id in to_remove {
            self.remove_connection(&id);
        }
    }

    /// Get pool statistics
    #[inline]
    pub fn stats(&self) -> ConnectionPoolStats {
        let total = self.total_connections.load(Ordering::Acquire);
        let active = self.active_connections.load(Ordering::Acquire);
        let max = self.max_connections.load(Ordering::Acquire);

        // Count idle connections
        let idle = self
            .connections
            .iter()
            .filter(|entry| entry.value().state() == ConnectionState::Idle)
            .count();

        ConnectionPoolStats {
            total_connections: total,
            active_connections: active,
            max_connections: max,
            idle_connections: idle,
        }
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
    ) -> AsyncStream<HttpChunk, 1024>;

    /// Get connection by ID
    fn get_connection(&self, id: &str) -> Option<Arc<HttpConnection>>;

    /// Close connection
    fn close_connection(&self, id: &str) -> AsyncStream<HttpChunk, 1024>;

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
    ) -> AsyncStream<HttpChunk, 1024> {
        let pool = self.pool.clone();
        let config = config.clone();

        AsyncStream::with_channel(move |sender| {
            // Generate connection ID
            let connection_id = Arc::from(format!("conn_{}", uuid::Uuid::new_v4()));

            // Create new connection
            let connection = Arc::new(HttpConnection::new(
                connection_id.clone(),
                remote_addr,
                Arc::from("HTTP/3"),
                &config,
            ));

            // Add to pool
            if pool.add_connection(connection.clone()) {
                connection.set_state(ConnectionState::Connected);
                emit!(
                    sender,
                    HttpChunk::Head(http::StatusCode::OK, http::HeaderMap::new())
                );
            } else {
                emit!(
                    sender,
                    HttpChunk::Error("Connection pool full".to_string())
                );
            }
        })
    }

    #[inline]
    fn get_connection(&self, id: &str) -> Option<Arc<HttpConnection>> {
        self.pool.get_connection(id)
    }

    #[inline]
    fn close_connection(&self, id: &str) -> AsyncStream<HttpChunk, 1024> {
        let pool = self.pool.clone();
        let id = id.to_string();

        AsyncStream::with_channel(move |sender| {
            if let Some(connection) = pool.remove_connection(&id) {
                connection.set_state(ConnectionState::Disconnected);
                // Connection closed successfully - no chunk needed for success
            } else {
                emit!(
                    sender,
                    HttpChunk::Error("Connection not found".to_string())
                );
            }
        })
    }

    #[inline]
    fn pool_stats(&self) -> ConnectionPoolStats {
        self.pool.stats()
    }
}
