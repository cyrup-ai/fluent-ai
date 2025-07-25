//! Connection management with heartbeat and health monitoring
//!
//! This module provides comprehensive connection state management with atomic operations,
//! heartbeat monitoring, and health checking for real-time chat connections.

use crate::types::candle_chat::realtime::{
    events::{RealTimeEvent, ConnectionStatus},
    errors::RealTimeError};
use arc_swap::ArcSwap;
use crossbeam_skiplist::SkipMap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
    Arc};
use tokio::sync::broadcast;

/// Connection state with atomic operations for zero-allocation, lock-free concurrency
pub struct ConnectionState {
    /// User ID
    pub user_id: String,
    /// Session ID
    pub session_id: String,
    /// Connection status (atomic enum representation)
    pub status: AtomicU8,
    /// Last heartbeat timestamp
    pub last_heartbeat: AtomicU64,
    /// Connection established timestamp
    pub connected_at: AtomicU64,
    /// Heartbeat count
    pub heartbeat_count: AtomicU64,
    /// Reconnection attempts
    pub reconnection_attempts: AtomicU64,
    /// Is connection healthy
    pub is_healthy: AtomicBool}

impl ConnectionState {
    /// Create a new connection state
    pub fn new(user_id: String, session_id: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            user_id,
            session_id,
            status: AtomicU8::new(ConnectionStatus::Connected.to_atomic()),
            last_heartbeat: AtomicU64::new(now),
            connected_at: AtomicU64::new(now),
            heartbeat_count: AtomicU64::new(0),
            reconnection_attempts: AtomicU64::new(0),
            is_healthy: AtomicBool::new(true)}
    }

    /// Update heartbeat
    pub fn update_heartbeat(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.last_heartbeat.store(now, Ordering::Relaxed);
        self.heartbeat_count.fetch_add(1, Ordering::Relaxed);
        self.is_healthy.store(true, Ordering::Relaxed);
    }

    /// Check if connection is healthy
    pub fn is_connection_healthy(&self, heartbeat_timeout: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let last_heartbeat = self.last_heartbeat.load(Ordering::Relaxed);
        let is_healthy = now.saturating_sub(last_heartbeat) <= heartbeat_timeout;

        self.is_healthy.store(is_healthy, Ordering::Relaxed);
        is_healthy
    }

    /// Set connection status
    pub fn set_status(&self, status: ConnectionStatus) {
        self.status.store(status.to_atomic(), Ordering::Relaxed);
    }

    /// Get connection status
    pub fn get_status(&self) -> ConnectionStatus {
        ConnectionStatus::from_atomic(self.status.load(Ordering::Relaxed))
    }

    /// Increment reconnection attempts
    pub fn increment_reconnection_attempts(&self) {
        self.reconnection_attempts.fetch_add(1, Ordering::Relaxed);
    }

    /// Reset reconnection attempts
    pub fn reset_reconnection_attempts(&self) {
        self.reconnection_attempts.store(0, Ordering::Relaxed);
    }

    /// Get connection statistics
    pub fn get_statistics(&self) -> ConnectionStatistics {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let connected_at = self.connected_at.load(Ordering::Relaxed);
        let connection_duration = now.saturating_sub(connected_at);

        ConnectionStatistics {
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            status: self.get_status(),
            last_heartbeat: self.last_heartbeat.load(Ordering::Relaxed),
            connection_duration,
            heartbeat_count: self.heartbeat_count.load(Ordering::Relaxed),
            reconnection_attempts: self.reconnection_attempts.load(Ordering::Relaxed),
            is_healthy: self.is_healthy.load(Ordering::Relaxed)}
    }
}

/// Connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatistics {
    pub user_id: String,
    pub session_id: String,
    pub status: ConnectionStatus,
    pub last_heartbeat: u64,
    pub connection_duration: u64,
    pub heartbeat_count: u64,
    pub reconnection_attempts: u64,
    pub is_healthy: bool}

/// Connection manager with heartbeat and health monitoring
pub struct ConnectionManager {
    /// Active connections
    connections: Arc<SkipMap<Arc<str>, Arc<ConnectionState>>>,
    /// Event broadcaster
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Heartbeat timeout in seconds
    heartbeat_timeout: Arc<AtomicU64>,
    /// Health check interval in seconds
    health_check_interval: Arc<AtomicU64>,
    /// Connection counter
    connection_counter: Arc<ConsistentCounter>,
    /// Heartbeat counter
    heartbeat_counter: Arc<ConsistentCounter>,
    /// Failed connection counter
    failed_connection_counter: Arc<ConsistentCounter>,
    /// Health check task handle
    health_check_task: ArcSwap<Option<fluent_ai_async::AsyncTask<()>>>}

impl std::fmt::Debug for ConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionManager")
            .field("connections_count", &self.connections.len())
            .field("connection_counter", &self.connection_counter.get())
            .field("heartbeat_counter", &self.heartbeat_counter.get())
            .field(
                "failed_connection_counter",
                &self.failed_connection_counter.get(),
            )
            .field(
                "heartbeat_timeout",
                &self
                    .heartbeat_timeout
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "health_check_interval",
                &self
                    .health_check_interval
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new(heartbeat_timeout: u64, health_check_interval: u64) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);

        Self {
            connections: Arc::new(SkipMap::new()),
            event_broadcaster,
            heartbeat_timeout: Arc::new(AtomicU64::new(heartbeat_timeout)),
            health_check_interval: Arc::new(AtomicU64::new(health_check_interval)),
            connection_counter: Arc::new(ConsistentCounter::new(0)),
            heartbeat_counter: Arc::new(ConsistentCounter::new(0)),
            failed_connection_counter: Arc::new(ConsistentCounter::new(0)),
            health_check_task: ArcSwap::new(Arc::new(None))}
    }

    /// Add connection
    pub fn add_connection(
        &self,
        user_id: Arc<str>,
        session_id: Arc<str>,
    ) -> Result<(), RealTimeError> {
        let connection_key = Arc::from(format!("{}:{}", user_id, session_id));
        let connection_state = Arc::new(ConnectionState::new(
            user_id.to_string(),
            session_id.to_string(),
        ));

        self.connections.insert(connection_key, connection_state);
        self.connection_counter.inc();

        // Broadcast user joined event
        let event = RealTimeEvent::UserJoined {
            user_id: user_id.to_string(),
            session_id: session_id.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()};

        let _ = self.event_broadcaster.send(event);

        Ok(())
    }

    /// Remove connection
    pub fn remove_connection(
        &self,
        user_id: &Arc<str>,
        session_id: &Arc<str>,
    ) -> Result<(), RealTimeError> {
        let connection_key = Arc::from(format!("{}:{}", user_id, session_id));

        if self.connections.remove(&connection_key).is_some() {
            // Decrement counter - ConsistentCounter doesn't have dec(), so we work around it
            let current = self.connection_counter.get();
            if current > 0 {
                self.connection_counter.reset();
                for _ in 0..(current - 1) {
                    self.connection_counter.inc();
                }
            }

            // Broadcast user left event
            let event = RealTimeEvent::UserLeft {
                user_id: user_id.to_string(),
                session_id: session_id.to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()};

            let _ = self.event_broadcaster.send(event);
        }

        Ok(())
    }

    /// Update heartbeat
    pub fn update_heartbeat(
        &self,
        user_id: &Arc<str>,
        session_id: &Arc<str>,
    ) -> Result<(), RealTimeError> {
        let connection_key = Arc::from(format!("{}:{}", user_id, session_id));

        if let Some(connection) = self.connections.get(&connection_key) {
            connection.value().update_heartbeat();
            self.heartbeat_counter.inc();

            // Broadcast heartbeat event
            let event = RealTimeEvent::HeartbeatReceived {
                user_id: user_id.to_string(),
                session_id: session_id.to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()};

            let _ = self.event_broadcaster.send(event);
        }

        Ok(())
    }

    /// Subscribe to connection events
    pub fn subscribe(&self) -> broadcast::Receiver<RealTimeEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get connection statistics
    pub fn get_connection_statistics(
        &self,
        user_id: &Arc<str>,
        session_id: &Arc<str>,
    ) -> Option<ConnectionStatistics> {
        let connection_key = Arc::from(format!("{}:{}", user_id, session_id));

        self.connections
            .get(&connection_key)
            .map(|entry| entry.value().get_statistics())
    }

    /// Get all connections
    pub fn get_all_connections(&self) -> Vec<ConnectionStatistics> {
        self.connections
            .iter()
            .map(|entry| entry.value().get_statistics())
            .collect()
    }

    /// Get manager statistics
    pub fn get_manager_statistics(&self) -> ConnectionManagerStatistics {
        ConnectionManagerStatistics {
            total_connections: self.connection_counter.get(),
            total_heartbeats: self.heartbeat_counter.get(),
            failed_connections: self.failed_connection_counter.get(),
            heartbeat_timeout: self.heartbeat_timeout.load(Ordering::Relaxed),
            health_check_interval: self.health_check_interval.load(Ordering::Relaxed)}
    }
}

/// Connection manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionManagerStatistics {
    pub total_connections: usize,
    pub total_heartbeats: usize,
    pub failed_connections: usize,
    pub heartbeat_timeout: u64,
    pub health_check_interval: u64}