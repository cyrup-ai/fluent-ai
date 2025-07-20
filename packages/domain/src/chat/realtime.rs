//! Real-time features for chat system
//!
//! This module provides comprehensive real-time features including typing indicators,
//! live message streaming, event-driven architecture, and connection management using
//! zero-allocation patterns and lock-free operations for blazing-fast performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use crossbeam_skiplist::SkipMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::{interval, sleep, timeout};
use uuid::Uuid;

use crate::message::Message;

/// Real-time event types with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealTimeEvent {
    /// User started typing
    TypingStarted {
        user_id: Arc<str>,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// User stopped typing
    TypingStopped {
        user_id: Arc<str>,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// New message received
    MessageReceived {
        message: Message,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// Message updated
    MessageUpdated {
        message_id: Arc<str>,
        content: Arc<str>,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// Message deleted
    MessageDeleted {
        message_id: Arc<str>,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// User joined session
    UserJoined {
        user_id: Arc<str>,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// User left session
    UserLeft {
        user_id: Arc<str>,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// Connection status changed
    ConnectionStatusChanged {
        user_id: Arc<str>,
        status: ConnectionStatus,
        timestamp: u64,
    },
    /// Heartbeat received
    HeartbeatReceived {
        user_id: Arc<str>,
        session_id: Arc<str>,
        timestamp: u64,
    },
    /// System notification
    SystemNotification {
        message: Arc<str>,
        level: NotificationLevel,
        timestamp: u64,
    },
}

/// Connection status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionStatus {
    /// Connected and active
    Connected,
    /// Disconnected
    Disconnected,
    /// Reconnecting
    Reconnecting,
    /// Connection failed
    Failed,
    /// Idle (connected but inactive)
    Idle,
    /// Unstable connection
    Unstable,
}

/// Notification level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationLevel {
    /// Information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Success
    Success,
}

/// Typing indicator state with atomic operations
#[derive(Debug)]
pub struct TypingState {
    /// User ID
    pub user_id: Arc<str>,
    /// Session ID
    pub session_id: Arc<str>,
    /// Last activity timestamp
    pub last_activity: AtomicU64,
    /// Is currently typing
    pub is_typing: AtomicBool,
    /// Typing duration in seconds
    pub typing_duration: AtomicU64,
}

impl TypingState {
    /// Create a new typing state
    pub fn new(user_id: Arc<str>, session_id: Arc<str>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            user_id,
            session_id,
            last_activity: AtomicU64::new(now),
            is_typing: AtomicBool::new(false),
            typing_duration: AtomicU64::new(0),
        }
    }

    /// Start typing
    pub fn start_typing(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.last_activity.store(now, Ordering::Relaxed);
        self.is_typing.store(true, Ordering::Relaxed);
    }

    /// Stop typing
    pub fn stop_typing(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let start_time = self.last_activity.load(Ordering::Relaxed);
        if start_time > 0 {
            let duration = now.saturating_sub(start_time);
            self.typing_duration.fetch_add(duration, Ordering::Relaxed);
        }

        self.last_activity.store(now, Ordering::Relaxed);
        self.is_typing.store(false, Ordering::Relaxed);
    }

    /// Check if typing has expired
    pub fn is_expired(&self, expiry_seconds: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let last_activity = self.last_activity.load(Ordering::Relaxed);
        now.saturating_sub(last_activity) > expiry_seconds
    }

    /// Get current typing status
    pub fn is_currently_typing(&self) -> bool {
        self.is_typing.load(Ordering::Relaxed)
    }

    /// Get total typing duration
    pub fn total_typing_duration(&self) -> u64 {
        self.typing_duration.load(Ordering::Relaxed)
    }
}

/// Typing indicator manager with atomic operations
pub struct TypingIndicator {
    /// Active typing states
    typing_states: SkipMap<Arc<str>, Arc<TypingState>>,
    /// Typing expiry duration in seconds
    expiry_duration: AtomicU64,
    /// Cleanup interval in seconds
    cleanup_interval: AtomicU64,
    /// Event broadcaster
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Active users counter
    active_users: ConsistentCounter,
    /// Total typing events counter
    typing_events: ConsistentCounter,
    /// Cleanup task handle
    cleanup_task: ArcSwap<Option<tokio::task::JoinHandle<()>>>,
}

impl TypingIndicator {
    /// Create a new typing indicator
    pub fn new(expiry_duration: u64, cleanup_interval: u64) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);

        Self {
            typing_states: SkipMap::new(),
            expiry_duration: AtomicU64::new(expiry_duration),
            cleanup_interval: AtomicU64::new(cleanup_interval),
            event_broadcaster,
            active_users: ConsistentCounter::new(0),
            typing_events: ConsistentCounter::new(0),
            cleanup_task: ArcSwap::new(Arc::new(None)),
        }
    }

    /// Start typing indicator
    pub fn start_typing(
        &self,
        user_id: Arc<str>,
        session_id: Arc<str>,
    ) -> Result<(), RealTimeError> {
        let key = Arc::from(format!("{}:{}", user_id, session_id));

        let typing_state = if let Some(existing) = self.typing_states.get(&key) {
            existing.value().clone()
        } else {
            let new_state = Arc::new(TypingState::new(user_id.clone(), session_id.clone()));
            self.typing_states.insert(key, new_state.clone());
            self.active_users.inc();
            new_state
        };

        typing_state.start_typing();
        self.typing_events.inc();

        // Broadcast typing started event
        let event = RealTimeEvent::TypingStarted {
            user_id,
            session_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let _ = self.event_broadcaster.send(event);

        Ok(())
    }

    /// Stop typing indicator
    pub fn stop_typing(
        &self,
        user_id: Arc<str>,
        session_id: Arc<str>,
    ) -> Result<(), RealTimeError> {
        let key = Arc::from(format!("{}:{}", user_id, session_id));

        if let Some(typing_state) = self.typing_states.get(&key) {
            typing_state.value().stop_typing();
            self.typing_events.inc();

            // Broadcast typing stopped event
            let event = RealTimeEvent::TypingStopped {
                user_id,
                session_id,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let _ = self.event_broadcaster.send(event);
        }

        Ok(())
    }

    /// Get currently typing users
    pub fn get_typing_users(&self, session_id: &str) -> Vec<Arc<str>> {
        let mut typing_users = Vec::new();

        for entry in self.typing_states.iter() {
            let typing_state = entry.value();
            if typing_state.session_id.as_ref() == session_id && typing_state.is_currently_typing()
            {
                typing_users.push(typing_state.user_id.clone());
            }
        }

        typing_users
    }

    /// Start cleanup task
    pub fn start_cleanup_task(&self) {
        let typing_states = self.typing_states.clone();
        let expiry_duration = self.expiry_duration.clone();
        let cleanup_interval = self.cleanup_interval.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let active_users = self.active_users.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(
                cleanup_interval.load(Ordering::Relaxed),
            ));

            loop {
                interval.tick().await;

                let expiry_seconds = expiry_duration.load(Ordering::Relaxed);
                let mut expired_keys = Vec::new();

                // Find expired typing states
                for entry in typing_states.iter() {
                    let typing_state = entry.value();
                    if typing_state.is_expired(expiry_seconds) {
                        expired_keys.push(entry.key().clone());

                        // Broadcast typing stopped event for expired states
                        if typing_state.is_currently_typing() {
                            let event = RealTimeEvent::TypingStopped {
                                user_id: typing_state.user_id.clone(),
                                session_id: typing_state.session_id.clone(),
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                            };

                            let _ = event_broadcaster.send(event);
                        }
                    }
                }

                // Remove expired states
                for key in expired_keys {
                    typing_states.remove(&key);
                    active_users.dec();
                }
            }
        });

        self.cleanup_task.store(Arc::new(Some(task)));
    }

    /// Subscribe to typing events
    pub fn subscribe(&self) -> broadcast::Receiver<RealTimeEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> TypingStatistics {
        TypingStatistics {
            active_users: self.active_users.get(),
            total_typing_events: self.typing_events.get(),
            expiry_duration: self.expiry_duration.load(Ordering::Relaxed),
            cleanup_interval: self.cleanup_interval.load(Ordering::Relaxed),
        }
    }
}

/// Typing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingStatistics {
    pub active_users: usize,
    pub total_typing_events: usize,
    pub expiry_duration: u64,
    pub cleanup_interval: u64,
}

/// Live update message with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveUpdateMessage {
    /// Message ID
    pub id: Arc<str>,
    /// Message content
    pub content: Arc<str>,
    /// Message type
    pub message_type: Arc<str>,
    /// Session ID
    pub session_id: Arc<str>,
    /// User ID
    pub user_id: Arc<str>,
    /// Timestamp
    pub timestamp: u64,
    /// Priority level
    pub priority: MessagePriority,
    /// Metadata
    pub metadata: Option<Arc<str>>,
}

/// Message priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Live update system with message streaming and backpressure handling
pub struct LiveUpdateSystem {
    /// Message queue for streaming
    message_queue: SegQueue<LiveUpdateMessage>,
    /// Event broadcaster for live updates
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Subscriber channels
    subscribers: Arc<RwLock<HashMap<Arc<str>, mpsc::UnboundedSender<LiveUpdateMessage>>>>,
    /// Message counter
    message_counter: ConsistentCounter,
    /// Subscriber counter
    subscriber_counter: ConsistentCounter,
    /// Queue size limit
    queue_size_limit: AtomicUsize,
    /// Backpressure threshold
    backpressure_threshold: AtomicUsize,
    /// Processing rate limiter
    rate_limiter: Arc<RwLock<tokio::time::Interval>>,
    /// System statistics
    stats: Arc<RwLock<LiveUpdateStatistics>>,
}

/// Live update statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveUpdateStatistics {
    pub total_messages: usize,
    pub active_subscribers: usize,
    pub queue_size: usize,
    pub backpressure_events: usize,
    pub processing_rate: f64,
    pub last_update: u64,
}

impl LiveUpdateSystem {
    /// Create a new live update system
    pub fn new(
        queue_size_limit: usize,
        backpressure_threshold: usize,
        processing_rate: u64,
    ) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);
        let rate_limiter = Arc::new(RwLock::new(interval(Duration::from_millis(
            1000 / processing_rate,
        ))));

        Self {
            message_queue: SegQueue::new(),
            event_broadcaster,
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            message_counter: ConsistentCounter::new(0),
            subscriber_counter: ConsistentCounter::new(0),
            queue_size_limit: AtomicUsize::new(queue_size_limit),
            backpressure_threshold: AtomicUsize::new(backpressure_threshold),
            rate_limiter,
            stats: Arc::new(RwLock::new(LiveUpdateStatistics {
                total_messages: 0,
                active_subscribers: 0,
                queue_size: 0,
                backpressure_events: 0,
                processing_rate: processing_rate as f64,
                last_update: 0,
            })),
        }
    }

    /// Send live update message
    pub async fn send_message(&self, message: LiveUpdateMessage) -> Result<(), RealTimeError> {
        let current_queue_size = self.message_counter.get();
        let queue_limit = self.queue_size_limit.load(Ordering::Relaxed);

        // Check for backpressure
        if current_queue_size >= queue_limit {
            let mut stats = self.stats.write().await;
            stats.backpressure_events += 1;
            drop(stats);

            return Err(RealTimeError::BackpressureExceeded {
                current_size: current_queue_size,
                limit: queue_limit,
            });
        }

        // Add message to queue
        self.message_queue.push(message.clone());
        self.message_counter.inc();

        // Broadcast real-time event
        let event = RealTimeEvent::MessageReceived {
            message: Message::user(message.user_id.clone(), message.content.clone()),
            session_id: message.session_id.clone(),
            timestamp: message.timestamp,
        };

        let _ = self.event_broadcaster.send(event);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_messages += 1;
        stats.queue_size = current_queue_size + 1;
        stats.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(())
    }

    /// Subscribe to live updates
    pub async fn subscribe(
        &self,
        subscriber_id: Arc<str>,
    ) -> Result<mpsc::UnboundedReceiver<LiveUpdateMessage>, RealTimeError> {
        let (tx, rx) = mpsc::unbounded_channel();

        let mut subscribers = self.subscribers.write().await;
        subscribers.insert(subscriber_id, tx);
        self.subscriber_counter.inc();

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.active_subscribers = subscribers.len();

        Ok(rx)
    }

    /// Unsubscribe from live updates
    pub async fn unsubscribe(&self, subscriber_id: &Arc<str>) -> Result<(), RealTimeError> {
        let mut subscribers = self.subscribers.write().await;
        if subscribers.remove(subscriber_id).is_some() {
            self.subscriber_counter.dec();

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.active_subscribers = subscribers.len();
        }

        Ok(())
    }

    /// Start message processing task
    pub async fn start_processing(&self) {
        let message_queue = self.message_queue.clone();
        let subscribers = self.subscribers.clone();
        let message_counter = self.message_counter.clone();
        let rate_limiter = self.rate_limiter.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            loop {
                // Rate limiting
                {
                    let mut limiter = rate_limiter.write().await;
                    limiter.tick().await;
                }

                // Process messages
                if let Some(message) = message_queue.pop() {
                    message_counter.dec();

                    let subscribers_guard = subscribers.read().await;
                    let mut failed_subscribers = Vec::new();

                    for (subscriber_id, sender) in subscribers_guard.iter() {
                        if sender.send(message.clone()).is_err() {
                            failed_subscribers.push(subscriber_id.clone());
                        }
                    }

                    drop(subscribers_guard);

                    // Remove failed subscribers
                    if !failed_subscribers.is_empty() {
                        let mut subscribers_guard = subscribers.write().await;
                        for subscriber_id in failed_subscribers {
                            subscribers_guard.remove(&subscriber_id);
                        }

                        // Update statistics
                        let mut stats_guard = stats.write().await;
                        stats_guard.active_subscribers = subscribers_guard.len();
                    }

                    // Update queue size statistics
                    let mut stats_guard = stats.write().await;
                    stats_guard.queue_size = message_counter.get();
                }

                // Small delay to prevent busy waiting
                sleep(Duration::from_millis(1)).await;
            }
        });
    }

    /// Get live update statistics
    pub async fn get_statistics(&self) -> LiveUpdateStatistics {
        self.stats.read().await.clone()
    }

    /// Subscribe to real-time events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<RealTimeEvent> {
        self.event_broadcaster.subscribe()
    }
}

/// Connection state with atomic operations
#[derive(Debug)]
pub struct ConnectionState {
    /// User ID
    pub user_id: Arc<str>,
    /// Session ID
    pub session_id: Arc<str>,
    /// Connection status
    pub status: ArcSwap<ConnectionStatus>,
    /// Last heartbeat timestamp
    pub last_heartbeat: AtomicU64,
    /// Connection established timestamp
    pub connected_at: AtomicU64,
    /// Heartbeat count
    pub heartbeat_count: AtomicU64,
    /// Reconnection attempts
    pub reconnection_attempts: AtomicU64,
    /// Is connection healthy
    pub is_healthy: AtomicBool,
}

impl ConnectionState {
    /// Create a new connection state
    pub fn new(user_id: Arc<str>, session_id: Arc<str>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            user_id,
            session_id,
            status: ArcSwap::new(Arc::new(ConnectionStatus::Connected)),
            last_heartbeat: AtomicU64::new(now),
            connected_at: AtomicU64::new(now),
            heartbeat_count: AtomicU64::new(0),
            reconnection_attempts: AtomicU64::new(0),
            is_healthy: AtomicBool::new(true),
        }
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
        self.status.store(Arc::new(status));
    }

    /// Get connection status
    pub fn get_status(&self) -> ConnectionStatus {
        (**self.status.load()).clone()
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
            is_healthy: self.is_healthy.load(Ordering::Relaxed),
        }
    }
}

/// Connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatistics {
    pub user_id: Arc<str>,
    pub session_id: Arc<str>,
    pub status: ConnectionStatus,
    pub last_heartbeat: u64,
    pub connection_duration: u64,
    pub heartbeat_count: u64,
    pub reconnection_attempts: u64,
    pub is_healthy: bool,
}

/// Connection manager with heartbeat and health monitoring
pub struct ConnectionManager {
    /// Active connections
    connections: SkipMap<Arc<str>, Arc<ConnectionState>>,
    /// Event broadcaster
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Heartbeat timeout in seconds
    heartbeat_timeout: AtomicU64,
    /// Health check interval in seconds
    health_check_interval: AtomicU64,
    /// Connection counter
    connection_counter: ConsistentCounter,
    /// Heartbeat counter
    heartbeat_counter: ConsistentCounter,
    /// Failed connection counter
    failed_connection_counter: ConsistentCounter,
    /// Health check task handle
    health_check_task: ArcSwap<Option<tokio::task::JoinHandle<()>>>,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new(heartbeat_timeout: u64, health_check_interval: u64) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);

        Self {
            connections: SkipMap::new(),
            event_broadcaster,
            heartbeat_timeout: AtomicU64::new(heartbeat_timeout),
            health_check_interval: AtomicU64::new(health_check_interval),
            connection_counter: ConsistentCounter::new(0),
            heartbeat_counter: ConsistentCounter::new(0),
            failed_connection_counter: ConsistentCounter::new(0),
            health_check_task: ArcSwap::new(Arc::new(None)),
        }
    }

    /// Add connection
    pub fn add_connection(
        &self,
        user_id: Arc<str>,
        session_id: Arc<str>,
    ) -> Result<(), RealTimeError> {
        let connection_key = Arc::from(format!("{}:{}", user_id, session_id));
        let connection_state = Arc::new(ConnectionState::new(user_id.clone(), session_id.clone()));

        self.connections.insert(connection_key, connection_state);
        self.connection_counter.inc();

        // Broadcast user joined event
        let event = RealTimeEvent::UserJoined {
            user_id,
            session_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

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
            self.connection_counter.dec();

            // Broadcast user left event
            let event = RealTimeEvent::UserLeft {
                user_id: user_id.clone(),
                session_id: session_id.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

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
                user_id: user_id.clone(),
                session_id: session_id.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let _ = self.event_broadcaster.send(event);
        }

        Ok(())
    }

    /// Start health check task
    pub fn start_health_check(&self) {
        let connections = self.connections.clone();
        let heartbeat_timeout = self.heartbeat_timeout.clone();
        let health_check_interval = self.health_check_interval.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let failed_connection_counter = self.failed_connection_counter.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(
                health_check_interval.load(Ordering::Relaxed),
            ));

            loop {
                interval.tick().await;

                let timeout_seconds = heartbeat_timeout.load(Ordering::Relaxed);
                let mut unhealthy_connections = Vec::new();

                // Check connection health
                for entry in connections.iter() {
                    let connection = entry.value();
                    if !connection.is_connection_healthy(timeout_seconds) {
                        unhealthy_connections.push(entry.key().clone());

                        // Update connection status
                        connection.set_status(ConnectionStatus::Unstable);

                        // Broadcast connection status change
                        let event = RealTimeEvent::ConnectionStatusChanged {
                            user_id: connection.user_id.clone(),
                            status: ConnectionStatus::Unstable,
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        };

                        let _ = event_broadcaster.send(event);
                    }
                }

                // Handle unhealthy connections
                for key in unhealthy_connections {
                    if let Some(connection) = connections.get(&key) {
                        let conn_state = connection.value();
                        conn_state.increment_reconnection_attempts();

                        // Set as failed after multiple attempts
                        if conn_state.reconnection_attempts.load(Ordering::Relaxed) > 3 {
                            conn_state.set_status(ConnectionStatus::Failed);
                            connections.remove(&key);
                            failed_connection_counter.inc();

                            // Broadcast connection failed event
                            let event = RealTimeEvent::ConnectionStatusChanged {
                                user_id: conn_state.user_id.clone(),
                                status: ConnectionStatus::Failed,
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                            };

                            let _ = event_broadcaster.send(event);
                        }
                    }
                }
            }
        });

        self.health_check_task.store(Arc::new(Some(task)));
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

    /// Subscribe to connection events
    pub fn subscribe(&self) -> broadcast::Receiver<RealTimeEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get manager statistics
    pub fn get_manager_statistics(&self) -> ConnectionManagerStatistics {
        ConnectionManagerStatistics {
            total_connections: self.connection_counter.get(),
            total_heartbeats: self.heartbeat_counter.get(),
            failed_connections: self.failed_connection_counter.get(),
            heartbeat_timeout: self.heartbeat_timeout.load(Ordering::Relaxed),
            health_check_interval: self.health_check_interval.load(Ordering::Relaxed),
        }
    }
}

/// Connection manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionManagerStatistics {
    pub total_connections: usize,
    pub total_heartbeats: usize,
    pub failed_connections: usize,
    pub heartbeat_timeout: u64,
    pub health_check_interval: u64,
}

/// Real-time system combining all components
pub struct RealTimeSystem {
    /// Typing indicator
    pub typing_indicator: Arc<TypingIndicator>,
    /// Live update system
    pub live_update_system: Arc<LiveUpdateSystem>,
    /// Connection manager
    pub connection_manager: Arc<ConnectionManager>,
    /// Global event broadcaster
    pub event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// System statistics
    pub statistics: Arc<RwLock<RealTimeSystemStatistics>>,
}

/// Real-time system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeSystemStatistics {
    pub typing_stats: TypingStatistics,
    pub live_update_stats: LiveUpdateStatistics,
    pub connection_stats: ConnectionManagerStatistics,
    pub total_events: usize,
    pub system_uptime: u64,
}

impl RealTimeSystem {
    /// Create a new real-time system
    pub fn new() -> Self {
        let typing_indicator = Arc::new(TypingIndicator::new(30, 10)); // 30s expiry, 10s cleanup
        let live_update_system = Arc::new(LiveUpdateSystem::new(10000, 8000, 100)); // 10k queue, 8k threshold, 100 msg/s
        let connection_manager = Arc::new(ConnectionManager::new(60, 30)); // 60s heartbeat timeout, 30s health check
        let (event_broadcaster, _) = broadcast::channel(1000);

        Self {
            typing_indicator,
            live_update_system,
            connection_manager,
            event_broadcaster,
            statistics: Arc::new(RwLock::new(RealTimeSystemStatistics {
                typing_stats: TypingStatistics {
                    active_users: 0,
                    total_typing_events: 0,
                    expiry_duration: 30,
                    cleanup_interval: 10,
                },
                live_update_stats: LiveUpdateStatistics {
                    total_messages: 0,
                    active_subscribers: 0,
                    queue_size: 0,
                    backpressure_events: 0,
                    processing_rate: 100.0,
                    last_update: 0,
                },
                connection_stats: ConnectionManagerStatistics {
                    total_connections: 0,
                    total_heartbeats: 0,
                    failed_connections: 0,
                    heartbeat_timeout: 60,
                    health_check_interval: 30,
                },
                total_events: 0,
                system_uptime: 0,
            })),
        }
    }

    /// Start all real-time services
    pub async fn start(&self) {
        // Start typing indicator cleanup
        self.typing_indicator.start_cleanup_task();

        // Start live update processing
        self.live_update_system.start_processing().await;

        // Start connection health checks
        self.connection_manager.start_health_check();

        // Start statistics update task
        self.start_statistics_update().await;
    }

    /// Start statistics update task
    async fn start_statistics_update(&self) {
        let typing_indicator = self.typing_indicator.clone();
        let live_update_system = self.live_update_system.clone();
        let connection_manager = self.connection_manager.clone();
        let statistics = self.statistics.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Update every minute

            loop {
                interval.tick().await;

                let typing_stats = typing_indicator.get_statistics();
                let live_update_stats = live_update_system.get_statistics().await;
                let connection_stats = connection_manager.get_manager_statistics();

                let mut stats = statistics.write().await;
                stats.typing_stats = typing_stats;
                stats.live_update_stats = live_update_stats;
                stats.connection_stats = connection_stats;
                stats.system_uptime += 60; // Increment uptime
            }
        });
    }

    /// Get system statistics
    pub async fn get_system_statistics(&self) -> RealTimeSystemStatistics {
        self.statistics.read().await.clone()
    }

    /// Subscribe to all real-time events
    pub fn subscribe_to_all_events(&self) -> broadcast::Receiver<RealTimeEvent> {
        self.event_broadcaster.subscribe()
    }
}

impl Default for RealTimeSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time system errors
#[derive(Debug, thiserror::Error)]
pub enum RealTimeError {
    #[error("Backpressure exceeded: current size {current_size}, limit {limit}")]
    BackpressureExceeded { current_size: usize, limit: usize },
    #[error("Connection not found: {user_id}:{session_id}")]
    ConnectionNotFound {
        user_id: Arc<str>,
        session_id: Arc<str>,
    },
    #[error("Subscription failed: {reason}")]
    SubscriptionFailed { reason: Arc<str> },
    #[error("Message delivery failed: {reason}")]
    MessageDeliveryFailed { reason: Arc<str> },
    #[error("System timeout: {operation}")]
    SystemTimeout { operation: Arc<str> },
    #[error("Invalid message format: {details}")]
    InvalidMessageFormat { details: Arc<str> },
    #[error("Rate limit exceeded: {current_rate}/{limit}")]
    RateLimitExceeded { current_rate: usize, limit: usize },
    #[error("System overload: {resource}")]
    SystemOverload { resource: Arc<str> },
}

/// Real-time system builder for ergonomic configuration
pub struct RealTimeSystemBuilder {
    typing_expiry: u64,
    typing_cleanup_interval: u64,
    queue_size_limit: usize,
    backpressure_threshold: usize,
    processing_rate: u64,
    heartbeat_timeout: u64,
    health_check_interval: u64,
}

impl RealTimeSystemBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            typing_expiry: 30,
            typing_cleanup_interval: 10,
            queue_size_limit: 10000,
            backpressure_threshold: 8000,
            processing_rate: 100,
            heartbeat_timeout: 60,
            health_check_interval: 30,
        }
    }

    /// Set typing expiry duration
    pub fn typing_expiry(mut self, seconds: u64) -> Self {
        self.typing_expiry = seconds;
        self
    }

    /// Set typing cleanup interval
    pub fn typing_cleanup_interval(mut self, seconds: u64) -> Self {
        self.typing_cleanup_interval = seconds;
        self
    }

    /// Set queue size limit
    pub fn queue_size_limit(mut self, limit: usize) -> Self {
        self.queue_size_limit = limit;
        self
    }

    /// Set backpressure threshold
    pub fn backpressure_threshold(mut self, threshold: usize) -> Self {
        self.backpressure_threshold = threshold;
        self
    }

    /// Set processing rate
    pub fn processing_rate(mut self, rate: u64) -> Self {
        self.processing_rate = rate;
        self
    }

    /// Set heartbeat timeout
    pub fn heartbeat_timeout(mut self, seconds: u64) -> Self {
        self.heartbeat_timeout = seconds;
        self
    }

    /// Set health check interval
    pub fn health_check_interval(mut self, seconds: u64) -> Self {
        self.health_check_interval = seconds;
        self
    }

    /// Build the real-time system
    pub fn build(self) -> RealTimeSystem {
        let typing_indicator = Arc::new(TypingIndicator::new(
            self.typing_expiry,
            self.typing_cleanup_interval,
        ));
        let live_update_system = Arc::new(LiveUpdateSystem::new(
            self.queue_size_limit,
            self.backpressure_threshold,
            self.processing_rate,
        ));
        let connection_manager = Arc::new(ConnectionManager::new(
            self.heartbeat_timeout,
            self.health_check_interval,
        ));
        let (event_broadcaster, _) = broadcast::channel(1000);

        RealTimeSystem {
            typing_indicator,
            live_update_system,
            connection_manager,
            event_broadcaster,
            statistics: Arc::new(RwLock::new(RealTimeSystemStatistics {
                typing_stats: TypingStatistics {
                    active_users: 0,
                    total_typing_events: 0,
                    expiry_duration: self.typing_expiry,
                    cleanup_interval: self.typing_cleanup_interval,
                },
                live_update_stats: LiveUpdateStatistics {
                    total_messages: 0,
                    active_subscribers: 0,
                    queue_size: 0,
                    backpressure_events: 0,
                    processing_rate: self.processing_rate as f64,
                    last_update: 0,
                },
                connection_stats: ConnectionManagerStatistics {
                    total_connections: 0,
                    total_heartbeats: 0,
                    failed_connections: 0,
                    heartbeat_timeout: self.heartbeat_timeout,
                    health_check_interval: self.health_check_interval,
                },
                total_events: 0,
                system_uptime: 0,
            })),
        }
    }
}

impl Default for RealTimeSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}
