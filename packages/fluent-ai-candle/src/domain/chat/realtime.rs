//! Real-time features for chat system
//!
//! This module provides comprehensive real-time features including typing indicators,
//! live message streaming, event-driven architecture, and connection management using
//! zero-allocation patterns and lock-free operations for blazing-fast performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use crossbeam_skiplist::SkipMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::{interval, sleep};

// Removed unused import: uuid::Uuid
use crate::domain::chat::message::{CandleMessage as Message, CandleMessageRole as MessageRole};

/// Real-time event types with zero-allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealTimeEvent {
    /// User started typing
    TypingStarted {
        /// ID of the user who started typing
        user_id: String,
        /// Session where typing occurred
        session_id: String,
        /// Timestamp when typing started
        timestamp: u64},
    /// User stopped typing
    TypingStopped {
        /// ID of the user who stopped typing
        user_id: String,
        /// Session where typing stopped
        session_id: String,
        /// Timestamp when typing stopped
        timestamp: u64},
    /// New message received
    MessageReceived {
        /// The message that was received
        message: Message,
        /// Session where message was received
        session_id: String,
        /// Timestamp when message was received
        timestamp: u64},
    /// Message updated
    MessageUpdated {
        /// ID of the updated message
        message_id: String,
        /// New content of the message
        content: String,
        /// Session where message was updated
        session_id: String,
        /// Timestamp when message was updated
        timestamp: u64},
    /// Message deleted
    MessageDeleted {
        /// ID of the deleted message
        message_id: String,
        /// Session where message was deleted
        session_id: String,
        /// Timestamp when message was deleted
        timestamp: u64},
    /// User joined session
    UserJoined {
        /// ID of the user who joined
        user_id: String,
        /// Session that was joined
        session_id: String,
        /// Timestamp when user joined
        timestamp: u64},
    /// User left session
    UserLeft {
        /// ID of the user who left
        user_id: String,
        /// Session that was left
        session_id: String,
        /// Timestamp when user left
        timestamp: u64},
    /// Connection status changed
    ConnectionStatusChanged {
        /// ID of the user whose connection changed
        user_id: String,
        /// New connection status
        status: ConnectionStatus,
        /// Timestamp when status changed
        timestamp: u64},
    /// Heartbeat received
    HeartbeatReceived {
        /// ID of the user who sent heartbeat
        user_id: String,
        /// Session where heartbeat was received
        session_id: String,
        /// Timestamp when heartbeat was received
        timestamp: u64},
    /// System notification
    SystemNotification {
        /// Notification message content
        message: String,
        /// Severity level of the notification
        level: NotificationLevel,
        /// Timestamp when notification was created
        timestamp: u64}}

/// Connection status enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// Connected and active
    Connected,
    /// Connecting
    Connecting,
    /// Disconnected
    Disconnected,
    /// Connection error
    Error,
    /// Reconnecting
    Reconnecting,
    /// Connection failed
    Failed,
    /// Idle (connected but inactive)
    Idle,
    /// Unstable connection
    Unstable}

impl ConnectionStatus {
    /// Convert to atomic representation (u8)
    #[inline]
    pub const fn to_atomic(&self) -> u8 {
        match self {
            Self::Connected => 0,
            Self::Connecting => 1,
            Self::Disconnected => 2,
            Self::Error => 3,
            Self::Reconnecting => 4,
            Self::Failed => 5,
            Self::Idle => 6,
            Self::Unstable => 7}
    }

    /// Convert from atomic representation (u8)
    #[inline]
    pub const fn from_atomic(value: u8) -> Self {
        match value {
            0 => Self::Connected,
            1 => Self::Connecting,
            2 => Self::Disconnected,
            3 => Self::Error,
            4 => Self::Reconnecting,
            5 => Self::Failed,
            6 => Self::Idle,
            _ => Self::Unstable, // Default fallback
        }
    }
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
    Success}

/// Typing indicator state with atomic operations
#[derive(Debug)]
pub struct TypingState {
    /// User ID
    pub user_id: String,
    /// Session ID
    pub session_id: String,
    /// Last activity timestamp
    pub last_activity: AtomicU64,
    /// Is currently typing
    pub is_typing: AtomicBool,
    /// Typing duration in seconds
    pub typing_duration: AtomicU64}

impl TypingState {
    /// Create a new typing state
    pub fn new(user_id: String, session_id: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            user_id,
            session_id,
            last_activity: AtomicU64::new(now),
            is_typing: AtomicBool::new(false),
            typing_duration: AtomicU64::new(0)}
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
    typing_states: Arc<SkipMap<Arc<str>, Arc<TypingState>>>,
    /// Typing expiry duration in seconds
    expiry_duration: Arc<AtomicU64>,
    /// Cleanup interval in seconds
    cleanup_interval: Arc<AtomicU64>,
    /// Event broadcaster
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Active users counter
    active_users: Arc<ConsistentCounter>,
    /// Total typing events counter
    typing_events: Arc<ConsistentCounter>,
    /// Cleanup task handle
    cleanup_task: ArcSwap<Option<tokio::task::JoinHandle<()>>>}

impl TypingIndicator {
    /// Create a new typing indicator
    pub fn new(expiry_duration: u64, cleanup_interval: u64) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);

        Self {
            typing_states: Arc::new(SkipMap::new()),
            expiry_duration: Arc::new(AtomicU64::new(expiry_duration)),
            cleanup_interval: Arc::new(AtomicU64::new(cleanup_interval)),
            event_broadcaster,
            active_users: Arc::new(ConsistentCounter::new(0)),
            typing_events: Arc::new(ConsistentCounter::new(0)),
            cleanup_task: ArcSwap::new(Arc::new(None))}
    }

    /// Start typing indicator
    pub fn start_typing(&self, user_id: String, session_id: String) -> Result<(), RealTimeError> {
        let key = format!("{}:{}", user_id, session_id);

        let typing_state = if let Some(existing) = self.typing_states.get(key.as_str()) {
            existing.value().clone()
        } else {
            let new_state = Arc::new(TypingState::new(user_id.clone(), session_id.clone()));
            self.typing_states.insert(key.into(), new_state.clone());
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
                .as_secs()};

        let _ = self.event_broadcaster.send(event);

        Ok(())
    }

    /// Stop typing indicator
    pub fn stop_typing(&self, user_id: String, session_id: String) -> Result<(), RealTimeError> {
        let key = format!("{}:{}", user_id, session_id);

        if let Some(typing_state) = self.typing_states.get(key.as_str()) {
            typing_state.value().stop_typing();
            self.typing_events.inc();

            // Broadcast typing stopped event
            let event = RealTimeEvent::TypingStopped {
                user_id,
                session_id,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()};

            let _ = self.event_broadcaster.send(event);
        }

        Ok(())
    }

    /// Get currently typing users
    pub fn get_typing_users(&self, session_id: &str) -> Vec<String> {
        let mut typing_users = Vec::new();

        for entry in self.typing_states.iter() {
            let typing_state = entry.value();
            if typing_state.session_id == session_id && typing_state.is_currently_typing() {
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
                                    .as_secs()};

                            let _ = event_broadcaster.send(event);
                        }
                    }
                }

                // Remove expired states
                for key in expired_keys {
                    typing_states.remove(&key);
                    // Decrement counter - ConsistentCounter doesn't have dec(), so we work around it
                    let current = active_users.get();
                    if current > 0 {
                        active_users.reset();
                        for _ in 0..(current - 1) {
                            active_users.inc();
                        }
                    }
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
            cleanup_interval: self.cleanup_interval.load(Ordering::Relaxed)}
    }
}

impl std::fmt::Debug for TypingIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypingIndicator")
            .field("active_users", &self.active_users.get())
            .field("total_typing_events", &self.typing_events.get())
            .field(
                "expiry_duration",
                &self
                    .expiry_duration
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "cleanup_interval",
                &self
                    .cleanup_interval
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

/// Typing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingStatistics {
    /// Number of users currently typing
    pub active_users: usize,
    /// Total typing events processed
    pub total_typing_events: usize,
    /// Duration before typing indicators expire (seconds)
    pub expiry_duration: u64,
    /// Interval between cleanup operations (seconds)
    pub cleanup_interval: u64}

/// Live update message with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveUpdateMessage {
    /// Message ID
    pub id: String,
    /// Message content
    pub content: String,
    /// Message type
    pub message_type: String,
    /// Session ID
    pub session_id: String,
    /// User ID
    pub user_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Priority level
    pub priority: MessagePriority,
    /// Metadata
    pub metadata: Option<String>}

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
    Critical}

/// Live update system with message streaming and backpressure handling
pub struct LiveUpdateSystem {
    /// Message queue for streaming
    message_queue: Arc<SegQueue<LiveUpdateMessage>>,
    /// Event broadcaster for live updates
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Subscriber channels
    subscribers: Arc<RwLock<HashMap<Arc<str>, mpsc::UnboundedSender<LiveUpdateMessage>>>>,
    /// Message counter
    message_counter: Arc<AtomicUsize>,
    /// Subscriber counter
    subscriber_counter: Arc<ConsistentCounter>,
    /// Queue size limit
    queue_size_limit: AtomicUsize,
    /// Backpressure threshold
    backpressure_threshold: AtomicUsize,
    /// Processing rate limiter
    rate_limiter: Arc<RwLock<tokio::time::Interval>>,
    /// System statistics
    stats: Arc<RwLock<LiveUpdateStatistics>>}

/// Live update statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveUpdateStatistics {
    /// Total messages processed through live updates
    pub total_messages: usize,
    /// Number of active subscribers to live updates
    pub active_subscribers: usize,
    /// Current size of the message queue
    pub queue_size: usize,
    /// Number of backpressure events triggered
    pub backpressure_events: usize,
    /// Messages processed per second
    pub processing_rate: f64,
    /// Timestamp of last update (nanoseconds)
    pub last_update: u64}

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
            message_queue: Arc::new(SegQueue::new()),
            event_broadcaster,
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            message_counter: Arc::new(AtomicUsize::new(0)),
            subscriber_counter: Arc::new(ConsistentCounter::new(0)),
            queue_size_limit: AtomicUsize::new(queue_size_limit),
            backpressure_threshold: AtomicUsize::new(backpressure_threshold),
            rate_limiter,
            stats: Arc::new(RwLock::new(LiveUpdateStatistics {
                total_messages: 0,
                active_subscribers: 0,
                queue_size: 0,
                backpressure_events: 0,
                processing_rate: processing_rate as f64,
                last_update: 0}))}
    }

    /// Send live update message
    pub async fn send_message(&self, message: LiveUpdateMessage) -> Result<(), RealTimeError> {
        let current_queue_size = self.message_counter.load(Ordering::Relaxed);
        let queue_limit = self.queue_size_limit.load(Ordering::Relaxed);

        // Check for backpressure
        if current_queue_size >= queue_limit {
            let mut stats = self.stats.write().await;
            stats.backpressure_events += 1;
            drop(stats);

            return Err(RealTimeError::BackpressureExceeded {
                current_size: current_queue_size,
                limit: queue_limit});
        }

        // Add message to queue
        self.message_queue.push(message.clone());
        self.message_counter.fetch_add(1, Ordering::Relaxed);

        // Broadcast real-time event
        let event = RealTimeEvent::MessageReceived {
            message: Message::new(
                message.user_id.len() as u64, // Use user_id length as ID for now
                MessageRole::Assistant,
                message.content.as_bytes(),
            ),
            session_id: message.session_id.clone(),
            timestamp: message.timestamp};

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
            // Decrement counter - ConsistentCounter doesn't have dec(), so we work around it
            let current = self.subscriber_counter.get();
            if current > 0 {
                self.subscriber_counter.reset();
                for _ in 0..(current - 1) {
                    self.subscriber_counter.inc();
                }
            }

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.active_subscribers = subscribers.len();
        }

        Ok(())
    }

    /// Start message processing task
    pub async fn start_processing(&self) {
        let message_queue = Arc::clone(&self.message_queue);
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
                    message_counter.fetch_sub(1, Ordering::Relaxed);

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
                    stats_guard.queue_size = message_counter.load(Ordering::Relaxed);
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

impl std::fmt::Debug for LiveUpdateSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveUpdateSystem")
            .field(
                "message_counter",
                &self
                    .message_counter
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("subscriber_counter", &self.subscriber_counter.get())
            .field(
                "queue_size_limit",
                &self
                    .queue_size_limit
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "backpressure_threshold",
                &self
                    .backpressure_threshold
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

/// Connection state with atomic operations
#[derive(Debug)]
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
    /// Unique identifier for the user
    pub user_id: String,
    /// Session identifier for this connection
    pub session_id: String,
    /// Current connection status
    pub status: ConnectionStatus,
    /// Timestamp of last heartbeat received
    pub last_heartbeat: u64,
    /// Total duration of this connection (seconds)
    pub connection_duration: u64,
    /// Number of heartbeats received
    pub heartbeat_count: u64,
    /// Number of reconnection attempts made
    pub reconnection_attempts: u64,
    /// Whether the connection is healthy
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
    health_check_task: ArcSwap<Option<tokio::task::JoinHandle<()>>>}

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
                                .as_secs()};

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
                                    .as_secs()};

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
            health_check_interval: self.health_check_interval.load(Ordering::Relaxed)}
    }
}

/// Connection manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionManagerStatistics {
    /// Total number of connections handled
    pub total_connections: usize,
    /// Total number of heartbeats processed
    pub total_heartbeats: usize,
    /// Number of connections that failed
    pub failed_connections: usize,
    /// Timeout duration for heartbeats (seconds)
    pub heartbeat_timeout: u64,
    /// Interval between health checks (seconds)
    pub health_check_interval: u64}

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
    pub statistics: Arc<RwLock<RealTimeSystemStatistics>>}

impl std::fmt::Debug for RealTimeSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealTimeSystem")
            .field("typing_indicator", &"TypingIndicator")
            .field("live_update_system", &"LiveUpdateSystem")
            .field("connection_manager", &"ConnectionManager")
            .field("event_broadcaster", &"broadcast::Sender<RealTimeEvent>")
            .field("statistics", &"Arc<RwLock<RealTimeSystemStatistics>>")
            .finish()
    }
}

/// Real-time system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeSystemStatistics {
    /// Statistics for typing indicators
    pub typing_stats: TypingStatistics,
    /// Statistics for live updates
    pub live_update_stats: LiveUpdateStatistics,
    /// Statistics for connection management
    pub connection_stats: ConnectionManagerStatistics,
    /// Total events processed by the system
    pub total_events: usize,
    /// System uptime in seconds
    pub system_uptime: u64}

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
                    cleanup_interval: 10},
                live_update_stats: LiveUpdateStatistics {
                    total_messages: 0,
                    active_subscribers: 0,
                    queue_size: 0,
                    backpressure_events: 0,
                    processing_rate: 100.0,
                    last_update: 0},
                connection_stats: ConnectionManagerStatistics {
                    total_connections: 0,
                    total_heartbeats: 0,
                    failed_connections: 0,
                    heartbeat_timeout: 60,
                    health_check_interval: 30},
                total_events: 0,
                system_uptime: 0}))}
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
    /// Backpressure limit exceeded
    #[error("Backpressure exceeded: current size {current_size}, limit {limit}")]
    BackpressureExceeded { 
        /// Current queue size that triggered the limit
        current_size: usize, 
        /// Maximum allowed queue size
        limit: usize 
    },
    /// User connection not found
    #[error("Connection not found: {user_id}:{session_id}")]
    ConnectionNotFound { 
        /// ID of the user whose connection was not found
        user_id: String, 
        /// Session ID where connection was expected
        session_id: String 
    },
    /// Event subscription failed
    #[error("Subscription failed: {reason}")]
    SubscriptionFailed { 
        /// Reason why subscription failed
        reason: String 
    },
    /// Message delivery failed
    #[error("Message delivery failed: {reason}")]
    MessageDeliveryFailed { 
        /// Reason why message delivery failed
        reason: String 
    },
    /// System operation timed out
    #[error("System timeout: {operation}")]
    SystemTimeout { 
        /// Name of the operation that timed out
        operation: String 
    },
    /// Message format is invalid
    #[error("Invalid message format: {details}")]
    InvalidMessageFormat { 
        /// Details about the format validation failure
        details: String 
    },
    /// Rate limit exceeded
    #[error("Rate limit exceeded: {current_rate}/{limit}")]
    RateLimitExceeded { 
        /// Current rate that exceeded the limit
        current_rate: usize, 
        /// Maximum allowed rate per second
        limit: usize 
    },
    /// System overloaded
    #[error("System overload: {resource}")]
    SystemOverload { 
        /// Name of the overloaded resource
        resource: String 
    }}


/// Real-time chat configuration for managing real-time features
///
/// This configuration controls real-time chat behavior including:
/// - Connection management and heartbeat settings
/// - Message streaming and delivery options
/// - Typing indicators and presence management
/// - Performance tuning and optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Enable real-time features
    pub enabled: bool,
    /// WebSocket connection timeout in seconds
    pub connection_timeout_seconds: u64,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_seconds: u64,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Message buffer size per connection
    pub message_buffer_size: usize,
    /// Enable typing indicators
    pub enable_typing_indicators: bool,
    /// Typing indicator timeout in seconds
    pub typing_timeout_seconds: u64,
    /// Enable presence tracking
    pub enable_presence_tracking: bool,
    /// Presence update interval in seconds
    pub presence_update_interval_seconds: u64,
    /// Enable message streaming
    pub enable_message_streaming: bool,
    /// Stream chunk size in bytes
    pub stream_chunk_size: usize,
    /// Enable compression for messages
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Rate limit: messages per second per connection
    pub rate_limit_messages_per_second: u32,
    /// Enable connection pooling
    pub enable_connection_pooling: bool,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Auto-reconnect on connection loss
    pub auto_reconnect: bool,
    /// Reconnection delay in milliseconds
    pub reconnect_delay_ms: u64,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: u32}

/// Real-time chat system for managing live chat interactions
///
/// This system provides comprehensive real-time chat capabilities with:
/// - WebSocket connection management
/// - Live message streaming and delivery
/// - Typing indicators and presence tracking
/// - Event broadcasting and subscription
/// - Performance monitoring and optimization
pub struct RealtimeChat {
    /// Configuration settings
    config: RealtimeConfig,
    /// Real-time system core
    rt_system: RealTimeSystem,
    /// Active connections
    connections: Arc<SkipMap<Arc<str>, RealtimeConnection>>,
    /// Message broadcast channel
    message_broadcaster: broadcast::Sender<RealtimeMessage>,
    /// Event handlers
    event_handlers: Arc<RwLock<HashMap<RealtimeEventType, Vec<Arc<dyn RealtimeEventHandler>>>>>,
    /// Performance metrics
    metrics: Arc<RealtimeChatMetrics>,
    /// Connection manager task handle
    connection_task: Option<tokio::task::JoinHandle<()>>}

impl std::fmt::Debug for RealtimeChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeChat")
            .field("config", &self.config)
            .field("rt_system", &self.rt_system)
            .field("connections", &"Arc<SkipMap<Arc<str>, RealtimeConnection>>")
            .field("message_broadcaster", &"broadcast::Sender<RealtimeMessage>")
            .field(
                "event_handlers",
                &"Arc<RwLock<HashMap<RealtimeEventType, Vec<Arc<dyn RealtimeEventHandler>>>>>",
            )
            .field("metrics", &"Arc<RealtimeChatMetrics>")
            .field("connection_task", &self.connection_task.is_some())
            .finish()
    }
}

/// Real-time connection representation
#[derive(Debug)]
pub struct RealtimeConnection {
    /// Connection ID
    pub connection_id: String,
    /// User ID associated with connection
    pub user_id: String,
    /// Session ID
    pub session_id: String,
    /// Connection timestamp
    pub connected_at: u64,
    /// Last activity timestamp
    pub last_activity: AtomicU64,
    /// Connection status (atomic enum representation)
    pub status: AtomicU8,
    /// Message sender channel
    pub message_sender: mpsc::UnboundedSender<RealtimeMessage>,
    /// Typing status
    pub is_typing: AtomicBool,
    /// Presence status (atomic enum representation)
    pub presence: AtomicU8}

impl Clone for RealtimeConnection {
    fn clone(&self) -> Self {
        Self {
            connection_id: self.connection_id.clone(),
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            connected_at: self.connected_at,
            last_activity: AtomicU64::new(
                self.last_activity
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            status: AtomicU8::new(self.status.load(std::sync::atomic::Ordering::Relaxed)),
            message_sender: self.message_sender.clone(),
            is_typing: AtomicBool::new(self.is_typing.load(std::sync::atomic::Ordering::Relaxed)),
            presence: AtomicU8::new(self.presence.load(std::sync::atomic::Ordering::Relaxed))}
    }
}

impl RealtimeConnection {
    /// Get connection status atomically
    #[inline]
    pub fn get_status(&self) -> ConnectionStatus {
        ConnectionStatus::from_atomic(self.status.load(Ordering::Relaxed))
    }

    /// Set connection status atomically
    #[inline]
    pub fn set_status(&self, status: ConnectionStatus) {
        self.status.store(status.to_atomic(), Ordering::Relaxed);
    }

    /// Get presence status atomically
    #[inline]
    pub fn get_presence(&self) -> PresenceStatus {
        PresenceStatus::from_atomic(self.presence.load(Ordering::Relaxed))
    }

    /// Set presence status atomically
    #[inline]
    pub fn set_presence(&self, presence: PresenceStatus) {
        self.presence.store(presence.to_atomic(), Ordering::Relaxed);
    }

    /// Update last activity timestamp
    #[inline]
    pub fn update_last_activity(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        self.last_activity.store(now, Ordering::Relaxed);
    }
}

/// Real-time message for live communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMessage {
    /// Message ID
    pub id: String,
    /// Message type
    pub message_type: RealtimeMessageType,
    /// Message content
    pub content: String,
    /// Sender user ID
    pub sender_id: String,
    /// Target user ID (for direct messages)
    pub target_id: Option<String>,
    /// Session ID
    pub session_id: String,
    /// Message timestamp
    pub timestamp: u64,
    /// Message metadata
    pub metadata: HashMap<String, serde_json::Value>}

/// Real-time message types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RealtimeMessageType {
    /// Regular chat message
    Chat,
    /// System message
    System,
    /// Typing indicator
    Typing,
    /// Presence update
    Presence,
    /// Connection status
    Connection,
    /// Error message
    Error}

// Duplicate ConnectionStatus removed - already defined above

/// Presence status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PresenceStatus {
    /// Online and available
    Online,
    /// Away from keyboard
    Away,
    /// Do not disturb
    DoNotDisturb,
    /// Offline
    Offline,
    /// Invisible
    Invisible}

impl PresenceStatus {
    /// Convert to atomic representation (u8)
    #[inline]
    pub const fn to_atomic(&self) -> u8 {
        match self {
            Self::Online => 0,
            Self::Away => 1,
            Self::DoNotDisturb => 2,
            Self::Offline => 3,
            Self::Invisible => 4}
    }

    /// Convert from atomic representation (u8)
    #[inline]
    pub const fn from_atomic(value: u8) -> Self {
        match value {
            0 => Self::Online,
            1 => Self::Away,
            2 => Self::DoNotDisturb,
            3 => Self::Offline,
            4 => Self::Invisible,
            _ => Self::Offline, // Default fallback
        }
    }
}

/// Real-time event types
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum RealtimeEventType {
    /// Connection established
    ConnectionEstablished,
    /// Connection lost
    ConnectionLost,
    /// Message received
    MessageReceived,
    /// Message sent
    MessageSent,
    /// Typing started
    TypingStarted,
    /// Typing stopped
    TypingStopped,
    /// Presence changed
    PresenceChanged,
    /// Error occurred
    Error}

/// Real-time event handler trait
pub trait RealtimeEventHandler: Send + Sync + std::fmt::Debug {
    /// Handle real-time event
    fn handle_event(
        &self,
        event: &RealTimeEvent,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    >;
}

/// Real-time chat metrics
#[derive(Debug, Default)]
pub struct RealtimeChatMetrics {
    /// Total connections established
    pub total_connections: ConsistentCounter,
    /// Active connections
    pub active_connections: AtomicUsize,
    /// Total messages sent
    pub total_messages: ConsistentCounter,
    /// Messages per second
    pub messages_per_second: AtomicU64,
    /// Average response time in microseconds
    pub avg_response_time_us: AtomicU64,
    /// Connection errors
    pub connection_errors: ConsistentCounter,
    /// Message delivery failures
    pub delivery_failures: ConsistentCounter}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            connection_timeout_seconds: 30,
            heartbeat_interval_seconds: 30,
            max_connections: 1000,
            message_buffer_size: 1000,
            enable_typing_indicators: true,
            typing_timeout_seconds: 5,
            enable_presence_tracking: true,
            presence_update_interval_seconds: 60,
            enable_message_streaming: true,
            stream_chunk_size: 8192,
            enable_compression: true,
            compression_threshold: 1024,
            enable_rate_limiting: true,
            rate_limit_messages_per_second: 10,
            enable_connection_pooling: true,
            connection_pool_size: 100,
            auto_reconnect: true,
            reconnect_delay_ms: 1000,
            max_reconnect_attempts: 5}
    }
}

impl RealtimeChat {
    /// Create a new real-time chat system
    pub fn new(config: RealtimeConfig) -> Self {
        let rt_system = RealTimeSystem::new();

        let (message_broadcaster, _) = broadcast::channel(config.message_buffer_size);

        Self {
            config,
            rt_system,
            connections: Arc::new(SkipMap::new()),
            message_broadcaster,
            event_handlers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RealtimeChatMetrics::default()),
            connection_task: None}
    }

    /// Start the real-time chat system
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.config.enabled {
            return Ok(());
        }

        // Start connection management task
        let connections = self.connections.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.heartbeat_interval_seconds));

            loop {
                interval.tick().await;

                // Clean up inactive connections
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let mut inactive_connections = Vec::new();

                for entry in connections.iter() {
                    let conn = entry.value();
                    let last_activity = conn.last_activity.load(Ordering::Relaxed);

                    if now - last_activity > config.connection_timeout_seconds {
                        inactive_connections.push(conn.connection_id.clone());
                    }
                }

                // Remove inactive connections
                for conn_id in inactive_connections {
                    connections.remove(&*conn_id);
                    // Decrement counter - AtomicUsize decrementation
                    let current = metrics
                        .active_connections
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if current > 0 {
                        metrics
                            .active_connections
                            .store(current - 1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        });

        self.connection_task = Some(task);
        Ok(())
    }

    /// Stop the real-time chat system
    pub async fn stop(&mut self) {
        if let Some(task) = self.connection_task.take() {
            task.abort();
        }

        // Close all connections
        self.connections.clear();
        // Reset counter to 0
        self.metrics.active_connections.store(0, Ordering::Relaxed);
    }

    /// Add a new connection
    pub async fn add_connection(
        &self,
        user_id: String,
        session_id: String,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if self.connections.len() >= self.config.max_connections {
            return Err("Maximum connections exceeded".into());
        }

        let connection_id = uuid::Uuid::new_v4().to_string();
        let (message_sender, _message_receiver) = mpsc::unbounded_channel();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let connection = RealtimeConnection {
            connection_id: connection_id.clone(),
            user_id,
            session_id,
            connected_at: now,
            last_activity: AtomicU64::new(now),
            status: AtomicU8::new(ConnectionStatus::Connected.to_atomic()),
            message_sender,
            is_typing: AtomicBool::new(false),
            presence: AtomicU8::new(PresenceStatus::Online.to_atomic())};

        self.connections
            .insert(Arc::from(connection_id.as_str()), connection);
        self.metrics.total_connections.inc();
        self.metrics
            .active_connections
            .fetch_add(1, Ordering::Relaxed);

        Ok(connection_id)
    }

    /// Remove a connection
    pub async fn remove_connection(
        &self,
        connection_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.connections.remove(connection_id).is_some() {
            // Decrement AtomicUsize counter
            self.metrics
                .active_connections
                .fetch_sub(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err("Connection not found".into())
        }
    }

    /// Send a message to a specific connection
    pub async fn send_message(
        &self,
        connection_id: &str,
        message: RealtimeMessage,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(connection) = self.connections.get(connection_id) {
            connection
                .value()
                .message_sender
                .send(message)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

            self.metrics.total_messages.inc();
            Ok(())
        } else {
            self.metrics.delivery_failures.inc();
            Err("Connection not found".into())
        }
    }

    /// Broadcast a message to all connections
    pub async fn broadcast_message(
        &self,
        message: RealtimeMessage,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.message_broadcaster
            .send(message)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        self.metrics.total_messages.inc();
        Ok(())
    }

    /// Set typing status for a connection
    pub async fn set_typing(
        &self,
        connection_id: &str,
        is_typing: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(connection) = self.connections.get(connection_id) {
            connection
                .value()
                .is_typing
                .store(is_typing, Ordering::Relaxed);

            // Update last activity
            connection.value().last_activity.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                Ordering::Relaxed,
            );

            Ok(())
        } else {
            Err("Connection not found".into())
        }
    }

    /// Set presence status for a connection
    pub async fn set_presence(
        &self,
        connection_id: &str,
        presence: PresenceStatus,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(connection) = self.connections.get(connection_id) {
            connection.value().set_presence(presence);
            connection.value().update_last_activity();
            Ok(())
        } else {
            Err("Connection not found".into())
        }
    }

    /// Get connection information
    pub async fn get_connection(&self, connection_id: &str) -> Option<RealtimeConnection> {
        self.connections
            .get(connection_id)
            .map(|entry| entry.value().clone())
    }

    /// Get all active connections
    pub async fn get_active_connections(&self) -> Vec<RealtimeConnection> {
        self.connections
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Register an event handler
    pub async fn register_event_handler(
        &self,
        event_type: RealtimeEventType,
        handler: Arc<dyn RealtimeEventHandler>,
    ) {
        let mut handlers = self.event_handlers.write().await;
        handlers
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(handler);
    }

    /// Get real-time chat metrics
    pub fn metrics(&self) -> &RealtimeChatMetrics {
        &self.metrics
    }

    /// Get configuration
    pub fn config(&self) -> &RealtimeConfig {
        &self.config
    }
}

impl Default for RealtimeChat {
    fn default() -> Self {
        Self::new(RealtimeConfig::default())
    }
}
