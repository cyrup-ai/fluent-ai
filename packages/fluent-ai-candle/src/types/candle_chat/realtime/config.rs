//! Real-time configuration and chat management
//!
//! This module provides configuration management for real-time chat features including
//! connection settings, message handling, and presence management.

use crate::types::candle_chat::realtime::{
    events::{RealTimeEvent, ConnectionStatus},
    builder::RealTimeSystemBuilder,
    system::RealTimeSystem};
use crossbeam_skiplist::SkipMap;
use crate::types::candle_chat::search::tagging::ConsistentCounter;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc, RwLock}};
use tokio::sync::{broadcast, mpsc};

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
    connection_task: Option<fluent_ai_async::AsyncTask<()>>}

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

impl RealtimeChat {
    /// Create a new real-time chat system
    pub fn new(config: RealtimeConfig) -> Self {
        let rt_system = RealTimeSystemBuilder::new()
            .typing_expiry(config.typing_timeout_seconds)
            .heartbeat_timeout(config.heartbeat_interval_seconds)
            .queue_size_limit(config.message_buffer_size)
            .build();

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

    /// Get configuration
    pub fn config(&self) -> &RealtimeConfig {
        &self.config
    }

    /// Get real-time chat metrics
    pub fn metrics(&self) -> &RealtimeChatMetrics {
        &self.metrics
    }
}

impl Default for RealtimeChat {
    fn default() -> Self {
        Self::new(RealtimeConfig::default())
    }
}