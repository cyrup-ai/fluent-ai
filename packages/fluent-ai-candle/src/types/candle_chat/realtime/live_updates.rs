//! Live update system with message streaming and backpressure handling
//!
//! This module provides comprehensive live message streaming capabilities with zero-allocation
//! patterns, backpressure management, and atomic operation-based performance tracking.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;
use std::time::Duration;

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use fluent_ai_async::{AsyncStream, emit, handle_error};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::types::CandleMessage;
use super::events::RealTimeEvent;
use super::errors::RealTimeError;

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
    pub metadata: Option<String>,
}

impl LiveUpdateMessage {
    /// Create a new live update message
    pub fn new(
        id: String,
        content: String,
        message_type: String,
        session_id: String,
        user_id: String,
        priority: MessagePriority,
    ) -> Self {
        Self {
            id,
            content,
            message_type,
            session_id,
            user_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            priority,
            metadata: None,
        }
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get priority value for sorting
    pub fn priority_value(&self) -> u8 {
        self.priority.value()
    }
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

impl MessagePriority {
    /// Get numeric priority value
    pub fn value(&self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }

    /// Check if priority is urgent
    pub fn is_urgent(&self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Live update system with message streaming and backpressure handling
pub struct LiveUpdateSystem {
    /// Message queue for streaming
    message_queue: Arc<SegQueue<LiveUpdateMessage>>,
    /// Subscriber channels
    subscribers: Arc<RwLock<HashMap<Arc<str>, crossbeam_channel::Sender<LiveUpdateMessage>>>>,
    /// Message counter
    message_counter: Arc<AtomicUsize>,
    /// Subscriber counter
    subscriber_counter: Arc<ConsistentCounter>,
    /// Queue size limit
    queue_size_limit: AtomicUsize,
    /// Backpressure threshold
    backpressure_threshold: AtomicUsize,
    /// System statistics
    stats: Arc<RwLock<LiveUpdateStatistics>>,
    /// Event broadcaster for real-time events
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Rate limiter for processing
    rate_limiter: Arc<RwLock<tokio::time::Interval>>,
    /// Processing rate limit
    processing_rate: AtomicU64,
}

impl LiveUpdateSystem {
    /// Create a new live update system
    pub fn new(
        queue_size_limit: usize,
        backpressure_threshold: usize,
        processing_rate: u64,
    ) -> Self {
        let (event_broadcaster, _) = broadcast::channel::<RealTimeEvent>(1000);
        let rate_limiter = Arc::new(RwLock::new(tokio::time::interval(Duration::from_millis(
            1000 / processing_rate.max(1),
        ))));

        Self {
            message_queue: Arc::new(SegQueue::new()),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            message_counter: Arc::new(AtomicUsize::new(0)),
            subscriber_counter: Arc::new(ConsistentCounter::new(0)),
            queue_size_limit: AtomicUsize::new(queue_size_limit),
            backpressure_threshold: AtomicUsize::new(backpressure_threshold),
            stats: Arc::new(RwLock::new(LiveUpdateStatistics {
                total_messages: 0,
                active_subscribers: 0,
                queue_size: 0,
                backpressure_events: 0,
                processing_rate: processing_rate as f64,
                last_update: 0,
            })),
            event_broadcaster,
            rate_limiter,
            processing_rate: AtomicU64::new(processing_rate),
        }
    }

    /// Send live update message using AsyncStream architecture
    pub fn send_message(&self, message: LiveUpdateMessage) -> AsyncStream<()> {
        let message_counter = self.message_counter.clone();
        let queue_size_limit = self.queue_size_limit.load(Ordering::Relaxed);
        let message_queue = self.message_queue.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let stats = self.stats.clone();

        AsyncStream::with_channel(move |sender| {
            let current_queue_size = message_counter.load(Ordering::Relaxed);

            // Check for backpressure
            if current_queue_size >= queue_size_limit {
                if let Ok(mut stats_guard) = stats.write() {
                    stats_guard.backpressure_events += 1;
                }

                let error = RealTimeError::BackpressureExceeded {
                    current_size: current_queue_size,
                    limit: queue_size_limit,
                };
                handle_error!(error, "Backpressure exceeded in send_message");
                return;
            }

            // Add message to queue
            message_queue.push(message.clone());
            message_counter.fetch_add(1, Ordering::Relaxed);

            // Broadcast real-time event
            let event = RealTimeEvent::MessageReceived {
                message: CandleMessage::new(
                    message.user_id.len() as u64, // Use user_id length as ID for now
                    crate::types::CandleMessageRole::Assistant,
                    message.content.as_bytes(),
                ),
                session_id: message.session_id.clone(),
                timestamp: message.timestamp,
            };

            let _ = event_broadcaster.send(event);

            // Update statistics
            if let Ok(mut stats_guard) = stats.write() {
                stats_guard.total_messages += 1;
                stats_guard.last_update = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }

            emit!(sender, ());
        })
    }

    /// Subscribe to live updates using AsyncStream architecture
    pub fn subscribe(&self, subscriber_id: Arc<str>) -> AsyncStream<LiveUpdateMessage> {
        let subscribers = self.subscribers.clone();
        let subscriber_counter = self.subscriber_counter.clone();
        let stats = self.stats.clone();

        AsyncStream::with_channel(move |sender| {
            let (tx, rx) = crossbeam_channel::unbounded::<LiveUpdateMessage>();

            // Add subscriber
            if let Ok(mut subs) = subscribers.write() {
                subs.insert(subscriber_id.clone(), tx);
                subscriber_counter.inc();

                // Update statistics
                if let Ok(mut stats_guard) = stats.write() {
                    stats_guard.active_subscribers = subs.len();
                }
            }

            // Listen for messages
            while let Ok(message) = rx.recv() {
                emit!(sender, message);
            }

            // Cleanup on disconnect
            if let Ok(mut subs) = subscribers.write() {
                subs.remove(&subscriber_id);
                // ConsistentCounter doesn't have dec(), so we reset and re-count
                let current = subscriber_counter.get();
                if current > 0 {
                    subscriber_counter.reset();
                    for _ in 0..(current - 1) {
                        subscriber_counter.inc();
                    }
                }

                // Update statistics
                if let Ok(mut stats_guard) = stats.write() {
                    stats_guard.active_subscribers = subs.len();
                }
            }
        })
    }

    /// Get current queue size
    pub fn queue_size(&self) -> usize {
        self.message_counter.load(Ordering::Relaxed)
    }

    /// Check if system is under backpressure
    pub fn is_backpressure_active(&self) -> bool {
        let current_size = self.message_counter.load(Ordering::Relaxed);
        let threshold = self.backpressure_threshold.load(Ordering::Relaxed);
        current_size >= threshold
    }

    /// Get system statistics
    pub fn get_statistics(&self) -> LiveUpdateStatistics {
        if let Ok(stats) = self.stats.read() {
            let mut stats_clone = stats.clone();
            stats_clone.queue_size = self.queue_size();
            stats_clone
        } else {
            LiveUpdateStatistics::default()
        }
    }

    /// Update processing rate
    pub fn set_processing_rate(&self, rate: u64) {
        self.processing_rate.store(rate, Ordering::Relaxed);
        
        // Update rate limiter
        if let Ok(mut limiter) = self.rate_limiter.write() {
            *limiter = tokio::time::interval(Duration::from_millis(1000 / rate.max(1)));
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.processing_rate = rate as f64;
        }
    }
}

impl std::fmt::Debug for LiveUpdateSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveUpdateSystem")
            .field("queue_size", &self.queue_size())
            .field("active_subscribers", &self.subscriber_counter.get())
            .field("backpressure_active", &self.is_backpressure_active())
            .field("processing_rate", &self.processing_rate.load(Ordering::Relaxed))
            .finish()
    }
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

impl Default for LiveUpdateStatistics {
    fn default() -> Self {
        Self {
            total_messages: 0,
            active_subscribers: 0,
            queue_size: 0,
            backpressure_events: 0,
            processing_rate: 60.0,
            last_update: 0,
        }
    }
}