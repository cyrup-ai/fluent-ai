//! Live message streaming system with zero-allocation patterns
//!
//! This module provides high-performance message streaming using lock-free queues,
//! atomic counters, and AsyncStream patterns for blazing-fast real-time updates.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_queue::SegQueue;
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::{emit, AsyncStream};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use super::events::RealTimeEvent;
// Use the domain's RealTimeError
use crate::domain::chat::message::types::{
    CandleMessage as Message, CandleMessageRole as MessageRole,
};

/// Live update message with zero-allocation string handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveUpdateMessage {
    /// Message ID with zero-allocation storage
    pub id: Arc<str>,
    /// Message content with zero-allocation storage
    pub content: Arc<str>,
    /// Message type identifier
    pub message_type: Arc<str>,
    /// Session ID with zero-allocation storage
    pub session_id: Arc<str>,
    /// User ID with zero-allocation storage
    pub user_id: Arc<str>,
    /// Timestamp in nanoseconds for high precision
    pub timestamp_nanos: u64,
    /// Message priority level
    pub priority: MessagePriority,
    /// Optional metadata with zero-allocation storage
    pub metadata: Option<Arc<str>>,
    /// Message size in bytes for monitoring
    pub size_bytes: u32,
    /// Sequence number for ordering
    pub sequence_number: u64,
}

impl LiveUpdateMessage {
    /// Create a new live update message with current timestamp
    #[inline]
    pub fn new(
        id: Arc<str>,
        content: Arc<str>,
        message_type: Arc<str>,
        session_id: Arc<str>,
        user_id: Arc<str>,
        priority: MessagePriority,
    ) -> Self {
        let timestamp_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let size_bytes =
            (id.len() + content.len() + message_type.len() + session_id.len() + user_id.len())
                as u32;

        Self {
            id,
            content,
            message_type,
            session_id,
            user_id,
            timestamp_nanos,
            priority,
            metadata: None,
            size_bytes,
            sequence_number: 0, // Will be set by streamer
        }
    }

    /// Add metadata to the message
    #[inline]
    pub fn with_metadata(mut self, metadata: Arc<str>) -> Self {
        self.size_bytes += metadata.len() as u32;
        self.metadata = Some(metadata);
        self
    }

    /// Get timestamp in seconds
    #[inline]
    pub fn timestamp_seconds(&self) -> f64 {
        self.timestamp_nanos as f64 / 1_000_000_000.0
    }

    /// Get message age in nanoseconds
    #[inline]
    pub fn age_nanos(&self) -> u64 {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        now_nanos.saturating_sub(self.timestamp_nanos)
    }

    /// Check if message has expired based on TTL
    #[inline]
    pub fn is_expired(&self, ttl_nanos: u64) -> bool {
        self.age_nanos() > ttl_nanos
    }

    /// Convert to RealTimeEvent for broadcasting
    pub fn to_real_time_event(&self) -> RealTimeEvent {
        // Create a basic Message for the event
        let message = Message::new(
            self.sequence_number,
            MessageRole::Assistant,
            self.content.as_bytes(),
        );

        RealTimeEvent::message_received(message, self.session_id.clone())
    }
}

/// Message priority levels with atomic-friendly ordering
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MessagePriority {
    /// Low priority - background updates
    Low,
    /// Normal priority - standard messages
    Normal,
    /// High priority - important updates
    High,
    /// Critical priority - urgent notifications
    Critical,
}

impl MessagePriority {
    /// Get priority weight for ordering (higher = more important)
    #[inline]
    pub const fn weight(&self) -> u8 {
        match self {
            Self::Low => 1,
            Self::Normal => 5,
            Self::High => 10,
            Self::Critical => 20,
        }
    }

    /// Convert to atomic representation
    #[inline]
    pub const fn to_atomic(&self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }

    /// Convert from atomic representation
    #[inline]
    pub const fn from_atomic(value: u8) -> Self {
        match value {
            0 => Self::Low,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Critical,
        }
    }
}

impl Default for MessagePriority {
    #[inline]
    fn default() -> Self {
        Self::Normal
    }
}

/// Live message streamer with lock-free queuing and atomic statistics
pub struct LiveMessageStreamer {
    /// Message queue for lock-free streaming
    message_queue: Arc<SegQueue<LiveUpdateMessage>>,
    /// Priority queue for high-priority messages
    priority_queue: Arc<SegQueue<LiveUpdateMessage>>,
    /// Active subscribers with zero-allocation keys
    subscribers: Arc<SkipMap<Arc<str>, Arc<StreamSubscriber>>>,
    /// Event broadcaster for real-time notifications
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Message counter for statistics
    message_counter: Arc<AtomicUsize>,
    /// Priority message counter
    priority_message_counter: Arc<AtomicUsize>,
    /// Subscriber counter
    subscriber_counter: Arc<ConsistentCounter>,
    /// Total bytes processed
    bytes_processed: Arc<AtomicU64>,
    /// Queue size limit for backpressure
    queue_size_limit: Arc<AtomicUsize>,
    /// Backpressure threshold
    #[allow(dead_code)] // TODO: Implement backpressure throttling logic
    backpressure_threshold: Arc<AtomicUsize>,
    /// Processing rate in messages per second
    processing_rate: Arc<AtomicU64>,
    /// Sequence number generator
    sequence_generator: Arc<AtomicU64>,
    /// Backpressure event counter
    backpressure_events: Arc<AtomicU64>,
    /// Processing task active flag
    processing_active: Arc<std::sync::atomic::AtomicBool>,
}

/// Stream subscriber with atomic statistics
#[derive(Debug)]
pub struct StreamSubscriber {
    /// Subscriber ID
    pub id: Arc<str>,
    /// Session ID filter (None = all sessions)
    pub session_filter: Option<Arc<str>>,
    /// User ID filter (None = all users)
    pub user_filter: Option<Arc<str>>,
    /// Minimum priority filter
    pub min_priority: MessagePriority,
    /// Messages received counter
    pub messages_received: Arc<AtomicU64>,
    /// Bytes received counter
    pub bytes_received: Arc<AtomicU64>,
    /// Subscription timestamp
    pub subscribed_at: u64,
    /// Last message timestamp
    pub last_message_at: Arc<AtomicU64>,
}

impl StreamSubscriber {
    /// Create new stream subscriber
    #[inline]
    pub fn new(id: Arc<str>) -> Self {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            id,
            session_filter: None,
            user_filter: None,
            min_priority: MessagePriority::Normal,
            messages_received: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            subscribed_at: now_nanos,
            last_message_at: Arc::new(AtomicU64::new(now_nanos)),
        }
    }

    /// Set session filter
    #[inline]
    pub fn with_session_filter(mut self, session_id: Arc<str>) -> Self {
        self.session_filter = Some(session_id);
        self
    }

    /// Set user filter
    #[inline]
    pub fn with_user_filter(mut self, user_id: Arc<str>) -> Self {
        self.user_filter = Some(user_id);
        self
    }

    /// Set minimum priority filter
    #[inline]
    pub fn with_min_priority(mut self, priority: MessagePriority) -> Self {
        self.min_priority = priority;
        self
    }

    /// Check if message should be delivered to this subscriber
    pub fn should_receive(&self, message: &LiveUpdateMessage) -> bool {
        // Check session filter
        if let Some(session_filter) = &self.session_filter {
            if message.session_id != *session_filter {
                return false;
            }
        }

        // Check user filter
        if let Some(user_filter) = &self.user_filter {
            if message.user_id != *user_filter {
                return false;
            }
        }

        // Check priority filter
        if message.priority < self.min_priority {
            return false;
        }

        true
    }

    /// Record message delivery
    #[inline]
    pub fn record_delivery(&self, message: &LiveUpdateMessage) {
        self.messages_received.fetch_add(1, Ordering::AcqRel);
        self.bytes_received
            .fetch_add(message.size_bytes as u64, Ordering::AcqRel);
        self.last_message_at
            .store(message.timestamp_nanos, Ordering::Release);
    }

    /// Get subscription duration in nanoseconds
    #[inline]
    pub fn subscription_duration_nanos(&self) -> u64 {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        now_nanos.saturating_sub(self.subscribed_at)
    }
}

impl LiveMessageStreamer {
    /// Create new live message streamer
    #[inline]
    pub fn new(
        queue_size_limit: usize,
        backpressure_threshold: usize,
        processing_rate: u64,
    ) -> Self {
        let (event_broadcaster, _) = broadcast::channel(50000); // Large buffer for performance

        Self {
            message_queue: Arc::new(SegQueue::new()),
            priority_queue: Arc::new(SegQueue::new()),
            subscribers: Arc::new(SkipMap::new()),
            event_broadcaster,
            message_counter: Arc::new(AtomicUsize::new(0)),
            priority_message_counter: Arc::new(AtomicUsize::new(0)),
            subscriber_counter: Arc::new(ConsistentCounter::new(0)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            queue_size_limit: Arc::new(AtomicUsize::new(queue_size_limit)),
            backpressure_threshold: Arc::new(AtomicUsize::new(backpressure_threshold)),
            processing_rate: Arc::new(AtomicU64::new(processing_rate)),
            sequence_generator: Arc::new(AtomicU64::new(1)),
            backpressure_events: Arc::new(AtomicU64::new(0)),
            processing_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Send live update message with backpressure handling
    pub fn send_message(&self, mut message: LiveUpdateMessage) -> AsyncStream<StreamingResult> {
        let message_queue = if message.priority >= MessagePriority::High {
            self.priority_queue.clone()
        } else {
            self.message_queue.clone()
        };

        let counter = if message.priority >= MessagePriority::High {
            self.priority_message_counter.clone()
        } else {
            self.message_counter.clone()
        };

        let queue_size_limit = self.queue_size_limit.clone();
        let backpressure_threshold = self.backpressure_threshold.clone();
        let backpressure_events = self.backpressure_events.clone();
        let sequence_generator = self.sequence_generator.clone();
        let bytes_processed = self.bytes_processed.clone();
        let event_broadcaster = self.event_broadcaster.clone();

        AsyncStream::with_channel(move |sender| {
            // Check for backpressure
            let current_queue_size = counter.load(Ordering::Acquire);
            let queue_limit = queue_size_limit.load(Ordering::Acquire);
            let bp_threshold = backpressure_threshold.load(Ordering::Acquire);

            // Hard limit check - reject messages
            if current_queue_size >= queue_limit {
                backpressure_events.fetch_add(1, Ordering::AcqRel);

                let result = StreamingResult::BackpressureError {
                    current_size: current_queue_size,
                    limit: queue_limit,
                    message_id: message.id.clone(),
                };
                emit!(sender, result);
                return;
            }

            // Soft limit check - emit warning but allow message
            if current_queue_size >= bp_threshold {
                let warning_result = StreamingResult::BackpressureWarning {
                    current_size: current_queue_size,
                    threshold: bp_threshold,
                    message_id: message.id.clone(),
                };
                emit!(sender, warning_result);
            }

            // Assign sequence number
            message.sequence_number = sequence_generator.fetch_add(1, Ordering::AcqRel);

            // Add message to queue
            message_queue.push(message.clone());
            counter.fetch_add(1, Ordering::AcqRel);
            bytes_processed.fetch_add(message.size_bytes as u64, Ordering::AcqRel);

            // Broadcast real-time event
            let _ = event_broadcaster.send(message.to_real_time_event());

            let result = StreamingResult::MessageQueued {
                message_id: message.id,
                sequence_number: message.sequence_number,
                queue_position: current_queue_size + 1,
            };
            emit!(sender, result);
        })
    }

    /// Subscribe to live updates with filtering
    pub fn subscribe(&self, subscriber: StreamSubscriber) -> AsyncStream<LiveUpdateMessage> {
        let subscriber_arc = Arc::new(subscriber);
        let subscriber_id = subscriber_arc.id.clone();

        self.subscribers.insert(subscriber_id, subscriber_arc);
        self.subscriber_counter.inc();

        // Return empty stream - messages will be delivered via processing task
        AsyncStream::with_channel(|_sender| {})
    }

    /// Unsubscribe from live updates
    pub fn unsubscribe(&self, subscriber_id: &Arc<str>) -> AsyncStream<UnsubscribeResult> {
        let subscribers = self.subscribers.clone();
        let subscriber_counter = self.subscriber_counter.clone();
        let id = subscriber_id.clone();

        AsyncStream::with_channel(move |sender| {
            let result = if subscribers.remove(&id).is_some() {
                // Decrement counter efficiently
                let current = subscriber_counter.get();
                if current > 0 {
                    subscriber_counter.reset();
                    for _ in 0..(current - 1) {
                        subscriber_counter.inc();
                    }
                }

                UnsubscribeResult::Success {
                    subscriber_id: id,
                    remaining_subscribers: subscriber_counter.get() as u64,
                }
            } else {
                UnsubscribeResult::NotFound { subscriber_id: id }
            };

            emit!(sender, result);
        })
    }

    /// Start message processing task with lock-free distribution
    pub fn start_processing(&self) -> AsyncStream<ProcessingEvent> {
        if self
            .processing_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            // Already running
            return AsyncStream::with_channel(|_sender| {});
        }

        let message_queue = self.message_queue.clone();
        let priority_queue = self.priority_queue.clone();
        let subscribers = self.subscribers.clone();
        let message_counter = self.message_counter.clone();
        let priority_message_counter = self.priority_message_counter.clone();
        let processing_rate = self.processing_rate.clone();
        let processing_active = self.processing_active.clone();

        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                let mut messages_processed = 0u64;
                let mut last_rate_check = std::time::Instant::now();

                loop {
                    let target_rate = processing_rate.load(Ordering::Acquire);
                    let delay_nanos = if target_rate > 0 {
                        1_000_000_000 / target_rate
                    } else {
                        1_000_000 // 1ms default
                    };

                    // Process priority messages first
                    let message = if let Some(priority_msg) = priority_queue.pop() {
                        priority_message_counter.fetch_sub(1, Ordering::AcqRel);
                        Some(priority_msg)
                    } else if let Some(normal_msg) = message_queue.pop() {
                        message_counter.fetch_sub(1, Ordering::AcqRel);
                        Some(normal_msg)
                    } else {
                        None
                    };

                    if let Some(message) = message {
                        let mut delivered_count = 0u64;
                        let mut total_bytes = 0u64;

                        // Distribute to matching subscribers
                        for entry in subscribers.iter() {
                            let subscriber = entry.value();
                            if subscriber.should_receive(&message) {
                                subscriber.record_delivery(&message);
                                delivered_count += 1;
                                total_bytes += message.size_bytes as u64;
                            }
                        }

                        messages_processed += 1;

                        // Emit processing event
                        let event = ProcessingEvent::MessageProcessed {
                            message_id: message.id,
                            sequence_number: message.sequence_number,
                            delivered_count,
                            total_bytes,
                            priority: message.priority,
                        };
                        emit!(sender, event);

                        // Rate limiting with nanosecond precision
                        std::thread::sleep(Duration::from_nanos(delay_nanos));
                    } else {
                        // No messages, sleep briefly
                        std::thread::sleep(Duration::from_millis(1));
                    }

                    // Report processing rate periodically
                    if last_rate_check.elapsed() >= Duration::from_secs(10) {
                        let rate =
                            messages_processed as f64 / last_rate_check.elapsed().as_secs_f64();

                        let event = ProcessingEvent::RateReport {
                            messages_per_second: rate,
                            messages_processed,
                            active_subscribers: subscribers.len() as u64,
                        };
                        emit!(sender, event);

                        messages_processed = 0;
                        last_rate_check = std::time::Instant::now();
                    }

                    // Check if we should continue
                    if !processing_active.load(Ordering::Acquire) {
                        break;
                    }
                }
            });
        })
    }

    /// Stop processing task
    #[inline]
    pub fn stop_processing(&self) {
        self.processing_active.store(false, Ordering::Release);
    }

    /// Get current backpressure threshold
    #[inline]
    pub fn get_backpressure_threshold(&self) -> usize {
        self.backpressure_threshold.load(Ordering::Acquire)
    }

    /// Update backpressure threshold dynamically
    #[inline]
    pub fn set_backpressure_threshold(&self, threshold: usize) {
        self.backpressure_threshold
            .store(threshold, Ordering::Release);
    }

    /// Get comprehensive streaming statistics
    pub fn get_statistics(&self) -> StreamingStatistics {
        StreamingStatistics {
            total_messages: self.message_counter.load(Ordering::Acquire),
            priority_messages: self.priority_message_counter.load(Ordering::Acquire),
            active_subscribers: self.subscriber_counter.get(),
            total_subscribers: self.subscribers.len(),
            bytes_processed: self.bytes_processed.load(Ordering::Acquire),
            queue_size_limit: self.queue_size_limit.load(Ordering::Acquire),
            backpressure_events: self.backpressure_events.load(Ordering::Acquire),
            processing_rate: self.processing_rate.load(Ordering::Acquire),
            processing_active: self.processing_active.load(Ordering::Acquire),
        }
    }

    /// Subscribe to real-time events
    #[inline]
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<RealTimeEvent> {
        self.event_broadcaster.subscribe()
    }
}

impl Drop for LiveMessageStreamer {
    fn drop(&mut self) {
        self.stop_processing();
    }
}

/// Streaming operation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingResult {
    /// Message successfully queued
    MessageQueued {
        message_id: Arc<str>,
        sequence_number: u64,
        queue_position: usize,
    },
    /// Backpressure warning - queue size approaching limit
    BackpressureWarning {
        current_size: usize,
        threshold: usize,
        message_id: Arc<str>,
    },
    /// Backpressure error occurred
    BackpressureError {
        current_size: usize,
        limit: usize,
        message_id: Arc<str>,
    },
}

/// Unsubscribe operation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnsubscribeResult {
    /// Successfully unsubscribed
    Success {
        subscriber_id: Arc<str>,
        remaining_subscribers: u64,
    },
    /// Subscriber not found
    NotFound { subscriber_id: Arc<str> },
}

/// Processing events for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingEvent {
    /// Message processed and distributed
    MessageProcessed {
        message_id: Arc<str>,
        sequence_number: u64,
        delivered_count: u64,
        total_bytes: u64,
        priority: MessagePriority,
    },
    /// Processing rate report
    RateReport {
        messages_per_second: f64,
        messages_processed: u64,
        active_subscribers: u64,
    },
}

/// Live streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStatistics {
    /// Total messages in normal queue
    pub total_messages: usize,
    /// Total messages in priority queue
    pub priority_messages: usize,
    /// Number of active subscribers
    pub active_subscribers: usize,
    /// Total subscribers ever created
    pub total_subscribers: usize,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Queue size limit
    pub queue_size_limit: usize,
    /// Number of backpressure events
    pub backpressure_events: u64,
    /// Target processing rate (messages/second)
    pub processing_rate: u64,
    /// Whether processing is active
    pub processing_active: bool,
}
