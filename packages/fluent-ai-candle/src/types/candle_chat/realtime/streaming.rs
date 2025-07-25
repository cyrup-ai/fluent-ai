//! Live message streaming with zero-allocation queue management
//!
//! This module provides the LiveMessageStreamer for real-time message streaming
//! with backpressure handling and atomic statistics tracking.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

use crate::types::candle_chat::search::tagging::ConsistentCounter;
use crossbeam_queue::SegQueue;
use fluent_ai_async::{AsyncStream, emit, handle_error};
use tokio::sync::broadcast;

use crate::types::CandleMessage;
use super::events::RealTimeEvent;
use super::errors::RealTimeError;
use super::live_updates::LiveUpdateMessage;

/// Live message streaming statistics with atomic counters
#[derive(Debug)]
pub struct StreamingStatistics {
    pub total_messages: usize,
    pub active_subscribers: usize,
    pub queue_size: usize,
    pub backpressure_events: usize,
    pub processing_rate: f64,
    pub last_update: u64}

/// High-performance message streamer with lock-free operations
pub struct LiveMessageStreamer {
    /// Lock-free message queue
    message_queue: Arc<SegQueue<LiveUpdateMessage>>,
    /// Subscriber channels with Arc<str> keys for zero-allocation
    subscribers: Arc<RwLock<HashMap<Arc<str>, crossbeam_channel::Sender<LiveUpdateMessage>>>>,
    /// Atomic message counter
    message_counter: Arc<AtomicUsize>,
    /// Subscriber counter with atomic operations
    subscriber_counter: Arc<ConsistentCounter>,
    /// Queue size limit
    queue_size_limit: AtomicUsize,
    /// Backpressure threshold
    backpressure_threshold: AtomicUsize,
    /// Streaming statistics
    stats: Arc<RwLock<StreamingStatistics>>,
    /// Event broadcaster for real-time events
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Processing rate limiter
    processing_rate: AtomicU64}

impl LiveMessageStreamer {
    /// Create new message streamer with optimal configuration
    pub fn new(
        queue_size_limit: usize,
        backpressure_threshold: usize,
        processing_rate: u64,
    ) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1024);

        Self {
            message_queue: Arc::new(SegQueue::new()),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            message_counter: Arc::new(AtomicUsize::new(0)),
            subscriber_counter: Arc::new(ConsistentCounter::new(0)),
            queue_size_limit: AtomicUsize::new(queue_size_limit),
            backpressure_threshold: AtomicUsize::new(backpressure_threshold),
            stats: Arc::new(RwLock::new(StreamingStatistics {
                total_messages: 0,
                active_subscribers: 0,
                queue_size: 0,
                backpressure_events: 0,
                processing_rate: processing_rate as f64,
                last_update: 0})),
            event_broadcaster,
            processing_rate: AtomicU64::new(processing_rate)}
    }

    /// Stream message with backpressure handling
    pub fn stream_message(&self, message: LiveUpdateMessage) -> AsyncStream<()> {
        let message_counter = self.message_counter.clone();
        let queue_size_limit = self.queue_size_limit.load(Ordering::Relaxed);
        let message_queue = self.message_queue.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let stats = self.stats.clone();

        AsyncStream::with_channel(move |sender| {
            let current_queue_size = message_counter.load(Ordering::Relaxed);

            // Backpressure check with atomic operations
            if current_queue_size >= queue_size_limit {
                if let Ok(mut stats_guard) = stats.write() {
                    stats_guard.backpressure_events += 1;
                }

                let error = RealTimeError::BackpressureExceeded {
                    current_size: current_queue_size,
                    limit: queue_size_limit};
                handle_error!(error, "Backpressure exceeded in stream_message");
            }

            // Zero-allocation queue push
            message_queue.push(message.clone());
            message_counter.fetch_add(1, Ordering::Release);

            // Broadcast real-time event
            let event = RealTimeEvent::MessageReceived {
                message: CandleMessage::new(
                    crate::types::CandleMessageRole::Assistant,
                    message.content.clone(),
                ),
                session_id: message.session_id.clone(),
                timestamp: message.timestamp};

            let _ = event_broadcaster.send(event);

            // Update statistics atomically
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

    /// Subscribe to message stream with AsyncStream pattern
    pub fn subscribe_stream(&self, subscriber_id: Arc<str>) -> AsyncStream<LiveUpdateMessage> {
        let subscribers = self.subscribers.clone();
        let subscriber_counter = self.subscriber_counter.clone();
        let stats = self.stats.clone();

        AsyncStream::with_channel(move |sender| {
            let (tx, rx) = crossbeam_channel::unbounded::<LiveUpdateMessage>();

            // Add subscriber with zero-allocation key
            if let Ok(mut subs) = subscribers.write() {
                subs.insert(subscriber_id.clone(), tx);
                subscriber_counter.inc();

                // Update statistics
                if let Ok(mut stats_guard) = stats.write() {
                    stats_guard.active_subscribers = subs.len();
                }
            }

            // Stream messages to subscriber
            while let Ok(message) = rx.recv() {
                emit!(sender, message);
            }
        })
    }

    /// Unsubscribe from message stream
    pub fn unsubscribe_stream(&self, subscriber_id: &Arc<str>) -> AsyncStream<bool> {
        let subscribers = self.subscribers.clone();
        let subscriber_counter = self.subscriber_counter.clone();
        let stats = self.stats.clone();
        let id = subscriber_id.clone();

        AsyncStream::with_channel(move |sender| {
            let removed = if let Ok(mut subs) = subscribers.write() {
                if subs.remove(&id).is_some() {
                    subscriber_counter.dec();
                    
                    // Update statistics
                    if let Ok(mut stats_guard) = stats.write() {
                        stats_guard.active_subscribers = subs.len();
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };

            emit!(sender, removed);
        })
    }

    /// Process message queue with rate limiting
    pub fn process_queue(&self) -> AsyncStream<usize> {
        let message_queue = self.message_queue.clone();
        let message_counter = self.message_counter.clone();
        let subscribers = self.subscribers.clone();
        let processing_rate = self.processing_rate.load(Ordering::Acquire);
        let stats = self.stats.clone();

        AsyncStream::with_channel(move |sender| {
            let mut processed_count = 0;
            let rate_limit = std::time::Duration::from_millis(1000 / processing_rate);

            while let Some(message) = message_queue.pop() {
                message_counter.fetch_sub(1, Ordering::AcqRel);
                processed_count += 1;

                // Distribute to all subscribers
                if let Ok(subs) = subscribers.read() {
                    for (_, tx) in subs.iter() {
                        let _ = tx.try_send(message.clone());
                    }
                }

                // Rate limiting
                std::thread::sleep(rate_limit);

                // Update queue size in statistics
                if let Ok(mut stats_guard) = stats.write() {
                    stats_guard.queue_size = message_counter.load(Ordering::Relaxed);
                }
            }

            emit!(sender, processed_count);
        })
    }

    /// Get current streaming statistics
    pub fn get_statistics(&self) -> AsyncStream<StreamingStatistics> {
        let stats = self.stats.clone();
        let message_counter = self.message_counter.clone();

        AsyncStream::with_channel(move |sender| {
            if let Ok(mut stats_guard) = stats.write() {
                stats_guard.queue_size = message_counter.load(Ordering::Relaxed);
                let stats_snapshot = StreamingStatistics {
                    total_messages: stats_guard.total_messages,
                    active_subscribers: stats_guard.active_subscribers,
                    queue_size: stats_guard.queue_size,
                    backpressure_events: stats_guard.backpressure_events,
                    processing_rate: stats_guard.processing_rate,
                    last_update: stats_guard.last_update};
                emit!(sender, stats_snapshot);
            }
        })
    }

    /// Broadcast event to all subscribers
    pub fn broadcast_event(&self, event: RealTimeEvent) -> AsyncStream<usize> {
        let event_broadcaster = self.event_broadcaster.clone();

        AsyncStream::with_channel(move |sender| {
            match event_broadcaster.send(event) {
                Ok(subscriber_count) => emit!(sender, subscriber_count),
                Err(_) => emit!(sender, 0)}
        })
    }

    /// Update processing rate dynamically
    pub fn set_processing_rate(&self, rate: u64) -> AsyncStream<()> {
        self.processing_rate.store(rate, Ordering::AcqRel);

        let stats = Arc::clone(&self.stats);
        AsyncStream::with_channel(move |sender| {
            if let Ok(mut stats_guard) = stats.write() {
                stats_guard.processing_rate = rate as f64;
            }

            emit!(sender, ());
        })
    }

    /// Get subscriber count
    pub fn get_subscriber_count(&self) -> usize {
        self.subscriber_counter.get()
    }

    /// Get queue size
    pub fn get_queue_size(&self) -> usize {
        self.message_counter.load(Ordering::Relaxed)
    }
}

impl Default for LiveMessageStreamer {
    fn default() -> Self {
        Self::new(1000, 800, 100)
    }
}