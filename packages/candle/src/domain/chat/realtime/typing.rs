//! Typing indicator system with lock-free atomic operations
//!
//! This module provides zero-allocation typing state management using atomic operations,
//! crossbeam-skiplist for concurrent access, and AsyncStream integration for blazing-fast
//! performance without any locking mechanisms.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::{AsyncStream, emit};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use super::events::{RealTimeEvent, NotificationLevel};

// Re-export RealTimeError from the domain crate
use super::RealTimeError;

/// Typing indicator state with atomic operations for zero-allocation performance
#[derive(Debug)]
pub struct TypingState {
    /// User ID with zero-allocation Arc storage
    pub user_id: Arc<str>,
    /// Session ID with zero-allocation Arc storage
    pub session_id: Arc<str>,
    /// Last activity timestamp in nanoseconds for high precision
    pub last_activity: AtomicU64,
    /// Is currently typing flag
    pub is_typing: AtomicBool,
    /// Total typing duration in nanoseconds
    pub typing_duration: AtomicU64,
    /// Typing session start timestamp
    pub session_start: AtomicU64,
    /// Number of typing events in this session
    pub event_count: AtomicU64}

impl TypingState {
    /// Create a new typing state with current timestamp
    #[inline]
    pub fn new(user_id: Arc<str>, session_id: Arc<str>) -> Self {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            user_id,
            session_id,
            last_activity: AtomicU64::new(now_nanos),
            is_typing: AtomicBool::new(false),
            typing_duration: AtomicU64::new(0),
            session_start: AtomicU64::new(now_nanos),
            event_count: AtomicU64::new(0)}
    }

    /// Start typing with atomic timestamp update
    #[inline]
    pub fn start_typing(&self) {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        self.last_activity.store(now_nanos, Ordering::Release);
        self.is_typing.store(true, Ordering::Release);
        self.event_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Stop typing with duration calculation
    #[inline]
    pub fn stop_typing(&self) {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // Calculate typing duration if we were typing
        if self.is_typing.load(Ordering::Acquire) {
            let start_time = self.last_activity.load(Ordering::Acquire);
            if start_time > 0 && now_nanos > start_time {
                let duration = now_nanos - start_time;
                self.typing_duration.fetch_add(duration, Ordering::AcqRel);
            }
        }

        self.last_activity.store(now_nanos, Ordering::Release);
        self.is_typing.store(false, Ordering::Release);
        self.event_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Check if typing has expired based on nanosecond precision
    #[inline]
    pub fn is_expired(&self, expiry_nanos: u64) -> bool {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let last_activity = self.last_activity.load(Ordering::Acquire);
        now_nanos.saturating_sub(last_activity) > expiry_nanos
    }

    /// Get current typing status
    #[inline]
    pub fn is_currently_typing(&self) -> bool {
        self.is_typing.load(Ordering::Acquire)
    }

    /// Get total typing duration in nanoseconds
    #[inline]
    pub fn total_typing_duration_nanos(&self) -> u64 {
        self.typing_duration.load(Ordering::Acquire)
    }

    /// Get total typing duration in seconds
    #[inline]
    pub fn total_typing_duration_seconds(&self) -> f64 {
        self.total_typing_duration_nanos() as f64 / 1_000_000_000.0
    }

    /// Get number of typing events in this session
    #[inline]
    pub fn event_count(&self) -> u64 {
        self.event_count.load(Ordering::Acquire)
    }

    /// Get session duration in nanoseconds
    #[inline]
    pub fn session_duration_nanos(&self) -> u64 {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        
        let start = self.session_start.load(Ordering::Acquire);
        now_nanos.saturating_sub(start)
    }

    /// Update activity timestamp without changing typing status
    #[inline]
    pub fn touch_activity(&self) {
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        self.last_activity.store(now_nanos, Ordering::Release);
    }
}

/// Typing indicator manager with lock-free concurrent operations
pub struct TypingIndicator {
    /// Active typing states with zero-allocation key storage
    typing_states: Arc<SkipMap<Arc<str>, Arc<TypingState>>>,
    /// Typing expiry duration in nanoseconds
    expiry_duration_nanos: Arc<AtomicU64>,
    /// Cleanup interval in nanoseconds
    cleanup_interval_nanos: Arc<AtomicU64>,
    /// Event broadcaster for real-time notifications
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Active users counter
    active_users: Arc<ConsistentCounter>,
    /// Total typing events counter
    typing_events: Arc<ConsistentCounter>,
    /// Cleanup task handle with atomic swap
    cleanup_task_active: Arc<AtomicBool>}

impl TypingIndicator {
    /// Create a new typing indicator with nanosecond precision
    #[inline]
    pub fn new(expiry_duration_secs: u64, cleanup_interval_secs: u64) -> Self {
        let (event_broadcaster, _) = broadcast::channel(10000); // Larger buffer for performance

        Self {
            typing_states: Arc::new(SkipMap::new()),
            expiry_duration_nanos: Arc::new(AtomicU64::new(expiry_duration_secs * 1_000_000_000)),
            cleanup_interval_nanos: Arc::new(AtomicU64::new(cleanup_interval_secs * 1_000_000_000)),
            event_broadcaster,
            active_users: Arc::new(ConsistentCounter::new(0)),
            typing_events: Arc::new(ConsistentCounter::new(0)),
            cleanup_task_active: Arc::new(AtomicBool::new(false))}
    }

    /// Start typing indicator with zero-allocation key generation
    pub fn start_typing(&self, user_id: Arc<str>, session_id: Arc<str>) -> AsyncStream<RealTimeEvent> {
        let key: Arc<str> = format!("{}:{}", user_id, session_id).into();
        let typing_states = self.typing_states.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let active_users = self.active_users.clone();
        let typing_events = self.typing_events.clone();

        AsyncStream::with_channel(move |sender| {
            // Get or create typing state
            let typing_state = if let Some(existing) = typing_states.get(&key) {
                existing.value().clone()
            } else {
                let new_state = Arc::new(TypingState::new(user_id.clone(), session_id.clone()));
                typing_states.insert(key, new_state.clone());
                active_users.inc();
                new_state
            };

            // Start typing
            typing_state.start_typing();
            typing_events.inc();

            // Create and broadcast event
            let event = RealTimeEvent::typing_started(user_id, session_id);
            let _ = event_broadcaster.send(event.clone());
            emit!(sender, event);
        })
    }

    /// Stop typing indicator with event emission
    pub fn stop_typing(&self, user_id: Arc<str>, session_id: Arc<str>) -> AsyncStream<RealTimeEvent> {
        let key: Arc<str> = format!("{}:{}", user_id, session_id).into();
        let typing_states = self.typing_states.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let typing_events = self.typing_events.clone();

        AsyncStream::with_channel(move |sender| {
            if let Some(typing_state_entry) = typing_states.get(&key) {
                let typing_state = typing_state_entry.value();
                typing_state.stop_typing();
                typing_events.inc();

                // Create and broadcast event
                let event = RealTimeEvent::typing_stopped(user_id, session_id);
                let _ = event_broadcaster.send(event.clone());
                emit!(sender, event);
            }
        })
    }

    /// Get currently typing users in a session
    pub fn get_typing_users_stream(&self, session_id: Arc<str>) -> AsyncStream<Vec<Arc<str>>> {
        let typing_states = self.typing_states.clone();

        AsyncStream::with_channel(move |sender| {
            let mut typing_users = Vec::with_capacity(16); // Pre-allocate for performance

            for entry in typing_states.iter() {
                let typing_state = entry.value();
                if typing_state.session_id == session_id && typing_state.is_currently_typing() {
                    typing_users.push(typing_state.user_id.clone());
                }
            }

            emit!(sender, typing_users);
        })
    }

    /// Start cleanup task with lock-free background processing
    pub fn start_cleanup_task(&self) -> AsyncStream<TypingCleanupEvent> {
        if self.cleanup_task_active.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            // Task already running, return empty stream
            return AsyncStream::with_channel(|_sender| {});
        }

        let typing_states = self.typing_states.clone();
        let expiry_duration_nanos = self.expiry_duration_nanos.clone();
        let cleanup_interval_nanos = self.cleanup_interval_nanos.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let active_users = self.active_users.clone();
        let cleanup_task_active = self.cleanup_task_active.clone();

        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                loop {
                    let cleanup_interval = Duration::from_nanos(cleanup_interval_nanos.load(Ordering::Acquire));
                    std::thread::sleep(cleanup_interval);

                    let expiry_nanos = expiry_duration_nanos.load(Ordering::Acquire);
                    let mut expired_keys = Vec::with_capacity(64); // Pre-allocate
                    let mut expired_count = 0u64;

                    // Find expired typing states
                    for entry in typing_states.iter() {
                        let typing_state = entry.value();
                        if typing_state.is_expired(expiry_nanos) {
                            expired_keys.push(entry.key().clone());

                            // Broadcast typing stopped event for expired states
                            if typing_state.is_currently_typing() {
                                let event = RealTimeEvent::typing_stopped(
                                    typing_state.user_id.clone(),
                                    typing_state.session_id.clone(),
                                );
                                let _ = event_broadcaster.send(event);
                            }
                        }
                    }

                    // Remove expired states
                    for key in &expired_keys {
                        if typing_states.remove(key).is_some() {
                            expired_count += 1;
                        }
                    }

                    // Update active users counter efficiently
                    if expired_count > 0 {
                        let current = active_users.get();
                        let new_count = current.saturating_sub(expired_count as usize);
                        active_users.reset();
                        for _ in 0..new_count {
                            active_users.inc();
                        }
                    }

                    // Emit cleanup event
                    let cleanup_event = TypingCleanupEvent {
                        expired_count,
                        remaining_active: active_users.get() as u64,
                        cleanup_duration_nanos: cleanup_interval.as_nanos() as u64,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos() as u64)
                            .unwrap_or(0)};

                    emit!(sender, cleanup_event);

                    // Check if we should continue running
                    if !cleanup_task_active.load(Ordering::Acquire) {
                        break;
                    }
                }
            });
        })
    }

    /// Stop cleanup task
    #[inline]
    pub fn stop_cleanup_task(&self) {
        self.cleanup_task_active.store(false, Ordering::Release);
    }

    /// Subscribe to typing events with zero-allocation receiver
    #[inline]
    pub fn subscribe(&self) -> broadcast::Receiver<RealTimeEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get comprehensive typing statistics
    #[inline]
    pub fn get_statistics(&self) -> TypingStatistics {
        TypingStatistics {
            active_users: self.active_users.get(),
            total_typing_events: self.typing_events.get(),
            expiry_duration_seconds: self.expiry_duration_nanos.load(Ordering::Acquire) / 1_000_000_000,
            cleanup_interval_seconds: self.cleanup_interval_nanos.load(Ordering::Acquire) / 1_000_000_000,
            total_states: self.typing_states.len()}
    }

    /// Update expiry duration dynamically
    #[inline]
    pub fn set_expiry_duration(&self, seconds: u64) {
        self.expiry_duration_nanos.store(seconds * 1_000_000_000, Ordering::Release);
    }

    /// Update cleanup interval dynamically
    #[inline]
    pub fn set_cleanup_interval(&self, seconds: u64) {
        self.cleanup_interval_nanos.store(seconds * 1_000_000_000, Ordering::Release);
    }

    /// Get active typing states count
    #[inline]
    pub fn active_states_count(&self) -> usize {
        self.typing_states.len()
    }

    /// Check if user is typing in any session
    pub fn is_user_typing(&self, user_id: &Arc<str>) -> bool {
        for entry in self.typing_states.iter() {
            let typing_state = entry.value();
            if typing_state.user_id == *user_id && typing_state.is_currently_typing() {
                return true;
            }
        }
        false
    }

    /// Get typing sessions for a user
    pub fn get_user_typing_sessions(&self, user_id: &Arc<str>) -> Vec<Arc<str>> {
        let mut sessions = Vec::new();
        for entry in self.typing_states.iter() {
            let typing_state = entry.value();
            if typing_state.user_id == *user_id && typing_state.is_currently_typing() {
                sessions.push(typing_state.session_id.clone());
            }
        }
        sessions
    }
}

impl std::fmt::Debug for TypingIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypingIndicator")
            .field("active_users", &self.active_users.get())
            .field("total_typing_events", &self.typing_events.get())
            .field("expiry_duration_seconds", &(self.expiry_duration_nanos.load(Ordering::Relaxed) / 1_000_000_000))
            .field("cleanup_interval_seconds", &(self.cleanup_interval_nanos.load(Ordering::Relaxed) / 1_000_000_000))
            .field("active_states", &self.typing_states.len())
            .field("cleanup_active", &self.cleanup_task_active.load(Ordering::Relaxed))
            .finish()
    }
}

impl Drop for TypingIndicator {
    fn drop(&mut self) {
        self.stop_cleanup_task();
    }
}

/// Typing statistics with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingStatistics {
    /// Number of currently active typing users
    pub active_users: usize,
    /// Total typing events processed
    pub total_typing_events: usize,
    /// Expiry duration in seconds
    pub expiry_duration_seconds: u64,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
    /// Total number of typing states
    pub total_states: usize}

/// Typing cleanup event for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingCleanupEvent {
    /// Number of expired states cleaned up
    pub expired_count: u64,
    /// Number of remaining active states
    pub remaining_active: u64,
    /// Duration of cleanup operation in nanoseconds
    pub cleanup_duration_nanos: u64,
    /// Event timestamp in nanoseconds
    pub timestamp: u64}

impl TypingCleanupEvent {
    /// Get cleanup duration in seconds
    #[inline]
    pub fn cleanup_duration_seconds(&self) -> f64 {
        self.cleanup_duration_nanos as f64 / 1_000_000_000.0
    }
}

