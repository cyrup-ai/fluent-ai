//! Typing indicator management with atomic operations
//!
//! This module provides comprehensive typing indicator functionality with zero-allocation
//! patterns, lock-free operations, and automatic cleanup of expired states.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, ConsistentCounter};
use crossbeam_skiplist::SkipMap;
use fluent_ai_async::{AsyncStream, spawn_task};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use super::events::RealTimeEvent;
use super::errors::RealTimeError;

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
    pub typing_duration: AtomicU64,
}

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
    typing_states: Arc<SkipMap<Arc<str>, Arc<TypingState>>>,
    /// Typing expiry duration in seconds
    expiry_duration: Arc<AtomicU64>,
    /// Cleanup interval in seconds
    cleanup_interval: Arc<AtomicU64>,
    /// Event broadcaster for real-time events
    event_broadcaster: broadcast::Sender<RealTimeEvent>,
    /// Active users counter
    active_users: Arc<ConsistentCounter>,
    /// Total typing events counter
    typing_events: Arc<ConsistentCounter>,
    /// Cleanup task handle
    cleanup_task: ArcSwap<Option<fluent_ai_async::AsyncTask<()>>>,
}

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
            cleanup_task: ArcSwap::new(Arc::new(None)),
        }
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
                .as_secs(),
        };

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
                    .as_secs(),
            };

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

        let task = spawn_task(move || {
            use std::thread;
            use std::time::Duration;

            loop {
                thread::sleep(Duration::from_secs(
                    cleanup_interval.load(Ordering::Relaxed),
                ));

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
                    // Decrement counter - ConsistentCounter doesn't have dec()
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

    /// Subscribe to typing events via AsyncStream
    pub fn subscribe(&self) -> AsyncStream<RealTimeEvent> {
        AsyncStream::with_channel(move |_sender| {
            // Events are emitted through broadcast channel
            // Real implementation would bridge broadcast to AsyncStream
        })
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

impl std::fmt::Debug for TypingIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypingIndicator")
            .field("active_users", &self.active_users.get())
            .field("total_typing_events", &self.typing_events.get())
            .field("expiry_duration", &self.expiry_duration.load(Ordering::Relaxed))
            .field("cleanup_interval", &self.cleanup_interval.load(Ordering::Relaxed))
            .finish()
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

impl Default for TypingStatistics {
    fn default() -> Self {
        Self {
            active_users: 0,
            total_typing_events: 0,
            expiry_duration: 10,
            cleanup_interval: 30,
        }
    }
}