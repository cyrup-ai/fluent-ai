//! Context Events for Real-time Streaming Monitoring
//!
//! Zero-allocation event types for context operation monitoring and performance tracking.

use std::time::SystemTime;
use serde::{Deserialize, Serialize};

/// Context events for real-time streaming monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextEvent {
    /// Provider lifecycle events
    ProviderStarted {
        provider_type: String,
        provider_id: String,
        timestamp: SystemTime},
    ProviderStopped {
        provider_type: String,
        provider_id: String,
        timestamp: SystemTime},

    /// Operation events
    ContextLoadStarted {
        context_type: String,
        source: String,
        timestamp: SystemTime},
    ContextLoadCompleted {
        context_type: String,
        source: String,
        documents_loaded: usize,
        duration_nanos: u64,
        timestamp: SystemTime},
    ContextLoadFailed {
        context_type: String,
        source: String,
        error: String,
        timestamp: SystemTime},

    /// Memory integration events
    MemoryCreated {
        memory_id: String,
        content_hash: String,
        timestamp: SystemTime},
    MemorySearchCompleted {
        query: String,
        results_count: usize,
        duration_nanos: u64,
        timestamp: SystemTime},

    /// Performance events
    PerformanceThresholdBreached {
        metric: String,
        threshold: f64,
        actual: f64,
        timestamp: SystemTime},

    /// Validation events
    ValidationFailed {
        validation_type: String,
        error: String,
        timestamp: SystemTime}}

impl ContextEvent {
    /// Get event timestamp
    pub fn timestamp(&self) -> SystemTime {
        match self {
            ContextEvent::ProviderStarted { timestamp, .. } => *timestamp,
            ContextEvent::ProviderStopped { timestamp, .. } => *timestamp,
            ContextEvent::ContextLoadStarted { timestamp, .. } => *timestamp,
            ContextEvent::ContextLoadCompleted { timestamp, .. } => *timestamp,
            ContextEvent::ContextLoadFailed { timestamp, .. } => *timestamp,
            ContextEvent::MemoryCreated { timestamp, .. } => *timestamp,
            ContextEvent::MemorySearchCompleted { timestamp, .. } => *timestamp,
            ContextEvent::PerformanceThresholdBreached { timestamp, .. } => *timestamp,
            ContextEvent::ValidationFailed { timestamp, .. } => *timestamp}
    }

    /// Get event type as string
    pub fn event_type(&self) -> &'static str {
        match self {
            ContextEvent::ProviderStarted { .. } => "provider_started",
            ContextEvent::ProviderStopped { .. } => "provider_stopped",
            ContextEvent::ContextLoadStarted { .. } => "context_load_started",
            ContextEvent::ContextLoadCompleted { .. } => "context_load_completed",
            ContextEvent::ContextLoadFailed { .. } => "context_load_failed",
            ContextEvent::MemoryCreated { .. } => "memory_created",
            ContextEvent::MemorySearchCompleted { .. } => "memory_search_completed",
            ContextEvent::PerformanceThresholdBreached { .. } => "performance_threshold_breached",
            ContextEvent::ValidationFailed { .. } => "validation_failed"}
    }
}