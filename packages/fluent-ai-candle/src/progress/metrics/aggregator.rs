//! Thread-safe metrics aggregator

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use std::collections::HashMap;

use super::session::SessionInfo;

/// Thread-safe metrics aggregator with atomic operations
pub struct MetricsAggregator {
    /// Atomic counter for total tokens
    total_tokens: AtomicU64,
    /// Atomic counter for total operations
    total_operations: AtomicU64,
    /// Atomic counter for failed operations
    failed_operations: AtomicU64,
    /// Current session count
    active_sessions: AtomicUsize,
    /// Session information storage
    sessions: Arc<parking_lot::RwLock<HashMap<String, SessionInfo>>>,
    /// Start time for overall metrics
    start_time: Instant,
}

impl MetricsAggregator {
    /// Create new metrics aggregator
    pub fn new() -> Self {
        Self {
            total_tokens: AtomicU64::new(0),
            total_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            active_sessions: AtomicUsize::new(0),
            sessions: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Record token generation
    pub fn record_tokens(&self, count: u64) {
        self.total_tokens.fetch_add(count, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record failed operation
    pub fn record_failure(&self) {
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Start new session
    pub fn start_session(&self, session_id: String, operation_name: String) {
        let session_info = SessionInfo::new(session_id.clone(), operation_name);

        {
            let mut sessions = self.sessions.write();
            sessions.insert(session_id, session_info);
        }

        self.active_sessions.fetch_add(1, Ordering::Relaxed);
    }

    /// End session
    pub fn end_session(&self, session_id: &str) -> Option<SessionInfo> {
        let session = {
            let mut sessions = self.sessions.write();
            sessions.remove(session_id)
        };

        if session.is_some() {
            self.active_sessions.fetch_sub(1, Ordering::Relaxed);
        }

        session
    }

    /// Update session activity
    pub fn update_session_activity(&self, session_id: &str, tokens: u64) {
        if let Some(session) = self.sessions.read().get(session_id) {
            session.add_tokens(tokens);
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> AggregatorStats {
        let total_tokens = self.total_tokens.load(Ordering::Relaxed);
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        let failed_ops = self.failed_operations.load(Ordering::Relaxed);
        let active_sessions = self.active_sessions.load(Ordering::Relaxed);

        let uptime = self.start_time.elapsed();
        let avg_tokens_per_sec = if uptime.as_secs_f64() > 0.0 {
            total_tokens as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };

        let success_rate = if total_ops > 0 {
            (total_ops - failed_ops) as f64 / total_ops as f64
        } else {
            1.0
        };

        AggregatorStats {
            total_tokens,
            total_operations: total_ops,
            failed_operations: failed_ops,
            active_sessions,
            uptime_seconds: uptime.as_secs(),
            avg_tokens_per_sec,
            success_rate,
        }
    }

    /// Get session count
    pub fn session_count(&self) -> usize {
        self.active_sessions.load(Ordering::Relaxed)
    }

    /// Get all active sessions
    pub fn get_active_sessions(&self) -> Vec<super::session::SessionStatus> {
        self.sessions
            .read()
            .values()
            .filter(|session| session.is_active())
            .map(|session| session.status_summary())
            .collect()
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Option<super::session::SessionStatus> {
        self.sessions
            .read()
            .get(session_id)
            .map(|session| session.status_summary())
    }

    /// Clear all metrics
    pub fn clear(&self) {
        self.total_tokens.store(0, Ordering::Relaxed);
        self.total_operations.store(0, Ordering::Relaxed);
        self.failed_operations.store(0, Ordering::Relaxed);
        self.active_sessions.store(0, Ordering::Relaxed);
        self.sessions.write().clear();
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> PerformanceSummary {
        let stats = self.get_stats();
        let active_sessions = self.get_active_sessions();
        
        let high_perf_sessions = active_sessions
            .iter()
            .filter(|s| s.is_high_performance())
            .count();

        PerformanceSummary {
            total_throughput: stats.avg_tokens_per_sec,
            success_rate: stats.success_rate,
            active_sessions: stats.active_sessions,
            high_performance_sessions: high_perf_sessions,
            uptime_hours: stats.uptime_seconds as f64 / 3600.0,
        }
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from the metrics aggregator
#[derive(Debug, Clone)]
pub struct AggregatorStats {
    /// Total tokens processed
    pub total_tokens: u64,
    /// Total operations performed
    pub total_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Currently active sessions
    pub active_sessions: usize,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Average tokens per second
    pub avg_tokens_per_sec: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

impl AggregatorStats {
    /// Get failure rate
    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate
    }

    /// Check if performance is good
    pub fn is_performing_well(&self) -> bool {
        self.success_rate > 0.95 && self.avg_tokens_per_sec > 20.0
    }
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_throughput: f64,
    pub success_rate: f64,
    pub active_sessions: usize,
    pub high_performance_sessions: usize,
    pub uptime_hours: f64,
}