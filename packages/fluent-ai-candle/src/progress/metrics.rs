//! Metrics collection and aggregation for progress reporting

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Aggregated metrics for inference operations
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Total tokens generated
    pub total_tokens: u64,
    /// Average tokens per second
    pub avg_tokens_per_sec: f64,
    /// Peak tokens per second
    pub peak_tokens_per_sec: f64,
    /// Average inference latency (milliseconds)
    pub avg_latency_ms: f64,
    /// Peak inference latency (milliseconds)
    pub peak_latency_ms: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Total inference operations
    pub total_operations: u64,
    /// Failed operations count
    pub failed_operations: u64,
    /// Current memory usage (MB)
    pub memory_usage_mb: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
}

impl InferenceMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            avg_tokens_per_sec: 0.0,
            peak_tokens_per_sec: 0.0,
            avg_latency_ms: 0.0,
            peak_latency_ms: 0.0,
            cache_hit_rate: 0.0,
            total_operations: 0,
            failed_operations: 0,
            memory_usage_mb: 0.0,
            peak_memory_mb: 0.0,
        }
    }

    /// Update metrics with new generation data
    pub fn update_generation(&mut self, tokens: u64, duration: Duration, cache_hits: u64, cache_total: u64) {
        self.total_tokens += tokens;
        self.total_operations += 1;

        let latency_ms = duration.as_millis() as f64;
        let tokens_per_sec = if duration.as_secs_f64() > 0.0 {
            tokens as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        // Update averages
        let ops = self.total_operations as f64;
        self.avg_latency_ms = (self.avg_latency_ms * (ops - 1.0) + latency_ms) / ops;
        self.avg_tokens_per_sec = (self.avg_tokens_per_sec * (ops - 1.0) + tokens_per_sec) / ops;

        // Update peaks
        if tokens_per_sec > self.peak_tokens_per_sec {
            self.peak_tokens_per_sec = tokens_per_sec;
        }
        if latency_ms > self.peak_latency_ms {
            self.peak_latency_ms = latency_ms;
        }

        // Update cache hit rate
        if cache_total > 0 {
            let hit_rate = cache_hits as f64 / cache_total as f64;
            self.cache_hit_rate = (self.cache_hit_rate * (ops - 1.0) + hit_rate) / ops;
        }
    }

    /// Update memory usage
    pub fn update_memory(&mut self, current_mb: f64) {
        self.memory_usage_mb = current_mb;
        if current_mb > self.peak_memory_mb {
            self.peak_memory_mb = current_mb;
        }
    }

    /// Record failed operation
    pub fn record_failure(&mut self) {
        self.failed_operations += 1;
        self.total_operations += 1;
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 1.0;
        }
        let successful = self.total_operations - self.failed_operations;
        successful as f64 / self.total_operations as f64
    }

    /// Get efficiency score (tokens per second per MB)
    pub fn efficiency_score(&self) -> f64 {
        if self.memory_usage_mb > 0.0 {
            self.avg_tokens_per_sec / self.memory_usage_mb
        } else {
            0.0
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

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
        let session_info = SessionInfo {
            id: session_id.clone(),
            operation_name,
            start_time: Instant::now(),
            tokens_generated: AtomicU64::new(0),
            operations_count: AtomicU64::new(0),
            last_activity: parking_lot::RwLock::new(Instant::now()),
        };

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
            session.tokens_generated.fetch_add(tokens, Ordering::Relaxed);
            session.operations_count.fetch_add(1, Ordering::Relaxed);
            *session.last_activity.write() = Instant::now();
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

    /// Clear all metrics
    pub fn clear(&self) {
        self.total_tokens.store(0, Ordering::Relaxed);
        self.total_operations.store(0, Ordering::Relaxed);
        self.failed_operations.store(0, Ordering::Relaxed);
        self.active_sessions.store(0, Ordering::Relaxed);
        self.sessions.write().clear();
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Session information for tracking individual operations
#[derive(Debug)]
pub struct SessionInfo {
    /// Session identifier
    pub id: String,
    /// Human-readable operation name
    pub operation_name: String,
    /// Session start time
    pub start_time: Instant,
    /// Tokens generated in this session
    pub tokens_generated: AtomicU64,
    /// Number of operations in this session
    pub operations_count: AtomicU64,
    /// Last activity timestamp
    pub last_activity: parking_lot::RwLock<Instant>,
}

impl SessionInfo {
    /// Get session duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get tokens generated
    pub fn tokens(&self) -> u64 {
        self.tokens_generated.load(Ordering::Relaxed)
    }

    /// Get operations count
    pub fn operations(&self) -> u64 {
        self.operations_count.load(Ordering::Relaxed)
    }

    /// Get time since last activity
    pub fn idle_time(&self) -> Duration {
        self.last_activity.read().elapsed()
    }

    /// Check if session is active (activity within last 30 seconds)
    pub fn is_active(&self) -> bool {
        self.idle_time() < Duration::from_secs(30)
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