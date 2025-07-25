//! Session information tracking

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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
    pub last_activity: parking_lot::RwLock<Instant>}

impl SessionInfo {
    /// Create new session info
    pub fn new(id: String, operation_name: String) -> Self {
        let now = Instant::now();
        Self {
            id,
            operation_name,
            start_time: now,
            tokens_generated: AtomicU64::new(0),
            operations_count: AtomicU64::new(0),
            last_activity: parking_lot::RwLock::new(now)}
    }

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

    /// Update activity timestamp
    pub fn update_activity(&self) {
        *self.last_activity.write() = Instant::now();
    }

    /// Add tokens to session
    pub fn add_tokens(&self, count: u64) {
        self.tokens_generated.fetch_add(count, Ordering::Relaxed);
        self.operations_count.fetch_add(1, Ordering::Relaxed);
        self.update_activity();
    }

    /// Increment operations count
    pub fn increment_operations(&self) {
        self.operations_count.fetch_add(1, Ordering::Relaxed);
        self.update_activity();
    }

    /// Get average tokens per second for this session
    pub fn avg_tokens_per_sec(&self) -> f64 {
        let duration = self.duration().as_secs_f64();
        if duration > 0.0 {
            self.tokens() as f64 / duration
        } else {
            0.0
        }
    }

    /// Get operations per second for this session
    pub fn operations_per_sec(&self) -> f64 {
        let duration = self.duration().as_secs_f64();
        if duration > 0.0 {
            self.operations() as f64 / duration
        } else {
            0.0
        }
    }

    /// Get session status summary
    pub fn status_summary(&self) -> SessionStatus {
        SessionStatus {
            id: self.id.clone(),
            operation_name: self.operation_name.clone(),
            duration_seconds: self.duration().as_secs(),
            tokens_generated: self.tokens(),
            operations_count: self.operations(),
            is_active: self.is_active(),
            avg_tokens_per_sec: self.avg_tokens_per_sec(),
            operations_per_sec: self.operations_per_sec()}
    }
}

/// Session status summary
#[derive(Debug, Clone)]
pub struct SessionStatus {
    pub id: String,
    pub operation_name: String,
    pub duration_seconds: u64,
    pub tokens_generated: u64,
    pub operations_count: u64,
    pub is_active: bool,
    pub avg_tokens_per_sec: f64,
    pub operations_per_sec: f64}

impl SessionStatus {
    /// Check if session is high performance
    pub fn is_high_performance(&self) -> bool {
        self.avg_tokens_per_sec > 50.0 && self.operations_per_sec > 10.0
    }

    /// Check if session is long-running
    pub fn is_long_running(&self) -> bool {
        self.duration_seconds > 300 // 5 minutes
    }

    /// Get performance category
    pub fn performance_category(&self) -> PerformanceCategory {
        if self.avg_tokens_per_sec > 100.0 {
            PerformanceCategory::Excellent
        } else if self.avg_tokens_per_sec > 50.0 {
            PerformanceCategory::Good
        } else if self.avg_tokens_per_sec > 20.0 {
            PerformanceCategory::Average
        } else {
            PerformanceCategory::Poor
        }
    }
}

/// Performance categories for sessions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceCategory {
    Excellent,
    Good,
    Average,
    Poor}

impl PerformanceCategory {
    /// Get category description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent (>100 tok/s)",
            Self::Good => "Good (50-100 tok/s)",
            Self::Average => "Average (20-50 tok/s)",
            Self::Poor => "Poor (<20 tok/s)"}
    }

    /// Get category color (for UI)
    pub fn color(&self) -> &'static str {
        match self {
            Self::Excellent => "green",
            Self::Good => "blue",
            Self::Average => "yellow",
            Self::Poor => "red"}
    }
}