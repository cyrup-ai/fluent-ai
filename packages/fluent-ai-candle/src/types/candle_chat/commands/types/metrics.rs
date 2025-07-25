//! Execution metrics and performance tracking
//!
//! Comprehensive metrics collection for command execution performance,
//! memory usage, and operational statistics with zero-allocation patterns.

use std::collections::HashMap;

/// Execution metrics for command performance tracking
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Execution duration in nanoseconds
    pub duration_nanos: u64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in nanoseconds
    pub cpu_time_nanos: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Total commands executed
    pub total_commands: u64,
    /// Most popular commands with counts
    pub popular_commands: HashMap<String, u64>,
    /// Total execution time across all commands
    pub total_execution_time: u64,
    /// Number of successful commands
    pub successful_commands: u64,
    /// Number of failed commands
    pub failed_commands: u64,
    /// Error counts by type
    pub error_counts: HashMap<String, u64>,
    /// Average execution time in nanoseconds
    pub average_execution_time: u64}

impl ExecutionMetrics {
    /// Create new execution metrics
    #[inline]
    pub fn new() -> Self {
        Self {
            duration_nanos: 0,
            memory_bytes: 0,
            cpu_time_nanos: 0,
            allocations: 0,
            peak_memory_bytes: 0,
            total_commands: 0,
            popular_commands: HashMap::new(),
            total_execution_time: 0,
            successful_commands: 0,
            failed_commands: 0,
            error_counts: HashMap::new(),
            average_execution_time: 0}
    }

    /// Calculate duration in milliseconds
    #[inline]
    pub fn duration_ms(&self) -> f64 {
        self.duration_nanos as f64 / 1_000_000.0
    }

    /// Calculate memory usage in MB
    #[inline]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl Default for ExecutionMetrics {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}