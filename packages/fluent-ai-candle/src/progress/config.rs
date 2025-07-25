//! Configuration for progress reporting

use std::time::Duration;

/// Configuration for ProgressHub reporter
#[derive(Debug, Clone)]
pub struct ProgressHubConfig {
    /// Enable real-time progress updates
    pub enable_realtime: bool,
    /// Update interval for progress reports
    pub update_interval_ms: u64,
    /// Maximum number of concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Buffer size for progress events
    pub event_buffer_size: usize,
    /// Enable detailed metrics collection
    pub enable_detailed_metrics: bool,
    /// Timeout for individual progress operations
    pub operation_timeout: Duration,
    /// Enable compression for progress data
    pub enable_compression: bool,
    /// Maximum memory usage for progress tracking (MB)
    pub max_memory_usage_mb: f64}

impl ProgressHubConfig {
    /// Create new configuration with sensible defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            enable_realtime: true,
            update_interval_ms: 10,
            max_concurrent_sessions: 100,
            event_buffer_size: 1000,
            enable_detailed_metrics: false,
            operation_timeout: Duration::from_millis(100),
            enable_compression: false,
            max_memory_usage_mb: 50.0}
    }

    /// Create configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            enable_realtime: true,
            update_interval_ms: 50,
            max_concurrent_sessions: 1000,
            event_buffer_size: 10000,
            enable_detailed_metrics: true,
            operation_timeout: Duration::from_secs(5),
            enable_compression: true,
            max_memory_usage_mb: 200.0}
    }

    /// Create configuration for minimal resource usage
    pub fn minimal() -> Self {
        Self {
            enable_realtime: false,
            update_interval_ms: 1000,
            max_concurrent_sessions: 10,
            event_buffer_size: 100,
            enable_detailed_metrics: false,
            operation_timeout: Duration::from_secs(30),
            enable_compression: true,
            max_memory_usage_mb: 10.0}
    }

    /// Set update interval
    pub fn with_update_interval(mut self, interval_ms: u64) -> Self {
        self.update_interval_ms = interval_ms;
        self
    }

    /// Set maximum concurrent sessions
    pub fn with_max_sessions(mut self, max_sessions: usize) -> Self {
        self.max_concurrent_sessions = max_sessions;
        self
    }

    /// Enable or disable detailed metrics
    pub fn with_detailed_metrics(mut self, enable: bool) -> Self {
        self.enable_detailed_metrics = enable;
        self
    }

    /// Set event buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.event_buffer_size = size;
        self
    }

    /// Set operation timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.operation_timeout = timeout;
        self
    }

    /// Enable or disable compression
    pub fn with_compression(mut self, enable: bool) -> Self {
        self.enable_compression = enable;
        self
    }

    /// Set maximum memory usage
    pub fn with_max_memory(mut self, memory_mb: f64) -> Self {
        self.max_memory_usage_mb = memory_mb;
        self
    }

    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.update_interval_ms == 0 {
            return Err("Update interval must be greater than 0".to_string());
        }

        if self.max_concurrent_sessions == 0 {
            return Err("Max concurrent sessions must be greater than 0".to_string());
        }

        if self.event_buffer_size == 0 {
            return Err("Event buffer size must be greater than 0".to_string());
        }

        if self.max_memory_usage_mb <= 0.0 {
            return Err("Max memory usage must be greater than 0".to_string());
        }

        if self.operation_timeout.is_zero() {
            return Err("Operation timeout must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Get update interval as Duration
    pub fn update_interval(&self) -> Duration {
        Duration::from_millis(self.update_interval_ms)
    }

    /// Check if configuration is optimized for performance
    pub fn is_performance_optimized(&self) -> bool {
        self.enable_realtime 
            && self.update_interval_ms <= 50
            && self.max_concurrent_sessions >= 100
            && !self.enable_compression
    }

    /// Check if configuration is optimized for resource efficiency
    pub fn is_resource_efficient(&self) -> bool {
        !self.enable_realtime
            || (self.update_interval_ms >= 500
                && self.max_concurrent_sessions <= 50
                && self.enable_compression
                && self.max_memory_usage_mb <= 50.0)
    }
}

impl Default for ProgressHubConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            update_interval_ms: 100,
            max_concurrent_sessions: 50,
            event_buffer_size: 1000,
            enable_detailed_metrics: true,
            operation_timeout: Duration::from_secs(10),
            enable_compression: false,
            max_memory_usage_mb: 100.0}
    }
}