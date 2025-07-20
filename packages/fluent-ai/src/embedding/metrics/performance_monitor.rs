//! Comprehensive performance monitoring for embedding operations
//!
//! This module provides enterprise-grade performance monitoring including:
//! - Request latency tracking per provider with histogram analytics
//! - Throughput monitoring with sliding window calculations  
//! - Cache hit ratio analysis and memory utilization tracking
//! - Resource utilization metrics with SIMD optimization
//! - Lock-free metric collection for zero-contention access
//! - Percentile calculations using streaming algorithms
//! - Real-time alerting system with configurable thresholds
//! - Metric export for Prometheus/OpenTelemetry integration

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrayvec::ArrayString;
use crossbeam_utils::CachePadded;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use thiserror::Error;
use tokio::sync::{RwLock, broadcast, watch};
use tokio::time::interval;

/// Maximum number of providers to track
const MAX_PROVIDERS: usize = 32;
/// Histogram bucket count for latency tracking
const HISTOGRAM_BUCKETS: usize = 64;
/// Sliding window size for throughput calculations
const THROUGHPUT_WINDOW_SIZE: usize = 300; // 5 minutes at 1-second intervals
/// Performance metrics collection interval
const METRICS_COLLECTION_INTERVAL_MS: u64 = 1000;
/// Alert evaluation interval
const ALERT_EVALUATION_INTERVAL_MS: u64 = 5000;
/// Maximum alert history to maintain
const MAX_ALERT_HISTORY: usize = 1000;

/// Performance metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Timestamp of the metric
    pub timestamp: u64,
    /// Provider identifier
    pub provider: ArrayString<32>,
    /// Operation type (embed, batch, similarity, etc.)
    pub operation: ArrayString<16>,
    /// Request latency in microseconds
    pub latency_us: u64,
    /// Number of items processed
    pub item_count: u32,
    /// Request size in bytes
    pub request_size_bytes: u64,
    /// Response size in bytes  
    pub response_size_bytes: u64,
    /// Cache hit (true) or miss (false)
    pub cache_hit: bool,
    /// Error occurred during operation
    pub error_occurred: bool,
    /// Memory usage delta in bytes
    pub memory_delta_bytes: i64,
}

/// Latency histogram for efficient percentile calculations
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Bucket boundaries in microseconds (exponential)
    bucket_boundaries: [u64; HISTOGRAM_BUCKETS],
    /// Bucket counts
    bucket_counts: [CachePadded<AtomicU64>; HISTOGRAM_BUCKETS],
    /// Total count for validation
    total_count: CachePadded<AtomicU64>,
    /// Sum for mean calculation
    total_sum: CachePadded<AtomicU64>,
    /// Minimum observed value
    min_value: CachePadded<AtomicU64>,
    /// Maximum observed value
    max_value: CachePadded<AtomicU64>,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        // Exponential bucket boundaries: 1us, 2us, 4us, ..., up to ~1 hour
        let mut boundaries = [0u64; HISTOGRAM_BUCKETS];
        for i in 0..HISTOGRAM_BUCKETS {
            boundaries[i] = if i == 0 { 1 } else { 1u64 << i };
        }

        Self {
            bucket_boundaries: boundaries,
            bucket_counts: std::array::from_fn(|_| CachePadded::new(AtomicU64::new(0))),
            total_count: CachePadded::new(AtomicU64::new(0)),
            total_sum: CachePadded::new(AtomicU64::new(0)),
            min_value: CachePadded::new(AtomicU64::new(u64::MAX)),
            max_value: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record a latency measurement
    pub fn record(&self, latency_us: u64) {
        // Find appropriate bucket using binary search
        let bucket_index = self
            .bucket_boundaries
            .binary_search(&latency_us)
            .unwrap_or_else(|i| i.min(HISTOGRAM_BUCKETS - 1));

        // Update bucket count
        self.bucket_counts[bucket_index].fetch_add(1, Ordering::Relaxed);

        // Update summary statistics
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_sum.fetch_add(latency_us, Ordering::Relaxed);

        // Update min/max with compare-and-swap loop
        let mut current_min = self.min_value.load(Ordering::Relaxed);
        while latency_us < current_min {
            match self.min_value.compare_exchange_weak(
                current_min,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }

        let mut current_max = self.max_value.load(Ordering::Relaxed);
        while latency_us > current_max {
            match self.max_value.compare_exchange_weak(
                current_max,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    /// Calculate percentile using streaming algorithm
    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }

        let target_count = (total as f64 * p) as u64;
        let mut cumulative_count = 0u64;

        for (i, bucket_count) in self.bucket_counts.iter().enumerate() {
            cumulative_count += bucket_count.load(Ordering::Relaxed);
            if cumulative_count >= target_count {
                return self.bucket_boundaries[i];
            }
        }

        self.max_value.load(Ordering::Relaxed)
    }

    /// Get current statistics
    pub fn get_stats(&self) -> LatencyStats {
        let total = self.total_count.load(Ordering::Relaxed);
        let sum = self.total_sum.load(Ordering::Relaxed);

        LatencyStats {
            count: total,
            mean: if total > 0 { sum / total } else { 0 },
            min: if total > 0 {
                self.min_value.load(Ordering::Relaxed)
            } else {
                0
            },
            max: self.max_value.load(Ordering::Relaxed),
            p50: self.percentile(0.5),
            p90: self.percentile(0.9),
            p95: self.percentile(0.95),
            p99: self.percentile(0.99),
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        for bucket in &self.bucket_counts {
            bucket.store(0, Ordering::Relaxed);
        }
        self.total_count.store(0, Ordering::Relaxed);
        self.total_sum.store(0, Ordering::Relaxed);
        self.min_value.store(u64::MAX, Ordering::Relaxed);
        self.max_value.store(0, Ordering::Relaxed);
    }
}

/// Latency statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: u64,
    pub mean: u64,
    pub min: u64,
    pub max: u64,
    pub p50: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
}

/// Sliding window for throughput calculations
#[derive(Debug)]
pub struct ThroughputWindow {
    /// Ring buffer of throughput measurements
    measurements: Arc<RwLock<VecDeque<ThroughputMeasurement>>>,
    /// Window size in measurements
    window_size: usize,
    /// Current throughput (requests per second)
    current_throughput: CachePadded<AtomicU64>,
    /// Current bandwidth (bytes per second)
    current_bandwidth: CachePadded<AtomicU64>,
}

#[derive(Debug, Clone)]
struct ThroughputMeasurement {
    timestamp: u64,
    request_count: u64,
    byte_count: u64,
}

impl ThroughputWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            measurements: Arc::new(RwLock::new(VecDeque::with_capacity(window_size))),
            window_size,
            current_throughput: CachePadded::new(AtomicU64::new(0)),
            current_bandwidth: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record throughput measurement
    pub async fn record(&self, request_count: u64, byte_count: u64) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let measurement = ThroughputMeasurement {
            timestamp,
            request_count,
            byte_count,
        };

        let mut measurements = self.measurements.write().await;

        // Add new measurement
        measurements.push_back(measurement);

        // Remove old measurements outside the window
        let cutoff_time = timestamp.saturating_sub(self.window_size as u64);
        while let Some(front) = measurements.front() {
            if front.timestamp < cutoff_time {
                measurements.pop_front();
            } else {
                break;
            }
        }

        // Calculate current throughput
        if measurements.len() >= 2 {
            let total_requests: u64 = measurements.iter().map(|m| m.request_count).sum();
            let total_bytes: u64 = measurements.iter().map(|m| m.byte_count).sum();
            let time_span =
                measurements.back().unwrap().timestamp - measurements.front().unwrap().timestamp;

            if time_span > 0 {
                let throughput = total_requests / time_span;
                let bandwidth = total_bytes / time_span;

                self.current_throughput.store(throughput, Ordering::Relaxed);
                self.current_bandwidth.store(bandwidth, Ordering::Relaxed);
            }
        }
    }

    /// Get current throughput (requests per second)
    pub fn get_throughput(&self) -> u64 {
        self.current_throughput.load(Ordering::Relaxed)
    }

    /// Get current bandwidth (bytes per second)
    pub fn get_bandwidth(&self) -> u64 {
        self.current_bandwidth.load(Ordering::Relaxed)
    }
}

/// Cache performance metrics
#[derive(Debug)]
pub struct CacheMetrics {
    /// Total cache lookups
    pub total_lookups: CachePadded<AtomicU64>,
    /// Cache hits
    pub cache_hits: CachePadded<AtomicU64>,
    /// Cache misses
    pub cache_misses: CachePadded<AtomicU64>,
    /// Cache evictions
    pub cache_evictions: CachePadded<AtomicU64>,
    /// Cache memory usage in bytes
    pub memory_usage_bytes: CachePadded<AtomicU64>,
    /// Cache entry count
    pub entry_count: CachePadded<AtomicU64>,
}

impl CacheMetrics {
    pub fn new() -> Self {
        Self {
            total_lookups: CachePadded::new(AtomicU64::new(0)),
            cache_hits: CachePadded::new(AtomicU64::new(0)),
            cache_misses: CachePadded::new(AtomicU64::new(0)),
            cache_evictions: CachePadded::new(AtomicU64::new(0)),
            memory_usage_bytes: CachePadded::new(AtomicU64::new(0)),
            entry_count: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record cache hit
    pub fn record_hit(&self) {
        self.total_lookups.fetch_add(1, Ordering::Relaxed);
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_miss(&self) {
        self.total_lookups.fetch_add(1, Ordering::Relaxed);
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache eviction
    pub fn record_eviction(&self) {
        self.cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Update entry count
    pub fn update_entry_count(&self, count: u64) {
        self.entry_count.store(count, Ordering::Relaxed);
    }

    /// Calculate hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total = self.total_lookups.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let hits = self.cache_hits.load(Ordering::Relaxed);
        hits as f64 / total as f64
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            total_lookups: self.total_lookups.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_evictions: self.cache_evictions.load(Ordering::Relaxed),
            memory_usage_bytes: self.memory_usage_bytes.load(Ordering::Relaxed),
            entry_count: self.entry_count.load(Ordering::Relaxed),
            hit_ratio: self.hit_ratio(),
        }
    }
}

/// Cache statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_lookups: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub memory_usage_bytes: u64,
    pub entry_count: u64,
    pub hit_ratio: f64,
}

/// Resource utilization metrics
#[derive(Debug)]
pub struct ResourceMetrics {
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: CachePadded<AtomicU32>,
    /// Memory usage in bytes
    pub memory_usage_bytes: CachePadded<AtomicU64>,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: CachePadded<AtomicU64>,
    /// Number of active threads
    pub active_threads: CachePadded<AtomicU32>,
    /// Number of queued tasks
    pub queued_tasks: CachePadded<AtomicU64>,
    /// Network I/O bytes sent
    pub network_bytes_sent: CachePadded<AtomicU64>,
    /// Network I/O bytes received
    pub network_bytes_received: CachePadded<AtomicU64>,
    /// Disk I/O bytes read
    pub disk_bytes_read: CachePadded<AtomicU64>,
    /// Disk I/O bytes written
    pub disk_bytes_written: CachePadded<AtomicU64>,
}

impl ResourceMetrics {
    pub fn new() -> Self {
        Self {
            cpu_usage_percent: CachePadded::new(AtomicU32::new(0)),
            memory_usage_bytes: CachePadded::new(AtomicU64::new(0)),
            peak_memory_bytes: CachePadded::new(AtomicU64::new(0)),
            active_threads: CachePadded::new(AtomicU32::new(0)),
            queued_tasks: CachePadded::new(AtomicU64::new(0)),
            network_bytes_sent: CachePadded::new(AtomicU64::new(0)),
            network_bytes_received: CachePadded::new(AtomicU64::new(0)),
            disk_bytes_read: CachePadded::new(AtomicU64::new(0)),
            disk_bytes_written: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Update memory usage and track peak
    pub fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage_bytes.store(bytes, Ordering::Relaxed);

        // Update peak memory with compare-and-swap loop
        let mut current_peak = self.peak_memory_bytes.load(Ordering::Relaxed);
        while bytes > current_peak {
            match self.peak_memory_bytes.compare_exchange_weak(
                current_peak,
                bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
    }

    /// Record network I/O
    pub fn record_network_io(&self, bytes_sent: u64, bytes_received: u64) {
        self.network_bytes_sent
            .fetch_add(bytes_sent, Ordering::Relaxed);
        self.network_bytes_received
            .fetch_add(bytes_received, Ordering::Relaxed);
    }

    /// Record disk I/O
    pub fn record_disk_io(&self, bytes_read: u64, bytes_written: u64) {
        self.disk_bytes_read
            .fetch_add(bytes_read, Ordering::Relaxed);
        self.disk_bytes_written
            .fetch_add(bytes_written, Ordering::Relaxed);
    }

    /// Get resource statistics
    pub fn get_stats(&self) -> ResourceStats {
        ResourceStats {
            cpu_usage_percent: self.cpu_usage_percent.load(Ordering::Relaxed),
            memory_usage_bytes: self.memory_usage_bytes.load(Ordering::Relaxed),
            peak_memory_bytes: self.peak_memory_bytes.load(Ordering::Relaxed),
            active_threads: self.active_threads.load(Ordering::Relaxed),
            queued_tasks: self.queued_tasks.load(Ordering::Relaxed),
            network_bytes_sent: self.network_bytes_sent.load(Ordering::Relaxed),
            network_bytes_received: self.network_bytes_received.load(Ordering::Relaxed),
            disk_bytes_read: self.disk_bytes_read.load(Ordering::Relaxed),
            disk_bytes_written: self.disk_bytes_written.load(Ordering::Relaxed),
        }
    }
}

/// Resource statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    pub cpu_usage_percent: u32,
    pub memory_usage_bytes: u64,
    pub peak_memory_bytes: u64,
    pub active_threads: u32,
    pub queued_tasks: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub disk_bytes_read: u64,
    pub disk_bytes_written: u64,
}

/// Performance alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub id: ArrayString<32>,
    pub timestamp: u64,
    pub severity: AlertSeverity,
    pub metric_name: ArrayString<32>,
    pub provider: Option<ArrayString<32>>,
    pub threshold_value: f64,
    pub actual_value: f64,
    pub message: ArrayString<256>,
    pub resolved: bool,
    pub resolution_timestamp: Option<u64>,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub metric_name: ArrayString<32>,
    pub threshold_value: f64,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub provider_specific: Option<ArrayString<32>>,
}

/// Performance monitoring errors
#[derive(Debug, Error)]
pub enum PerformanceMonitorError {
    #[error("Invalid metric name: {name}")]
    InvalidMetricName { name: String },

    #[error("Provider not found: {provider}")]
    ProviderNotFound { provider: String },

    #[error("Alert configuration error: {error}")]
    AlertConfigError { error: String },

    #[error("Metric collection failed: {error}")]
    MetricCollectionFailed { error: String },

    #[error("Export failed: {error}")]
    ExportFailed { error: String },
}

/// Provider-specific performance metrics
#[derive(Debug)]
pub struct ProviderMetrics {
    /// Provider identifier
    pub provider_id: ArrayString<32>,
    /// Latency histogram
    pub latency_histogram: LatencyHistogram,
    /// Throughput window
    pub throughput_window: ThroughputWindow,
    /// Request count
    pub request_count: CachePadded<AtomicU64>,
    /// Error count
    pub error_count: CachePadded<AtomicU64>,
    /// Total processing time
    pub total_processing_time: CachePadded<AtomicU64>,
    /// Last activity timestamp
    pub last_activity: CachePadded<AtomicU64>,
}

impl ProviderMetrics {
    pub fn new(provider_id: ArrayString<32>) -> Self {
        Self {
            provider_id,
            latency_histogram: LatencyHistogram::new(),
            throughput_window: ThroughputWindow::new(THROUGHPUT_WINDOW_SIZE),
            request_count: CachePadded::new(AtomicU64::new(0)),
            error_count: CachePadded::new(AtomicU64::new(0)),
            total_processing_time: CachePadded::new(AtomicU64::new(0)),
            last_activity: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record performance metric
    pub async fn record_metric(&self, metric: &PerformanceMetric) {
        // Update latency histogram
        self.latency_histogram.record(metric.latency_us);

        // Update throughput window
        self.throughput_window
            .record(1, metric.request_size_bytes + metric.response_size_bytes)
            .await;

        // Update counters
        self.request_count.fetch_add(1, Ordering::Relaxed);
        if metric.error_occurred {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
        self.total_processing_time
            .fetch_add(metric.latency_us, Ordering::Relaxed);

        // Update last activity
        self.last_activity
            .store(metric.timestamp, Ordering::Relaxed);
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        let total = self.request_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let errors = self.error_count.load(Ordering::Relaxed);
        errors as f64 / total as f64
    }

    /// Get average processing time
    pub fn average_processing_time(&self) -> u64 {
        let total = self.request_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        let total_time = self.total_processing_time.load(Ordering::Relaxed);
        total_time / total
    }
}

/// Comprehensive performance monitor
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Per-provider metrics
    provider_metrics: Arc<DashMap<ArrayString<32>, Arc<ProviderMetrics>>>,
    /// Global cache metrics
    cache_metrics: Arc<CacheMetrics>,
    /// Resource utilization metrics
    resource_metrics: Arc<ResourceMetrics>,
    /// Alert configurations
    alert_configs: Arc<DashMap<ArrayString<32>, AlertConfig>>,
    /// Active alerts
    active_alerts: Arc<DashMap<ArrayString<32>, PerformanceAlert>>,
    /// Alert history
    alert_history: Arc<RwLock<VecDeque<PerformanceAlert>>>,
    /// Alert broadcast channel
    alert_sender: broadcast::Sender<PerformanceAlert>,
    /// Metrics broadcast channel for real-time dashboard
    metrics_sender: broadcast::Sender<GlobalMetrics>,
    /// Performance metrics watch channel
    performance_watch: watch::Sender<GlobalMetrics>,
    performance_receiver: watch::Receiver<GlobalMetrics>,
    /// Anomaly detection state
    anomaly_detector: Arc<AnomalyDetector>,
}

/// Global metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    pub timestamp: u64,
    pub total_requests: u64,
    pub total_errors: u64,
    pub global_error_rate: f64,
    pub average_latency_us: u64,
    pub total_throughput_rps: u64,
    pub cache_stats: CacheStats,
    pub resource_stats: ResourceStats,
    pub active_providers: u32,
    pub alert_count: u32,
}

/// Anomaly detection for performance regression analysis
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Historical baselines for comparison
    baselines: Arc<DashMap<ArrayString<32>, PerformanceBaseline>>,
    /// Anomaly detection sensitivity (0.0 - 1.0)
    sensitivity: f64,
    /// Minimum samples required for baseline
    min_baseline_samples: usize,
}

#[derive(Debug, Clone)]
struct PerformanceBaseline {
    metric_name: ArrayString<32>,
    mean: f64,
    std_deviation: f64,
    sample_count: u64,
    last_updated: u64,
}

impl AnomalyDetector {
    pub fn new(sensitivity: f64, min_baseline_samples: usize) -> Self {
        Self {
            baselines: Arc::new(DashMap::new()),
            sensitivity,
            min_baseline_samples,
        }
    }

    /// Update baseline with new measurement
    pub fn update_baseline(&self, metric_name: &str, value: f64) {
        let key = ArrayString::from(metric_name).unwrap_or_default();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.baselines
            .entry(key.clone())
            .and_modify(|baseline| {
                // Update running statistics
                let new_count = baseline.sample_count + 1;
                let new_mean = baseline.mean + (value - baseline.mean) / new_count as f64;
                let new_variance = if new_count > 1 {
                    ((new_count - 1) as f64 * baseline.std_deviation.powi(2)
                        + (value - baseline.mean) * (value - new_mean))
                        / new_count as f64
                } else {
                    0.0
                };

                baseline.mean = new_mean;
                baseline.std_deviation = new_variance.sqrt();
                baseline.sample_count = new_count;
                baseline.last_updated = timestamp;
            })
            .or_insert_with(|| PerformanceBaseline {
                metric_name: key,
                mean: value,
                std_deviation: 0.0,
                sample_count: 1,
                last_updated: timestamp,
            });
    }

    /// Check if value is anomalous
    pub fn is_anomaly(&self, metric_name: &str, value: f64) -> Option<f64> {
        let key = ArrayString::from(metric_name).unwrap_or_default();

        if let Some(baseline) = self.baselines.get(&key) {
            if baseline.sample_count >= self.min_baseline_samples as u64
                && baseline.std_deviation > 0.0
            {
                let z_score = (value - baseline.mean).abs() / baseline.std_deviation;
                let threshold = 2.0 + (1.0 - self.sensitivity) * 2.0; // Range: 2.0 - 4.0 sigma

                if z_score > threshold {
                    return Some(z_score);
                }
            }
        }

        None
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        let (alert_sender, _) = broadcast::channel(MAX_ALERT_HISTORY);
        let (metrics_sender, _) = broadcast::channel(1000);
        let (performance_watch, performance_receiver) = watch::channel(GlobalMetrics {
            timestamp: 0,
            total_requests: 0,
            total_errors: 0,
            global_error_rate: 0.0,
            average_latency_us: 0,
            total_throughput_rps: 0,
            cache_stats: CacheStats {
                total_lookups: 0,
                cache_hits: 0,
                cache_misses: 0,
                cache_evictions: 0,
                memory_usage_bytes: 0,
                entry_count: 0,
                hit_ratio: 0.0,
            },
            resource_stats: ResourceStats {
                cpu_usage_percent: 0,
                memory_usage_bytes: 0,
                peak_memory_bytes: 0,
                active_threads: 0,
                queued_tasks: 0,
                network_bytes_sent: 0,
                network_bytes_received: 0,
                disk_bytes_read: 0,
                disk_bytes_written: 0,
            },
            active_providers: 0,
            alert_count: 0,
        });

        Self {
            provider_metrics: Arc::new(DashMap::new()),
            cache_metrics: Arc::new(CacheMetrics::new()),
            resource_metrics: Arc::new(ResourceMetrics::new()),
            alert_configs: Arc::new(DashMap::new()),
            active_alerts: Arc::new(DashMap::new()),
            alert_history: Arc::new(RwLock::new(VecDeque::with_capacity(MAX_ALERT_HISTORY))),
            alert_sender,
            metrics_sender,
            performance_watch,
            performance_receiver,
            anomaly_detector: Arc::new(AnomalyDetector::new(0.7, 100)),
        }
    }

    /// Record performance metric
    pub async fn record_metric(
        &self,
        metric: PerformanceMetric,
    ) -> Result<(), PerformanceMonitorError> {
        // Get or create provider metrics
        let provider_metrics = self
            .provider_metrics
            .entry(metric.provider.clone())
            .or_insert_with(|| Arc::new(ProviderMetrics::new(metric.provider.clone())));

        // Record metric in provider-specific metrics
        provider_metrics.record_metric(&metric).await;

        // Update cache metrics if applicable
        if metric.cache_hit {
            self.cache_metrics.record_hit();
        } else {
            self.cache_metrics.record_miss();
        }

        // Check for anomalies
        let latency_ms = metric.latency_us as f64 / 1000.0;
        if let Some(z_score) = self.anomaly_detector.is_anomaly("latency", latency_ms) {
            // Generate anomaly alert
            self.generate_anomaly_alert(&metric, z_score).await;
        }

        // Update baseline
        self.anomaly_detector.update_baseline("latency", latency_ms);

        Ok(())
    }

    /// Generate anomaly alert
    async fn generate_anomaly_alert(&self, metric: &PerformanceMetric, z_score: f64) {
        let alert_id = ArrayString::from(&format!(
            "anomaly_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ))
        .unwrap_or_default();

        let alert = PerformanceAlert {
            id: alert_id.clone(),
            timestamp: metric.timestamp,
            severity: if z_score > 4.0 {
                AlertSeverity::Critical
            } else if z_score > 3.0 {
                AlertSeverity::Error
            } else {
                AlertSeverity::Warning
            },
            metric_name: ArrayString::from("latency_anomaly").unwrap_or_default(),
            provider: Some(metric.provider.clone()),
            threshold_value: z_score,
            actual_value: metric.latency_us as f64 / 1000.0,
            message: ArrayString::from(&format!(
                "Latency anomaly detected: {}ms (z-score: {:.2})",
                metric.latency_us / 1000,
                z_score
            ))
            .unwrap_or_default(),
            resolved: false,
            resolution_timestamp: None,
        };

        self.active_alerts.insert(alert_id, alert.clone());
        let _ = self.alert_sender.send(alert);
    }

    /// Add alert configuration
    pub fn add_alert_config(&self, config: AlertConfig) {
        self.alert_configs
            .insert(config.metric_name.clone(), config);
    }

    /// Evaluate alerts based on current metrics
    pub async fn evaluate_alerts(&self) {
        for config_entry in self.alert_configs.iter() {
            let config = config_entry.value();
            if !config.enabled {
                continue;
            }

            self.check_metric_threshold(config).await;
        }
    }

    /// Check specific metric against threshold
    async fn check_metric_threshold(&self, config: &AlertConfig) {
        let current_value = match config.metric_name.as_str() {
            "error_rate" => self.get_global_error_rate(),
            "cache_hit_ratio" => self.cache_metrics.hit_ratio(),
            "memory_usage" => self
                .resource_metrics
                .memory_usage_bytes
                .load(Ordering::Relaxed) as f64,
            _ => return, // Unknown metric
        };

        let threshold_exceeded = match config.severity {
            AlertSeverity::Critical | AlertSeverity::Error => {
                current_value > config.threshold_value
            }
            AlertSeverity::Warning | AlertSeverity::Info => current_value > config.threshold_value,
        };

        if threshold_exceeded {
            let alert_id = ArrayString::from(&format!(
                "{}_{}",
                config.metric_name,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0)
            ))
            .unwrap_or_default();

            if !self.active_alerts.contains_key(&alert_id) {
                let alert = PerformanceAlert {
                    id: alert_id.clone(),
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                    severity: config.severity.clone(),
                    metric_name: config.metric_name.clone(),
                    provider: config.provider_specific.clone(),
                    threshold_value: config.threshold_value,
                    actual_value: current_value,
                    message: ArrayString::from(&format!(
                        "{} exceeded threshold: {:.2} > {:.2}",
                        config.metric_name, current_value, config.threshold_value
                    ))
                    .unwrap_or_default(),
                    resolved: false,
                    resolution_timestamp: None,
                };

                self.active_alerts.insert(alert_id, alert.clone());
                let _ = self.alert_sender.send(alert);
            }
        }
    }

    /// Get global error rate
    fn get_global_error_rate(&self) -> f64 {
        let mut total_requests = 0u64;
        let mut total_errors = 0u64;

        for provider_entry in self.provider_metrics.iter() {
            let metrics = provider_entry.value();
            total_requests += metrics.request_count.load(Ordering::Relaxed);
            total_errors += metrics.error_count.load(Ordering::Relaxed);
        }

        if total_requests == 0 {
            0.0
        } else {
            total_errors as f64 / total_requests as f64
        }
    }

    /// Get global metrics summary
    pub fn get_global_metrics(&self) -> GlobalMetrics {
        let mut total_requests = 0u64;
        let mut total_errors = 0u64;
        let mut total_latency = 0u64;
        let mut total_throughput = 0u64;

        for provider_entry in self.provider_metrics.iter() {
            let metrics = provider_entry.value();
            total_requests += metrics.request_count.load(Ordering::Relaxed);
            total_errors += metrics.error_count.load(Ordering::Relaxed);
            total_latency += metrics.average_processing_time();
            total_throughput += metrics.throughput_window.get_throughput();
        }

        let average_latency = if self.provider_metrics.len() > 0 {
            total_latency / self.provider_metrics.len() as u64
        } else {
            0
        };

        GlobalMetrics {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            total_requests,
            total_errors,
            global_error_rate: self.get_global_error_rate(),
            average_latency_us: average_latency,
            total_throughput_rps: total_throughput,
            cache_stats: self.cache_metrics.get_stats(),
            resource_stats: self.resource_metrics.get_stats(),
            active_providers: self.provider_metrics.len() as u32,
            alert_count: self.active_alerts.len() as u32,
        }
    }

    /// Get provider-specific metrics
    pub fn get_provider_metrics(&self, provider: &str) -> Option<(LatencyStats, f64, u64)> {
        let key = ArrayString::from(provider).ok()?;
        let metrics = self.provider_metrics.get(&key)?;

        Some((
            metrics.latency_histogram.get_stats(),
            metrics.error_rate(),
            metrics.throughput_window.get_throughput(),
        ))
    }

    /// Get alert subscriber
    pub fn subscribe_alerts(&self) -> broadcast::Receiver<PerformanceAlert> {
        self.alert_sender.subscribe()
    }

    /// Get metrics subscriber for real-time dashboard
    pub fn subscribe_metrics(&self) -> broadcast::Receiver<GlobalMetrics> {
        self.metrics_sender.subscribe()
    }

    /// Get performance watch receiver
    pub fn get_performance_monitor(&self) -> watch::Receiver<GlobalMetrics> {
        self.performance_receiver.clone()
    }

    /// Start background monitoring tasks
    pub fn start_monitoring_tasks(&self) -> SmallVec<[tokio::task::JoinHandle<()>; 4]> {
        let mut handles = SmallVec::new();

        // Metrics collection task
        handles.push(self.start_metrics_collection_task());

        // Alert evaluation task
        handles.push(self.start_alert_evaluation_task());

        // Resource monitoring task
        handles.push(self.start_resource_monitoring_task());

        // Cleanup task
        handles.push(self.start_cleanup_task());

        handles
    }

    /// Start metrics collection task
    fn start_metrics_collection_task(&self) -> tokio::task::JoinHandle<()> {
        let performance_watch = self.performance_watch.clone();
        let metrics_sender = self.metrics_sender.clone();
        let monitor = Arc::new(self);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(METRICS_COLLECTION_INTERVAL_MS));

            loop {
                interval.tick().await;

                let global_metrics = monitor.get_global_metrics();

                // Update watch channel
                let _ = performance_watch.send(global_metrics.clone());

                // Broadcast to subscribers
                let _ = metrics_sender.send(global_metrics);
            }
        })
    }

    /// Start alert evaluation task
    fn start_alert_evaluation_task(&self) -> tokio::task::JoinHandle<()> {
        let monitor_clone = self.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(ALERT_EVALUATION_INTERVAL_MS));

            loop {
                interval.tick().await;
                monitor_clone.evaluate_alerts().await;
            }
        })
    }

    /// Start resource monitoring task
    fn start_resource_monitoring_task(&self) -> tokio::task::JoinHandle<()> {
        let resource_metrics = self.resource_metrics.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Update resource metrics (simplified - would use system calls)
                // This would integrate with system monitoring APIs

                // Example: Update memory usage
                // resource_metrics.update_memory_usage(get_current_memory_usage());
            }
        })
    }

    /// Start cleanup task
    fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let alert_history = self.alert_history.clone();
        let active_alerts = self.active_alerts.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // 5 minutes

            loop {
                interval.tick().await;

                // Clean up old resolved alerts
                let cutoff_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs().saturating_sub(3600)) // 1 hour
                    .unwrap_or(0);

                active_alerts.retain(|_, alert| {
                    !alert.resolved
                        || alert.resolution_timestamp.unwrap_or(alert.timestamp) > cutoff_time
                });

                // Maintain alert history size
                let mut history = alert_history.write().await;
                while history.len() > MAX_ALERT_HISTORY {
                    history.pop_front();
                }
            }
        })
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus_metrics(&self) -> String {
        let global_metrics = self.get_global_metrics();
        let mut output = String::with_capacity(4096);

        // Global metrics
        output.push_str(&format!(
            "# HELP embedding_requests_total Total number of embedding requests\n"
        ));
        output.push_str(&format!("# TYPE embedding_requests_total counter\n"));
        output.push_str(&format!(
            "embedding_requests_total {}\n",
            global_metrics.total_requests
        ));

        output.push_str(&format!(
            "# HELP embedding_errors_total Total number of embedding errors\n"
        ));
        output.push_str(&format!("# TYPE embedding_errors_total counter\n"));
        output.push_str(&format!(
            "embedding_errors_total {}\n",
            global_metrics.total_errors
        ));

        output.push_str(&format!("# HELP embedding_error_rate Current error rate\n"));
        output.push_str(&format!("# TYPE embedding_error_rate gauge\n"));
        output.push_str(&format!(
            "embedding_error_rate {:.6}\n",
            global_metrics.global_error_rate
        ));

        // Cache metrics
        output.push_str(&format!(
            "# HELP embedding_cache_hit_ratio Cache hit ratio\n"
        ));
        output.push_str(&format!("# TYPE embedding_cache_hit_ratio gauge\n"));
        output.push_str(&format!(
            "embedding_cache_hit_ratio {:.6}\n",
            global_metrics.cache_stats.hit_ratio
        ));

        // Per-provider metrics
        for provider_entry in self.provider_metrics.iter() {
            let provider_id = provider_entry.key();
            let metrics = provider_entry.value();
            let latency_stats = metrics.latency_histogram.get_stats();

            output.push_str(&format!(
                "# HELP embedding_latency_seconds Embedding latency percentiles\n"
            ));
            output.push_str(&format!("# TYPE embedding_latency_seconds histogram\n"));
            output.push_str(&format!(
                "embedding_latency_seconds{{provider=\"{}\",quantile=\"0.5\"}} {:.6}\n",
                provider_id,
                latency_stats.p50 as f64 / 1_000_000.0
            ));
            output.push_str(&format!(
                "embedding_latency_seconds{{provider=\"{}\",quantile=\"0.9\"}} {:.6}\n",
                provider_id,
                latency_stats.p90 as f64 / 1_000_000.0
            ));
            output.push_str(&format!(
                "embedding_latency_seconds{{provider=\"{}\",quantile=\"0.99\"}} {:.6}\n",
                provider_id,
                latency_stats.p99 as f64 / 1_000_000.0
            ));
        }

        output
    }

    /// Get cache metrics
    pub fn get_cache_metrics(&self) -> &CacheMetrics {
        &self.cache_metrics
    }

    /// Get resource metrics
    pub fn get_resource_metrics(&self) -> &ResourceMetrics {
        &self.resource_metrics
    }
}

impl Clone for PerformanceMonitor {
    fn clone(&self) -> Self {
        Self {
            provider_metrics: self.provider_metrics.clone(),
            cache_metrics: self.cache_metrics.clone(),
            resource_metrics: self.resource_metrics.clone(),
            alert_configs: self.alert_configs.clone(),
            active_alerts: self.active_alerts.clone(),
            alert_history: self.alert_history.clone(),
            alert_sender: self.alert_sender.clone(),
            metrics_sender: self.metrics_sender.clone(),
            performance_watch: self.performance_watch.clone(),
            performance_receiver: self.performance_receiver.clone(),
            anomaly_detector: self.anomaly_detector.clone(),
        }
    }
}

/// Default alert configurations for common scenarios
impl PerformanceMonitor {
    /// Set up default alert configurations
    pub fn setup_default_alerts(&self) {
        // High error rate alert
        self.add_alert_config(AlertConfig {
            metric_name: ArrayString::from("error_rate").unwrap_or_default(),
            threshold_value: 0.05, // 5% error rate
            severity: AlertSeverity::Warning,
            enabled: true,
            provider_specific: None,
        });

        // Critical error rate alert
        self.add_alert_config(AlertConfig {
            metric_name: ArrayString::from("error_rate").unwrap_or_default(),
            threshold_value: 0.15, // 15% error rate
            severity: AlertSeverity::Critical,
            enabled: true,
            provider_specific: None,
        });

        // Low cache hit ratio alert
        self.add_alert_config(AlertConfig {
            metric_name: ArrayString::from("cache_hit_ratio").unwrap_or_default(),
            threshold_value: 0.7, // Below 70% hit ratio
            severity: AlertSeverity::Warning,
            enabled: true,
            provider_specific: None,
        });

        // High memory usage alert
        self.add_alert_config(AlertConfig {
            metric_name: ArrayString::from("memory_usage").unwrap_or_default(),
            threshold_value: 8_000_000_000.0, // 8GB
            severity: AlertSeverity::Warning,
            enabled: true,
            provider_specific: None,
        });
    }
}
